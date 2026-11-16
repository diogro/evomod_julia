"""
Pleiotropic vector model, with optional pairwise epistasis.

Each allele `i` has an effect size `y[i]` and a unit direction `u_i` in trait
space; column `i` of `b` stores `u_i * y[i]`. With no epistasis the phenotype
is the sum of the columns, `z = sum(b, dims = 2)`. A mutation hits a whole
locus (rate `mu` per allele): `y[i]` gets `Normal(0, mu_sigma)` added and the
direction becomes `normalize(u_i + Normal(0, mu_sigma_b))`.

With an epistasis array `e` (alleles x alleles x traits, symmetric in its first
two dimensions, see `random_epistasis`), trait `k` also gets
`sum(e[i, j, k] * b[k, i] * b[k, j] for i > j)`.
"""
module PleiotropicVectors

using DataFrames
using Distributions
using LinearAlgebra
using LogExpFunctions: logsumexp
using ProgressMeter
using Random
using Statistics
using StatsBase: countmap
using ..Common

export Individual, Population, RandomInd, RandomPop!, run_pop, next_generation!,
       fitness!, mutation!, cross!, changeSurface!, moments, moments!,
       random_epistasis, phenotype, classifyBj, countBclass, directional_matrices

mutable struct Individual
    m::Int              # number of alleles (2 x loci)
    p::Int              # number of traits
    mu::Float64         # mutation rate, per allele
    mu_sigma::Float64   # standard deviation of mutations in effect size
    mu_sigma_b::Float64 # standard deviation of mutations in effect direction
    e::Union{Nothing, Array{Float64, 3}}  # epistatic coefficients, or nothing
    y::Vector{Float64}
    b::Matrix{Float64}
    z::Vector{Float64}
    function Individual(m, p, mu, mu_sigma, mu_sigma_b, e = nothing)
        if e !== nothing
            size(e) == (2m, 2m, p) ||
                throw(ArgumentError("e must be (2m, 2m, p) = $((2m, 2m, p))"))
            all(k -> issymmetric(e[:, :, k]), 1:p) ||
                throw(ArgumentError("e[:, :, k] must be symmetric; see random_epistasis"))
        end
        new(2 * m, p, mu, mu_sigma, mu_sigma_b, e)
    end
end

"""
    Individual(m, p, mu, mu_sigma, mu_sigma_b, e = nothing)

Parameter holder (`base_ind`) for `m` diploid loci and `p` traits, optionally
with an epistasis array `e`, shared by every individual in the population.
"""
Individual

"""
    random_epistasis(m, p, sigma_e)

Epistatic coefficients for `m` loci (2m alleles) and `p` traits: normal with
standard deviation `sigma_e`, symmetrised as `(e + e') / 2` for each trait.
"""
function random_epistasis(m, p, sigma_e)
    e = rand(Normal(0, sigma_e), 2m, 2m, p)
    for k in 1:p
        e[:, :, k] = (e[:, :, k] + e[:, :, k]') / 2
    end
    e
end

"Phenotype from the effect matrix `b` and epistasis `e` (or `nothing`)."
phenotype(b, ::Nothing) = vec(sum(b, dims = 2))

function phenotype(b, e::Array{Float64, 3})
    z = vec(sum(b, dims = 2))
    for k in eachindex(z)
        v = b[k, :]
        E = @view e[:, :, k]
        # sum over i > j of E[i, j] v[i] v[j], using the symmetry of E
        z[k] += (dot(v, E * v) - sum(diag(E) .* v .^ 2)) / 2
    end
    z
end

function RandomInd!(ind::Individual, base_ind::Individual)
    ind.y = rand(Normal(0, base_ind.mu_sigma), ind.m)
    ind.b = rand(Normal(0, base_ind.mu_sigma_b), ind.p, ind.m)
    for i in 1:ind.m
        ind.b[:, i] = normalize(ind.b[:, i]) * ind.y[i]
    end
    ind.z = phenotype(ind.b, ind.e)
    ind
end

function RandomInd(base_ind::Individual)
    new_ind = Individual(base_ind.m ÷ 2, base_ind.p, base_ind.mu, base_ind.mu_sigma,
                         base_ind.mu_sigma_b, base_ind.e)
    RandomInd!(new_ind, base_ind)
end

include("generations.jl")

# The epistasis array is a model parameter shared by all individuals: keep it
# shared when populations are deep-copied.
function Base.deepcopy_internal(ind::Individual, dict::IdDict)
    haskey(dict, ind) && return dict[ind]
    new_ind = Individual(ind.m ÷ 2, ind.p, ind.mu, ind.mu_sigma, ind.mu_sigma_b, nothing)
    new_ind.e = ind.e
    isdefined(ind, :y) && (new_ind.y = copy(ind.y))
    isdefined(ind, :b) && (new_ind.b = copy(ind.b))
    isdefined(ind, :z) && (new_ind.z = copy(ind.z))
    dict[ind] = new_ind
end

function mutation!(ind::Individual, bin_locus)
    n_mut = rand(bin_locus)
    n_mut == 0 && return
    norm_sigma_y = Normal(0, ind.mu_sigma)
    norm_sigma_b = Normal(0, ind.mu_sigma_b)
    for _ in 1:n_mut
        i = rand(1:ind.m)
        u = ind.b[:, i] / ind.y[i]  # unit pleiotropic direction (b = u * y)
        ind.y[i] += rand(norm_sigma_y)
        ind.b[:, i] = normalize(u + rand(norm_sigma_b, ind.p)) * ind.y[i]
    end
end

function mutation!(pop::Population)
    bin_locus = Binomial(pop.base_ind.m, pop.base_ind.mu)
    for ind in pop.pop
        mutation!(ind, bin_locus)
    end
end

"Free recombination; see `BinaryB.cross!`."
function cross!(ind_1::Individual, ind_2::Individual, new_ind::Individual,
                alele_1::Vector{Int}, alele_2::Vector{Int})
    rand!(alele_1, 0:1)
    rand!(alele_2, 0:1)
    for l in 1:(ind_1.m ÷ 2)
        a = 2l - 1
        new_ind.y[a]     = ind_1.y[a + alele_1[l]]
        new_ind.y[a + 1] = ind_2.y[a + alele_2[l]]
        for j in 1:ind_1.p
            new_ind.b[j, a]     = ind_1.b[j, a + alele_1[l]]
            new_ind.b[j, a + 1] = ind_2.b[j, a + alele_2[l]]
        end
    end
    new_ind.z = phenotype(new_ind.b, new_ind.e)
    new_ind
end

# Reference directions for classifying effect vectors (4 traits, modules 1:2 and 3:4)
const directional_matrices = [
    "Local" => Float64[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1;
                       -1 0 0 0; 0 -1 0 0; 0 0 -1 0; 0 0 0 -1],
    "Modular" => Float64[1 1 0 0; -1 -1 0 0; 0 0 1 1; 0 0 -1 -1;
                         1 0.5 0 0; 0.5 1 0 0; -0.5 -1 0 0; -1 -0.5 0 0;
                         0 0 0.5 1; 0 0 1 0.5; 0 0 -0.5 -1; 0 0 -1 -0.5],
    "Antagonistic" => Float64[1 1 0 -1; 1 1 -1 0; 1 1 -1 -1; 1 0 0 -1; 1 0 -1 0;
                              1 0 -1 -1; 0 1 0 -1; 0 1 -1 0; 0 1 -1 -1; 0 -1 1 1;
                              0 -1 1 0; 0 -1 0 1; -1 0 1 1; -1 0 1 0; -1 0 0 1;
                              -1 -1 1 1; -1 -1 1 0],
    "Integrated" => Float64[1 1 1 1; 1 1 1 0; 1 1 0 1; 1 0 1 1; -1 -1 -1 -1;
                            0 1 1 1; 0 -1 -1 -1; -1 0 -1 -1; -1 -1 0 -1; -1 -1 -1 0;
                            0 1 0 1; 1 0 1 0; 0 1 1 0; 1 0 0 1; 0 -1 0 -1;
                            -1 0 -1 0; 0 -1 -1 0; -1 0 0 -1],
    "Intra module" => Float64[1 1 1 -1; 1 1 -1 1; 1 0 1 -1; 1 0 -1 1; 1 -1 1 1;
                              1 -1 1 0; 1 -1 1 -1; 1 -1 0 1; 1 -1 0 0; 1 -1 0 -1;
                              1 -1 -1 1; 1 -1 -1 0; 1 -1 -1 -1; 0 1 1 -1; 0 1 -1 1;
                              0 0 1 -1; 0 0 -1 1; 0 -1 -1 1; -1 1 1 1; -1 1 1 0;
                              -1 1 1 -1; -1 1 0 1; -1 1 0 0; -1 1 0 -1; -1 1 -1 1;
                              -1 1 -1 0; -1 1 -1 -1; -1 0 1 -1; -1 0 -1 1; -1 -1 -1 1],
]

const B_CLASSES = first.(directional_matrices)

vectorCor(x, y) = dot(normalize(x), normalize(y))

"Highest cosine between `b_j` and any row of `current_matrix` (at least 0)."
function compareRows(b_j, current_matrix)
    max_corr = 0.0
    for i in 1:size(current_matrix, 1)
        max_corr = max(max_corr, vectorCor(b_j, current_matrix[i, :]))
    end
    max_corr
end

"""
    classifyBj(b_j)

Class whose reference directions are closest (by cosine) to the effect vector
`b_j`; "Neutral" if no reference direction has a positive cosine. A matrix is
classified column by column.
"""
function classifyBj(b_j::AbstractVector{<:Real})
    length(b_j) == 4 || throw(ArgumentError("classifyBj is defined for 4 traits"))
    max_corr = 0.0
    key = "Neutral"
    for (class, mat) in directional_matrices
        corr = compareRows(b_j, mat)
        if corr > max_corr
            key = class
            max_corr = corr
        end
    end
    key
end

classifyBj(b::AbstractMatrix{<:Real}) = [classifyBj(b[:, j]) for j in 1:size(b, 2)]

"Number of effect vectors in each class, as a DataFrame."
function countBclass(b_class)
    counts = countmap(b_class)
    DataFrame(class = B_CLASSES, count = [get(counts, c, 0) for c in B_CLASSES])
end

function moments(pop::Population)
    ys = reduce(vcat, (ind.y' for ind in pop.pop))
    zs = reduce(vcat, (ind.z' for ind in pop.pop))
    mean_b = sum(ind.b for ind in pop.pop) / pop.n
    count_b = countBclass(classifyBj(mean_b))
    count_b[!, :gen] .= pop.generation
    Dict("mean_y" => vec(mean(ys, dims = 1)),
         "mean_b" => mean_b,
         "mean_z" => vec(mean(zs, dims = 1)),
         "zs" => zs,
         "theta" => copy(mean(pop.surface)),
         "gen" => pop.generation,
         "G" => cov(zs),
         "corrG" => cor(zs),
         "count_b" => count_b)
end

end
