"""
Antagonistic pleiotropy model.

Like `BinaryB`, but entries of the ontogenetic matrix `b` are -1, 0 or 1, so an
allele can push traits in opposite directions, and there is no environmental
noise (`z = b * y`). Mutation of `b` sets a 0 entry to -1 or 1 and a non-zero
entry to 0. Allelic effect vectors are classified as Modular, Antagonistic,
Local, Integrated or Neutral (`classifyBj`, defined for 4 traits).
"""
module Antagonistic

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
       classifyBj, countBclass

mutable struct Individual
    m::Int            # number of alleles (2 x loci)
    p::Int            # number of traits
    mu::Float64       # mutation rate of allelic effects, per allele
    mu_b::Float64     # mutation rate of b, per entry
    mu_sigma::Float64 # standard deviation of mutational effects
    y::Vector{Float64}
    b::Matrix{Int}
    z::Vector{Float64}
    Individual(m, p, mu, mu_b, mu_sigma) = new(2 * m, p, mu, mu_b, mu_sigma)
end

"""
    Individual(m, p, mu, mu_b, mu_sigma)

Parameter holder (`base_ind`) for `m` diploid loci and `p` traits. If the
`b` field of the base individual is set, random individuals start with a copy
of it; otherwise `b` is drawn uniformly from {-1, 0, 1}.
"""
Individual

function RandomInd!(ind::Individual, base_ind::Individual)
    ind.y = rand(Normal(0, base_ind.mu_sigma), ind.m)
    ind.b = isdefined(base_ind, :b) ? copy(base_ind.b) : rand(-1:1, ind.p, ind.m)
    ind.z = ind.b * ind.y
    ind
end

function RandomInd(base_ind::Individual)
    new_ind = Individual(base_ind.m ÷ 2, base_ind.p, base_ind.mu, base_ind.mu_b,
                         base_ind.mu_sigma)
    RandomInd!(new_ind, base_ind)
end

include("generations.jl")

function mutation!(ind::Individual, bin_y, bin_b)
    for _ in 1:rand(bin_y)
        i = rand(1:ind.m)
        ind.y[i] += rand(Normal(0, ind.mu_sigma))
    end
    for _ in 1:rand(bin_b)
        i = rand(1:ind.p)
        j = rand(1:ind.m)
        ind.b[i, j] = ind.b[i, j] == 0 ? rand((-1, 1)) : 0
    end
end

function mutation!(pop::Population)
    bin_y = Binomial(pop.base_ind.m, pop.base_ind.mu)
    bin_b = Binomial(pop.base_ind.m * pop.base_ind.p, pop.base_ind.mu_b)
    for ind in pop.pop
        mutation!(ind, bin_y, bin_b)
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
    mul!(new_ind.z, new_ind.b, new_ind.y)
    new_ind
end

const B_CLASSES = ["Modular", "Antagonistic", "Local", "Integrated", "Neutral", "Other"]

"""
    classifyBj(b_j)

Class of the effect vector `b_j` (a column of `b`, 4 traits, modules 1:2 and
3:4). A matrix is classified column by column.
"""
function classifyBj(b_j::AbstractVector{<:Integer})
    length(b_j) == 4 || throw(ArgumentError("classifyBj is defined for 4 traits"))
    b_j = collect(b_j)
    a = abs.(b_j)
    if a == [1, 1, 0, 0] || a == [0, 0, 1, 1]
        "Modular"
    elseif b_j in ([1, 1, -1, -1], [-1, -1, 1, 1],
                   [-1, 1, 1, 1], [1, -1, 1, 1], [1, 1, -1, 1], [1, 1, 1, -1],
                   [1, -1, -1, -1], [-1, 1, -1, -1], [-1, -1, 1, -1], [-1, -1, -1, 1],
                   [0, 1, -1, -1], [1, 0, -1, -1], [1, 1, 0, -1], [1, 1, -1, 0],
                   [0, -1, 1, 1], [-1, 0, 1, 1], [-1, -1, 0, 1], [-1, -1, 1, 0],
                   [0, -1, 0, 1], [0, 1, 0, -1], [-1, 0, 1, 0], [1, 0, -1, 0],
                   [0, -1, 1, 0], [-1, 0, 0, 1], [1, 0, 0, -1], [0, 1, -1, 0])
        "Antagonistic"
    elseif sum(a) == 1
        "Local"
    elseif b_j in ([1, 1, 1, 1], [-1, -1, -1, -1],
                   [0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0],
                   [0, -1, -1, -1], [-1, 0, -1, -1], [-1, -1, 0, -1], [-1, -1, -1, 0],
                   [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 1, 0], [1, 0, 0, 1],
                   [0, -1, 0, -1], [-1, 0, -1, 0], [0, -1, -1, 0], [-1, 0, 0, -1])
        "Integrated"
    elseif b_j == [0, 0, 0, 0]
        "Neutral"
    else
        "Other"
    end
end

classifyBj(b::AbstractMatrix{<:Integer}) = [classifyBj(b[:, j]) for j in 1:size(b, 2)]

"Number of effect vectors in each class, as a DataFrame."
function countBclass(b_class)
    counts = countmap(b_class)
    DataFrame(class = B_CLASSES, count = [get(counts, c, 0) for c in B_CLASSES])
end

function moments(pop::Population)
    ys = reduce(vcat, (ind.y' for ind in pop.pop))
    zs = reduce(vcat, (ind.z' for ind in pop.pop))
    mean_b = sum(ind.b for ind in pop.pop) / pop.n
    count_b = countBclass(classifyBj(round.(Int, mean_b)))
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
