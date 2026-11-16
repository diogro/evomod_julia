"""
Additive and multilinear-epistasis model with multivariate allelic effects.

Each allele carries an effect vector in trait space (column of `y`, traits x
alleles). Loci are additive within themselves: the effect of locus `i` is the
sum of its two allele vectors, `Y_i`. Genotypic values are

    x = S + sum_{i < j} ε(Y_i, Y_j),    S = sum_i Y_i,

where `ε` is a bilinear map given by a `p x p x p` tensor,
`ε(a, b)[k] = sum_{l, m} ε[k, l, m] a[l] b[m]`, shared by all pairs of loci
(pairwise multilinear epistasis in the sense of Hansen & Wagner 2001).
`ε[k, :, :]` must be symmetric. Without `ε` (`epsilon = nothing`) the model is
purely additive. Effects are measured from the reference genotype `y = 0`.
The phenotype is `z = x + e`, `e ~ Normal(0, v_e)` (standard deviation).

Mutation hits each allele with probability `mu` and adds a draw from
`MvNormal(0, mutation_cov)` to its effect vector.
"""
module Multilinear

using Distributions
using LinearAlgebra
using LogExpFunctions: logsumexp
using ProgressMeter
using Random
using Statistics
using ..Common

export Params, Individual, Population, RandomInd, RandomPop!, run_pop, next_generation!,
       fitness!, mutation!, cross!, changeSurface!, moments, moments!,
       genotypic_value, locus_effects, epistatic_term, random_epsilon, local_effects

"""
    Params(loci, p; mu, mutation_cov, v_e = 0.0, epsilon = nothing, init_sigma = 0.0)

Model parameters, shared by every individual (never copied).

- `loci`, `p`: number of diploid loci and of traits
- `mu`: mutation probability per allele per generation
- `mutation_cov`: covariance matrix of mutational effects (`p x p`), or a
  number `s` for `s^2 * I`
- `v_e`: environmental standard deviation
- `epsilon`: `p x p x p` epistasis tensor, or `nothing` for the additive model
- `init_sigma`: standard deviation of allelic effects in the initial
  population (0 starts every individual at the reference genotype)
"""
struct Params
    loci::Int
    p::Int
    mu::Float64
    mutation_cov::Matrix{Float64}
    v_e::Float64
    epsilon::Union{Nothing, Array{Float64, 3}}
    init_sigma::Float64
    function Params(loci, p; mu, mutation_cov, v_e = 0.0, epsilon = nothing, init_sigma = 0.0)
        M = mutation_cov isa Real ? mutation_cov^2 * Matrix(1.0I, p, p) : Matrix{Float64}(mutation_cov)
        size(M) == (p, p) || throw(ArgumentError("mutation_cov must be $p x $p"))
        isposdef(Symmetric(M)) || throw(ArgumentError("mutation_cov must be positive definite"))
        if epsilon !== nothing
            size(epsilon) == (p, p, p) ||
                throw(ArgumentError("epsilon must be a $p x $p x $p array"))
            all(k -> issymmetric(epsilon[k, :, :]), 1:p) ||
                throw(ArgumentError("epsilon[k, :, :] must be symmetric for every trait k"))
            epsilon = Array{Float64, 3}(epsilon)
        end
        new(loci, p, mu, Symmetric(M) |> Matrix, v_e, epsilon, init_sigma)
    end
end

# parameters are shared, never duplicated by deepcopy
Base.deepcopy_internal(par::Params, ::IdDict) = par

mutable struct Individual
    m::Int            # number of alleles (2 x loci)
    p::Int            # number of traits
    par::Params
    y::Matrix{Float64}   # allelic effect vectors, traits x alleles
    x::Vector{Float64}   # genotypic value
    z::Vector{Float64}   # phenotype
end

"""
    Individual(par::Params)

Individual at the reference genotype (all effects zero); used as `base_ind`.
"""
function Individual(par::Params)
    m = 2 * par.loci
    ind = Individual(m, par.p, par, zeros(par.p, m), zeros(par.p), zeros(par.p))
    ind
end

"""
    random_epsilon(p; sd = 1.0, mean = 0.0)

Epistasis tensor with entries `Normal(mean, sd)`, symmetrised in its last two
indices. A positive `mean` gives positive directional epistasis on average.
"""
function random_epsilon(p; sd = 1.0, mean = 0.0)
    e = rand(Normal(mean, sd), p, p, p)
    for k in 1:p
        e[k, :, :] = (e[k, :, :] + e[k, :, :]') / 2
    end
    e
end

"Locus effects `Y` (traits x loci): sum of the two allele vectors of each locus."
locus_effects(y::AbstractMatrix) = y[:, 1:2:end] .+ y[:, 2:2:end]

"`ε(a, b)` for the tensor `epsilon`."
bilinear(epsilon, a, b) = [dot(a, view(epsilon, k, :, :), b) for k in axes(epsilon, 1)]

"""
    epistatic_term(epsilon, Y)

`sum_{i < j} ε(Y_i, Y_j)` for the locus effects `Y` (traits x loci), computed
as `(ε(S, S) - sum_i ε(Y_i, Y_i)) / 2` with `S = sum_i Y_i`.
"""
function epistatic_term(epsilon::Array{Float64, 3}, Y::AbstractMatrix)
    S = vec(sum(Y, dims = 2))
    YY = Y * Y'
    [(dot(S, view(epsilon, k, :, :), S) - dot(view(epsilon, k, :, :), YY)) / 2
     for k in axes(epsilon, 1)]
end

"Genotypic value of the allelic effects `y` under parameters `par`."
function genotypic_value(par::Params, y::AbstractMatrix)
    Y = locus_effects(y)
    x = vec(sum(Y, dims = 2))
    par.epsilon === nothing ? x : x + epistatic_term(par.epsilon, Y)
end

"""
    local_effects(par, y)

Effect of each locus in the background of the others (traits x loci):
`Y_i + ε(Y_i, S - Y_i)`, the change in genotypic value from replacing locus
`i` with the reference genotype. Equal to `locus_effects(y)` without epistasis.
"""
function local_effects(par::Params, y::AbstractMatrix)
    Y = locus_effects(y)
    par.epsilon === nothing && return Y
    S = vec(sum(Y, dims = 2))
    reduce(hcat, (Y[:, i] + bilinear(par.epsilon, Y[:, i], S - Y[:, i]) for i in axes(Y, 2)))
end

function set_phenotype!(ind::Individual)
    ind.x = genotypic_value(ind.par, ind.y)
    ind.z = ind.x .+ ind.par.v_e .* randn(ind.p)
    ind
end

function RandomInd(base_ind::Individual)
    par = base_ind.par
    ind = Individual(par)
    ind.y .= par.init_sigma .* randn(par.p, ind.m)
    set_phenotype!(ind)
end

include("generations.jl")

"""
    Population(n, par::Params)

Population of `n` individuals with parameters `par`; fill it with
`RandomPop!` or start it with `run_pop(par, n, ...)`.
"""
Population(n::Integer, par::Params) = Population(n, Individual(par))

run_pop(par::Params, n_e::Integer, args...; kwargs...) =
    run_pop(Individual(par), n_e, args...; kwargs...)

function mutation!(ind::Individual, bin, mut)
    for _ in 1:rand(bin)
        a = rand(1:ind.m)
        ind.y[:, a] .+= rand(mut)
    end
end

function mutation!(pop::Population)
    par = pop.base_ind.par
    bin = Binomial(pop.base_ind.m, par.mu)
    mut = MvNormal(zeros(par.p), par.mutation_cov)
    for ind in pop.pop
        mutation!(ind, bin, mut)
    end
end

"Free recombination; see `BinaryB.cross!`."
function cross!(ind_1::Individual, ind_2::Individual, new_ind::Individual,
                alele_1::Vector{Int}, alele_2::Vector{Int})
    rand!(alele_1, 0:1)
    rand!(alele_2, 0:1)
    for l in 1:(ind_1.m ÷ 2)
        a = 2l - 1
        new_ind.y[:, a]     .= view(ind_1.y, :, a + alele_1[l])
        new_ind.y[:, a + 1] .= view(ind_2.y, :, a + alele_2[l])
    end
    set_phenotype!(new_ind)
end

function moments(pop::Population)
    xs = reduce(vcat, (ind.x' for ind in pop.pop))
    zs = reduce(vcat, (ind.z' for ind in pop.pop))
    mean_y = sum(ind.y for ind in pop.pop) / pop.n
    P = cov(zs)
    G = cov(xs)
    Dict("mean_y" => mean_y,
         "mean_x" => vec(mean(xs, dims = 1)),
         "mean_z" => vec(mean(zs, dims = 1)),
         "zs" => zs,
         "theta" => copy(mean(pop.surface)),
         "gen" => pop.generation,
         "P" => P,
         "G" => G,
         "h2" => diag(G) ./ diag(P),
         "corrG" => cor(xs),
         "corrP" => cor(zs))
end

end
