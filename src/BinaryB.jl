"""
Binary ontogenetic matrix model (closest to the C model of Melo & Marroig 2015).

Each allele has an effect `y`; the binary matrix `b` (traits x alleles) says
which traits each allele affects. Additive values are `x = b * y` and the
phenotype is `z = x + e`, with `e ~ Normal(0, v_e)` (`v_e` is a standard
deviation, as in the C code). Mutation adds `Normal(0, mu_sigma)` to allelic
effects (rate `mu` per allele) and flips entries of `b` (rate `mu_b` per entry).
"""
module BinaryB

using Distributions
using LinearAlgebra
using LogExpFunctions: logsumexp
using ProgressMeter
using Random
using Statistics
using ..Common

export Individual, Population, RandomInd, RandomPop!, run_pop, next_generation!,
       fitness!, mutation!, cross!, changeSurface!, moments, moments!

mutable struct Individual
    m::Int            # number of alleles (2 x loci)
    p::Int            # number of traits
    v_e::Float64      # environmental standard deviation
    mu::Float64       # mutation rate of allelic effects, per allele
    mu_b::Float64     # mutation rate of b, per entry
    mu_sigma::Float64 # standard deviation of mutational effects
    y::Vector{Float64}
    b::Matrix{Int}
    x::Vector{Float64}
    z::Vector{Float64}
    Individual(m, p, v_e, mu, mu_b, mu_sigma) = new(2 * m, p, v_e, mu, mu_b, mu_sigma)
end

"""
    Individual(m, p, v_e, mu, mu_b, mu_sigma)

Parameter holder (`base_ind`) for `m` diploid loci and `p` traits. Use
`RandomInd` to create an individual with a genotype.
"""
Individual

function RandomInd!(ind::Individual, y_mean::Float64 = 0.0, g_sigma::Float64 = ind.mu_sigma)
    ind.y = rand(Normal(y_mean, g_sigma), ind.m)
    ind.b = rand(0:1, ind.p, ind.m)
    ind.x = ind.b * ind.y
    ind.z = ind.x + rand(Normal(0, ind.v_e), ind.p)
    ind
end

function RandomInd(base_ind::Individual, y_mean::Float64 = 0.0,
                   g_sigma::Float64 = base_ind.mu_sigma)
    new_ind = Individual(base_ind.m ÷ 2, base_ind.p, base_ind.v_e, base_ind.mu,
                         base_ind.mu_b, base_ind.mu_sigma)
    RandomInd!(new_ind, y_mean, g_sigma)
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
        ind.b[i, j] = 1 - ind.b[i, j]
    end
end

function mutation!(pop::Population)
    bin_y = Binomial(pop.base_ind.m, pop.base_ind.mu)
    bin_b = Binomial(pop.base_ind.m * pop.base_ind.p, pop.base_ind.mu_b)
    for ind in pop.pop
        mutation!(ind, bin_y, bin_b)
    end
end

"""
    cross!(ind_1, ind_2, new_ind, alele_1, alele_2)

Free recombination: alleles `2l - 1` and `2l` of locus `l` in the offspring
are a random allele of that locus from `ind_1` and from `ind_2`.
"""
function cross!(ind_1::Individual, ind_2::Individual, new_ind::Individual,
                alele_1::Vector{Int}, alele_2::Vector{Int})
    rand!(alele_1, 0:1)
    rand!(alele_2, 0:1)
    for l in 1:(ind_1.m ÷ 2)
        first = 2l - 1
        new_ind.y[first]     = ind_1.y[first + alele_1[l]]
        new_ind.y[first + 1] = ind_2.y[first + alele_2[l]]
        for j in 1:ind_1.p
            new_ind.b[j, first]     = ind_1.b[j, first + alele_1[l]]
            new_ind.b[j, first + 1] = ind_2.b[j, first + alele_2[l]]
        end
    end
    mul!(new_ind.x, new_ind.b, new_ind.y)
    new_ind.z .= new_ind.x .+ rand(Normal(0, new_ind.v_e), new_ind.p)
    new_ind
end

function moments(pop::Population)
    ys = reduce(vcat, (ind.y' for ind in pop.pop))
    xs = reduce(vcat, (ind.x' for ind in pop.pop))
    zs = reduce(vcat, (ind.z' for ind in pop.pop))
    mean_b = sum(ind.b for ind in pop.pop) / pop.n
    P = cov(zs)
    G = cov(xs)
    Dict("mean_y" => vec(mean(ys, dims = 1)),
         "mean_b" => mean_b,
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
