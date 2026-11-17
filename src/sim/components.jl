# Components of a model: genetic architectures, mutation and recombination.
#
# Genomes are matrices with one column per allele (alleles 2l - 1 and 2l are
# the paternal and maternal copies of locus l) and rows holding the allele's
# state: an effect vector (one row per trait) for additive and multilinear
# architectures; `y` followed by the column of B for `binary_b`; one scalar
# for `contribution_matrix`. Populations store genomes as arrays
# rows x alleles x individuals, as evomod2 does.

# ---- Architectures ---------------------------------------------------------------

abstract type Architecture end

"Additive: `x = sum_l Y_l`, with `Y_l` the sum of the two allele vectors of locus `l`."
struct Additive <: Architecture
    loci::Int
    traits::Int
end

"Pairwise multilinear epistasis with one tensor shared by all pairs of loci (Hansen & Wagner 2001)."
struct Multilinear <: Architecture
    loci::Int
    traits::Int
    epsilon::Array{Float64, 3}     # epsilon[k, a, b], symmetric in a, b
end

"Pairwise multilinear epistasis with a tensor for every pair of loci (Jones, Burger & Arnold 2014)."
struct MultilinearPairwise <: Architecture
    loci::Int
    traits::Int
    epsilon::Array{Float64, 5}     # epsilon[k, a, b, i, j], used for i < j
end

"Binary ontogenetic matrix (Melo & Marroig 2015): each allele has `y` and a 0/1 column of `B`, `x = B y`."
struct BinaryB <: Architecture
    loci::Int
    traits::Int
    init_b::Float64
    init_y::Symbol                 # :normal or :mutations
end

"Contribution matrix (do O & Whitlock 2023): `x = C a`, weight genes `a` and contribution genes `C` as loci."
struct ContributionMatrix <: Architecture
    traits::Int
    genes::Int
    init_weights::Vector{Float64}
    init_contributions::Matrix{Float64}
end

nloci(a::Architecture) = a.loci
nloci(a::ContributionMatrix) = a.genes + a.traits * a.genes
ntraits(a::Architecture) = a.traits
state_rows(a::Architecture) = a.traits
state_rows(a::BinaryB) = a.traits + 1
state_rows(::ContributionMatrix) = 1

"""
    additive(loci, traits)
    multilinear(loci, traits, epsilon)
    binary_b(loci, traits; init_b = 0.25, init_y = :normal)
    contribution_matrix(traits, genes = traits; init_weights, init_contributions)

Genetic architectures, as in evomod2. `epsilon` for `multilinear` is either a
`traits x traits x traits` array shared by all pairs of loci (symmetric in its
last two indices; see `random_epsilon`) or a `traits x traits x traits x loci
x loci` array with a tensor per pair of loci (see `pairwise_epsilon`).
"""
additive(loci::Integer, traits::Integer) = Additive(loci, traits)

function multilinear(loci::Integer, traits::Integer, epsilon::AbstractArray)
    p = traits
    if ndims(epsilon) == 5
        size(epsilon) == (p, p, p, loci, loci) ||
            throw(ArgumentError("pairwise epsilon must be $p x $p x $p x $loci x $loci"))
        return MultilinearPairwise(loci, traits, Array{Float64, 5}(epsilon))
    end
    size(epsilon) == (p, p, p) || throw(ArgumentError("epsilon must be $p x $p x $p"))
    all(k -> issymmetric(epsilon[k, :, :]), 1:p) ||
        throw(ArgumentError("epsilon[k, :, :] must be symmetric"))
    Multilinear(loci, traits, Array{Float64, 3}(epsilon))
end

function binary_b(loci::Integer, traits::Integer; init_b = 0.25, init_y = :normal)
    0 <= init_b <= 1 || throw(ArgumentError("init_b must be a probability"))
    init_y in (:normal, :mutations) || throw(ArgumentError("init_y must be :normal or :mutations"))
    BinaryB(loci, traits, init_b, init_y)
end

function contribution_matrix(traits::Integer, genes::Integer = traits;
                             init_weights = ones(genes),
                             init_contributions = Matrix{Float64}(I, traits, genes))
    length(init_weights) == genes || throw(ArgumentError("one initial weight per gene"))
    size(init_contributions) == (traits, genes) ||
        throw(ArgumentError("init_contributions must be $traits x $genes"))
    ContributionMatrix(traits, genes, collect(float.(init_weights)), Matrix{Float64}(init_contributions))
end

"Epistasis tensor with entries Normal(mean, sd), symmetrised in its last two indices."
function random_epsilon(traits::Integer; sd = 1.0, mean = 0.0)
    e = mean .+ sd .* randn(traits, traits, traits)
    for k in 1:traits
        e[k, :, :] = (e[k, :, :] + e[k, :, :]') / 2
    end
    e
end

"""
    pairwise_epsilon(loci, traits; sd = 1, mean = 0, within_trait = false, traits_with_epistasis = 1:traits)

A random tensor for every pair of loci `i < j` (zero for `i >= j`).
"""
function pairwise_epsilon(loci::Integer, traits::Integer; sd = 1.0, mean = 0.0,
                          within_trait = false, traits_with_epistasis = 1:traits)
    p = traits
    e = zeros(p, p, p, loci, loci)
    for i in 1:(loci - 1), j in (i + 1):loci
        t = mean .+ sd .* randn(p, p, p)
        if within_trait
            keep = zeros(p, p, p)
            for k in 1:p
                keep[k, k, k] = 1
            end
            t .*= keep
        end
        t[setdiff(1:p, traits_with_epistasis), :, :] .= 0
        e[:, :, :, i, j] = t
    end
    e
end

"Locus effects (rows x loci): the sum of the two allele columns of each locus."
function locus_effects(g::AbstractMatrix)
    L = size(g, 2) ÷ 2
    [g[r, 2l - 1] + g[r, 2l] for r in 1:size(g, 1), l in 1:L]
end

genotypic_value(::Additive, g::AbstractMatrix) = vec(sum(g, dims = 2))

function genotypic_value(a::Multilinear, g::AbstractMatrix)
    Y = locus_effects(g)
    S = vec(sum(Y, dims = 2))
    YY = Y * Y'
    [S[k] + (dot(S, view(a.epsilon, k, :, :), S) - dot(view(a.epsilon, k, :, :), YY)) / 2
     for k in 1:a.traits]
end

function genotypic_value(a::MultilinearPairwise, g::AbstractMatrix)
    Y = locus_effects(g)
    x = vec(sum(Y, dims = 2))
    p, L = a.traits, a.loci
    @inbounds for i in 1:(L - 1), j in (i + 1):L, k in 1:p
        t = 0.0
        for b in 1:p, c in 1:p
            t += a.epsilon[k, b, c, i, j] * Y[b, i] * Y[c, j]
        end
        x[k] += t
    end
    x
end

genotypic_value(a::BinaryB, g::AbstractMatrix) = g[2:end, :] * g[1, :]

function contributions(a::ContributionMatrix, g::AbstractMatrix)
    k, p = a.genes, a.traits
    w = [g[1, 2j - 1] + g[1, 2j] for j in 1:k]
    C = [g[1, 2(k + (j - 1) * p + i) - 1] + g[1, 2(k + (j - 1) * p + i)] for i in 1:p, j in 1:k]
    w, C
end

function genotypic_value(a::ContributionMatrix, g::AbstractMatrix)
    w, C = contributions(a, g)
    C * w
end

"""
    local_effects(arch, genome)

Effect of each locus in the background of the rest of the genome (traits x
loci): the change in genotypic value when the locus is set to the reference
genotype (all effects zero).
"""
local_effects(::Additive, g::AbstractMatrix) = locus_effects(g)

function local_effects(a::Multilinear, g::AbstractMatrix)
    Y = locus_effects(g)
    S = vec(sum(Y, dims = 2))
    reduce(hcat, [Y[:, i] + [dot(Y[:, i], view(a.epsilon, k, :, :), S - Y[:, i]) for k in 1:a.traits]
                  for i in 1:a.loci])
end

function local_effects(a::MultilinearPairwise, g::AbstractMatrix)
    Y = locus_effects(g)
    out = copy(Y)
    p, L = a.traits, a.loci
    for i in 1:(L - 1), j in (i + 1):L, k in 1:p
        t = sum(a.epsilon[k, b, c, i, j] * Y[b, i] * Y[c, j] for b in 1:p, c in 1:p)
        out[k, i] += t
        out[k, j] += t
    end
    out
end

function local_effects(a::BinaryB, g::AbstractMatrix)
    E = g[2:end, :] .* g[1, :]'
    locus_effects(E)
end

function local_effects(a::ContributionMatrix, g::AbstractMatrix)
    w, C = contributions(a, g)
    k, p = a.genes, a.traits
    L = zeros(p, k + p * k)
    for j in 1:k
        L[:, j] = C[:, j] * w[j]
        for i in 1:p
            L[i, k + (j - 1) * p + i] = C[i, j] * w[j]
        end
    end
    L
end

"Effect vector of each allele when genotypic values are sums of them; `nothing` otherwise."
allele_effects(::Architecture, g) = nothing
allele_effects(::Additive, g::AbstractMatrix) = g
allele_effects(::BinaryB, g::AbstractMatrix) = g[2:end, :] .* g[1, :]'

"Genomes of a new population (rows x alleles x n)."
function init_genomes(a::Architecture, n::Integer, init_sd::Real)
    init_sd .* randn(state_rows(a), 2 * nloci(a), n)
end

function init_genomes(a::BinaryB, n::Integer, init_sd::Real)
    m = 2 * a.loci
    g = zeros(a.traits + 1, m, n)
    for i in 1:n
        if a.init_y == :normal
            g[1, :, i] = init_sd .* randn(m)
        else   # as the C code: 2 x loci mutations scattered over the genome
            hits = zeros(Int, m)
            for _ in 1:m
                hits[rand(1:m)] += 1
            end
            g[1, :, i] = init_sd .* sqrt.(hits) .* randn(m)
        end
        g[2:end, :, i] = rand(a.traits, m) .< a.init_b
    end
    g
end

function init_genomes(a::ContributionMatrix, n::Integer, init_sd::Real)
    values = [a.init_weights; vec(a.init_contributions)]
    g = repeat(reshape(repeat(values ./ 2, inner = 2), 1, :), 1, 1, n)
    g .+ init_sd .* randn(size(g))
end

# ---- Mutation ----------------------------------------------------------------------

abstract type Mutation end

"Each allele mutates with probability `rate`; new effect `persistence * e + MvNormal(0, cov)`."
struct GaussianMutation <: Mutation
    rate::Float64
    cov::Matrix{Float64}
    chol::Matrix{Float64}          # lower Cholesky factor of cov
    persistence::Float64
end

"Binary B mutation: `y` mutates at `rate`, entries of `B` at `rate_b` (flip, or random value)."
struct BinaryBMutation <: Mutation
    rate::Float64
    sd::Float64
    rate_b::Float64
    random_state::Bool
    persistence::Float64
end

"Contribution matrix mutation: separate rates for weight and contribution genes."
struct ContributionMutation <: Mutation
    rate_weight::Float64
    rate_contribution::Float64
    sd::Float64
    weight_loci::Int
    persistence::Float64
end

check_persistence(p) = 0 <= p <= 1 || throw(ArgumentError("persistence must be between 0 and 1"))
check_rate(r) = 0 <= r <= 1 || throw(ArgumentError("mutation rates must be probabilities"))

"""
    mutation(rate, cov; persistence = 1)
    hoc_mutation(rate, cov)
    binary_b_mutation(rate, sd, rate_b; b_mutation = :flip, persistence = 1)
    contribution_mutation(rate_weight, rate_contribution, sd; persistence = 1)

Mutation models, as in evomod2. `cov` is a covariance matrix or a number `s`
(the standard deviation per trait). `persistence = 1` is a random walk, `0`
the house of cards model.
"""
mutation(rate, cov; persistence = 1.0) = (check_rate(rate); check_persistence(persistence);
    (rate = float(rate), cov = cov, persistence = float(persistence), type = :gaussian))
hoc_mutation(rate, cov) = mutation(rate, cov; persistence = 0.0)

function binary_b_mutation(rate, sd, rate_b; b_mutation = :flip, persistence = 1.0)
    check_rate(rate); check_rate(rate_b); check_persistence(persistence)
    b_mutation in (:flip, :random) || throw(ArgumentError("b_mutation must be :flip or :random"))
    BinaryBMutation(rate, sd, rate_b, b_mutation == :random, persistence)
end

function contribution_mutation(rate_weight, rate_contribution, sd; persistence = 1.0)
    check_rate(rate_weight); check_rate(rate_contribution); check_persistence(persistence)
    (rate_weight = float(rate_weight), rate_contribution = float(rate_contribution),
     sd = float(sd), persistence = float(persistence), type = :contribution)
end

# build the mutation component for an architecture (checks that they match)
function make_mutation(a::Architecture, spec)
    a isa BinaryB && throw(ArgumentError("the binary_b architecture needs binary_b_mutation()"))
    a isa ContributionMatrix && throw(ArgumentError("the contribution_matrix architecture needs contribution_mutation()"))
    (spec isa NamedTuple && spec.type == :gaussian) ||
        throw(ArgumentError("additive and multilinear architectures need mutation() or hoc_mutation()"))
    p = a.traits
    cov = spec.cov isa Number ? spec.cov^2 * Matrix{Float64}(I, p, p) : Matrix{Float64}(spec.cov)
    size(cov) == (p, p) && issymmetric(cov) || throw(ArgumentError("mutational cov must be a symmetric $p x $p matrix"))
    isposdef(cov) || throw(ArgumentError("mutational cov must be positive definite"))
    GaussianMutation(spec.rate, cov, Matrix(cholesky(Symmetric(cov)).L), spec.persistence)
end

make_mutation(a::BinaryB, spec::BinaryBMutation) = spec
make_mutation(a::BinaryB, spec) = throw(ArgumentError("the binary_b architecture needs binary_b_mutation()"))

function make_mutation(a::ContributionMatrix, spec)
    (spec isa NamedTuple && spec.type == :contribution) ||
        throw(ArgumentError("the contribution_matrix architecture needs contribution_mutation()"))
    ContributionMutation(spec.rate_weight, spec.rate_contribution, spec.sd, a.genes, spec.persistence)
end

randbinom(n, p) = p == 0 ? 0 : rand(Binomial(n, p))

function mutate!(m::GaussianMutation, g::AbstractMatrix)
    na = size(g, 2)
    for _ in 1:randbinom(na, m.rate)
        mutate_allele!(m, g, rand(1:na))
    end
end

mutate_allele!(m::GaussianMutation, g, a) =
    (g[:, a] .= m.persistence .* view(g, :, a) .+ m.chol * randn(size(g, 1)))
mutate_one!(m::GaussianMutation, g::AbstractMatrix) = mutate_allele!(m, g, rand(1:size(g, 2)))
expected_mutations(m::GaussianMutation, g::AbstractMatrix) = m.rate * size(g, 2)

function flip_b!(m::BinaryBMutation, g, pos)
    p = size(g, 1) - 1
    a, k = (pos - 1) ÷ p + 1, (pos - 1) % p + 2
    if !m.random_state || rand() < 0.5
        g[k, a] = 1 - g[k, a]
    end
end

function mutate!(m::BinaryBMutation, g::AbstractMatrix)
    na, p = size(g, 2), size(g, 1) - 1
    for _ in 1:randbinom(na, m.rate)
        a = rand(1:na)
        g[1, a] = m.persistence * g[1, a] + m.sd * randn()
    end
    for _ in 1:randbinom(na * p, m.rate_b)
        flip_b!(m, g, rand(1:(na * p)))
    end
end

function mutate_one!(m::BinaryBMutation, g::AbstractMatrix)
    na, p = size(g, 2), size(g, 1) - 1
    ey, eb = m.rate * na, m.rate_b * na * p
    if rand() * (ey + eb) < ey
        a = rand(1:na)
        g[1, a] = m.persistence * g[1, a] + m.sd * randn()
    else
        flip_b!(m, g, rand(1:(na * p)))
    end
end

expected_mutations(m::BinaryBMutation, g::AbstractMatrix) =
    m.rate * size(g, 2) + m.rate_b * size(g, 2) * (size(g, 1) - 1)

function mutate!(m::ContributionMutation, g::AbstractMatrix)
    w = 2 * m.weight_loci
    c = size(g, 2) - w
    for _ in 1:randbinom(w, m.rate_weight)
        a = rand(1:w)
        g[1, a] = m.persistence * g[1, a] + m.sd * randn()
    end
    for _ in 1:randbinom(c, m.rate_contribution)
        a = w + rand(1:c)
        g[1, a] = m.persistence * g[1, a] + m.sd * randn()
    end
end

function mutate_one!(m::ContributionMutation, g::AbstractMatrix)
    w = 2 * m.weight_loci
    c = size(g, 2) - w
    a = rand() * (m.rate_weight * w + m.rate_contribution * c) < m.rate_weight * w ?
        rand(1:w) : w + rand(1:c)
    g[1, a] = m.persistence * g[1, a] + m.sd * randn()
end

expected_mutations(m::ContributionMutation, g::AbstractMatrix) =
    m.rate_weight * 2 * m.weight_loci + m.rate_contribution * (size(g, 2) - 2 * m.weight_loci)

# ---- Recombination -------------------------------------------------------------------

abstract type Recombination end
struct FreeRecombination <: Recombination end
struct LinkageMap <: Recombination
    rates::Vector{Float64}         # crossover probability between loci l and l + 1
end

"""
    free_recombination()
    genetic_map(chromosomes, length)
    linkage(rates)

Recombination: unlinked loci (default); loci evenly spaced on `chromosomes`
chromosomes of `length` Morgans (Haldane map function); or explicit
recombination fractions between neighbouring loci.
"""
free_recombination() = FreeRecombination()
genetic_map(chromosomes::Integer, length::Real) = (type = :genetic_map, chromosomes = chromosomes, length = length)
function linkage(rates::AbstractVector)
    all(0 .<= rates .<= 0.5) || throw(ArgumentError("rates must be recombination fractions between 0 and 0.5"))
    (type = :linkage, rates = collect(float.(rates)))
end

haldane(d) = (1 - exp(-2d)) / 2

make_recombination(r::FreeRecombination, loci) = r
function make_recombination(r::NamedTuple, loci)
    if r.type == :linkage
        length(r.rates) == loci - 1 || throw(ArgumentError("rates must have $(loci - 1) values"))
        return LinkageMap(r.rates)
    end
    k = r.chromosomes
    k <= loci || throw(ArgumentError("more chromosomes than loci"))
    chrom = sort(repeat(1:k, outer = cld(loci, k))[1:loci])
    per = [count(==(c), chrom) for c in 1:k]
    LinkageMap([chrom[l] != chrom[l + 1] ? 0.5 : haldane(r.length / (per[chrom[l]] - 1))
                for l in 1:(loci - 1)])
end

function offspring!(::FreeRecombination, sire, dame, kid)
    for l in 1:(size(kid, 2) ÷ 2)
        kid[:, 2l - 1] .= view(sire, :, 2l - 2 + rand(1:2))
        kid[:, 2l] .= view(dame, :, 2l - 2 + rand(1:2))
    end
    kid
end

function gamete!(r::LinkageMap, parent, kid, slot)
    h = rand(0:1)
    for l in 1:(size(kid, 2) ÷ 2)
        if l > 1 && rand() < r.rates[l - 1]
            h = 1 - h
        end
        kid[:, 2l - 1 + slot] .= view(parent, :, 2l - 1 + h)
    end
end

offspring!(r::LinkageMap, sire, dame, kid) = (gamete!(r, sire, kid, 0); gamete!(r, dame, kid, 1); kid)
