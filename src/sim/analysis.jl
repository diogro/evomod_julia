# Recorded statistics and analysis tools, as in evomod2.

trait_names(p) = ["trait$i" for i in 1:p]

"""
    trait_means(pop)

Mean genotypic value (`mean_x`), mean phenotype (`mean_z`) and optimum
(`theta`) of each trait at every recorded generation (long format).
"""
function trait_means(pop::Population)
    r, p = pop.records, ntraits(pop.model)
    DataFrame(generation = repeat(r.generation, inner = p),
              trait = repeat(trait_names(p), length(r.generation)),
              mean_x = reduce(vcat, r.mean_x), mean_z = reduce(vcat, r.mean_z),
              theta = reduce(vcat, r.theta))
end

"Additive genetic (`G`) and phenotypic (`P`) variances and heritability (`h2`) of each trait."
function variances(pop::Population)
    r, p = pop.records, ntraits(pop.model)
    g = reduce(vcat, diag.(r.G))
    pp = reduce(vcat, diag.(r.P))
    DataFrame(generation = repeat(r.generation, inner = p),
              trait = repeat(trait_names(p), length(r.generation)),
              G = g, P = pp, h2 = g ./ pp)
end

"Mean fitness on the surface of the current regime at every recorded generation (NaN under drift)."
mean_fitness(pop::Population) = DataFrame(generation = pop.records.generation,
                                          mean_fitness = pop.records.mean_fitness)

"""
    g_matrices(pop; component = :total)

G matrices by recorded generation (a `Dict`). `component = :genic` or `:ld`
for the genic and linkage disequilibrium parts (architectures whose genotypic
values are sums of allele effects).
"""
function g_matrices(pop::Population; component = :total)
    r = pop.records
    if component != :total && isempty(r.G_genic)
        throw(ArgumentError("the genic part of G is only defined for architectures where genotypic " *
                            "values are sums of allele effects (additive, binary_b)"))
    end
    mats = component == :total ? r.G : component == :genic ? r.G_genic :
           component == :ld ? r.G .- r.G_genic : throw(ArgumentError("component must be :total, :genic or :ld"))
    Dict(zip(r.generation, mats))
end

"P matrices by recorded generation (a `Dict`)."
p_matrices(pop::Population) = Dict(zip(pop.records.generation, pop.records.P))

"Genomes of the current population (rows x alleles x individuals)."
genomes(pop::Population) = pop.genomes

architecture_of(x::Architecture) = x
architecture_of(x::Model) = x.architecture
architecture_of(x::Population) = x.model.architecture

"Genotypic values (traits x individuals) of a genome or an array of genomes."
function genotypic_values(x, g::AbstractArray = genomes(x))
    arch = architecture_of(x)
    ndims(g) == 2 && return genotypic_value(arch, g)
    reduce(hcat, [genotypic_value(arch, view(g, :, :, i)) for i in 1:size(g, 3)])
end

local_effects(pop::Population; individual = 1) =
    local_effects(pop.model.architecture, view(pop.genomes, :, :, individual))
local_effects(m::Model, g::AbstractMatrix) = local_effects(m.architecture, g)

"""
    shuffle_genomes(g)

Pool the 2n allele copies of each locus and assign them to individuals at
random: keeps allele frequencies and allele states, removes associations
between and within loci.
"""
function shuffle_genomes(g::AbstractArray{<:Real, 3})
    rows, na, n = size(g)
    out = copy(g)
    for l in 1:(na ÷ 2)
        pool = reshape(g[:, (2l - 1):(2l), :], rows, 2n)
        out[:, (2l - 1):(2l), :] = reshape(pool[:, randperm(2n)], rows, 2, n)
    end
    out
end

"""
    g_decomposition(pop; method = :genic, shuffles = 10)

`G = genic + ld`: the G expected without associations between loci, and the
rest. `:genic` is exact for architectures whose genotypic values are sums of
allele effects; `:shuffle` works for any architecture (mean G of shuffled
populations). Melo, Marroig & Wolf (2019); do O & Whitlock (2023).
"""
function g_decomposition(pop::Population; method = :genic, shuffles = 10)
    arch = pop.model.architecture
    G = cov(pop.x')
    if method == :genic
        genic = genic_covariance(arch, pop.genomes)
        genic === nothing && throw(ArgumentError(
            "the genic part of G is only defined for architectures where genotypic values are sums of " *
            "allele effects (additive, binary_b); use method = :shuffle"))
    elseif method == :shuffle
        genic = sum(cov(genotypic_values(arch, shuffle_genomes(pop.genomes))') for _ in 1:shuffles) ./ shuffles
    else
        throw(ArgumentError("method must be :genic or :shuffle"))
    end
    (G = G, genic = genic, ld = G .- genic)
end

"""
    additive_covariance(pop)

Least-squares additive genetic covariance from the state of every locus (any
architecture); `nonadditive = G - additive`.
"""
function additive_covariance(pop::Population)
    arch = pop.model.architecture
    n = pop.n
    pred = reduce(vcat, [begin
        g = view(pop.genomes, :, :, i)
        s = arch isa BinaryB ? g[2:end, :] .* g[1, :]' : g
        vec(locus_effects(s))'
    end for i in 1:n])
    size(pred, 2) + 1 < n || throw(ArgumentError("additive_covariance() needs more individuals than loci x traits ($(size(pred, 2)))"))
    D = [ones(n) pred]
    X = Matrix(pop.x')
    fitted = D * (D \ X)
    G = cov(X)
    A = cov(fitted)
    (G = G, additive = A, nonadditive = G .- A)
end

"""
    mutational_covariance(pop; n = 10000, effects = false)

Effects of single mutations on genotypic values, on genomes drawn from the
population: mean effect, per-mutation covariance, expected mutations per
genome per generation, and `M` per generation.
"""
function mutational_covariance(pop::Population; n::Integer = 10000, effects = false)
    m = pop.model
    p = ntraits(m)
    dx = zeros(p, n)
    for s in 1:n
        g = pop.genomes[:, :, rand(1:pop.n)]
        before = genotypic_value(m.architecture, g)
        mutate_one!(m.mutation, g)
        dx[:, s] = genotypic_value(m.architecture, g) .- before
    end
    rate = expected_mutations(m.mutation, view(pop.genomes, :, :, 1))
    out = (mean = vec(mean(dx, dims = 2)), per_mutation = cov(dx'), expected_mutations = rate,
           M = rate .* (dx * dx' ./ n))
    effects ? merge(out, (effects = dx,)) : out
end

"""
    avg_corr(m; modules = [first half, second half])

Mean correlation within each module, between modules, and their ratio.
"""
function avg_corr(m::AbstractMatrix; modules = nothing)
    p = size(m, 1)
    modules === nothing && (modules = [collect(1:(p ÷ 2)), collect((p ÷ 2 + 1):p)])
    s = sqrt.(diag(m))
    r = m ./ (s * s')
    within = [mean([r[i, j] for i in mod, j in mod if i > j]) for mod in modules]
    same = falses(p, p)
    for mod in modules
        same[mod, mod] .= true
    end
    between = mean(r[.!same])
    (within = within, between = between, avg_ratio = mean(within) / between)
end

"""
    autonomy(m, beta)
    mean_autonomy(m; n = 1000)
    eigen_variance(m)

Autonomy in direction(s) `beta` (columns), `1 / ((b' G^-1 b)(b' G b))` for unit
`b` (Hansen & Houle 2008); its mean over random directions; and the
proportion of variance on each principal component.
"""
function autonomy(m::AbstractMatrix, beta::AbstractVecOrMat)
    B = beta isa AbstractVector ? reshape(beta, :, 1) : beta
    B = B ./ sqrt.(sum(B .^ 2, dims = 1))
    iv = inv(m)
    vec(1 ./ (sum(B .* (iv * B), dims = 1) .* sum(B .* (m * B), dims = 1)))
end

mean_autonomy(m::AbstractMatrix; n::Integer = 1000) = mean(autonomy(m, randn(size(m, 1), n)))

function eigen_variance(m::AbstractMatrix)
    ev = sort(eigvals(Symmetric(Matrix(m))), rev = true)
    ev ./ sum(ev)
end
