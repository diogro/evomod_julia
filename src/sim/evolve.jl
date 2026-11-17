# Models, regimes, life cycles and the generation loop, as in evomod2.

# ---- Life cycles ----------------------------------------------------------------------

abstract type LifeCycle end
struct WrightFisher <: LifeCycle end
struct Moran <: LifeCycle end
struct ViabilitySelection <: LifeCycle
    mates::Int
    offspring_per_mate::Int
end

"""
    wright_fisher()
    moran()
    viability_selection(; mates = 2, offspring_per_mate = 2)

Life cycles, as in evomod2: discrete generations with parents drawn in
proportion to fitness (default); overlapping generations (`n` birth-death
events per generation); or separate sexes, `mates` random males per female,
`offspring_per_mate` offspring each, viability selection on zygotes and
regulation back to `n` (Jones, Arnold & Burger).
"""
wright_fisher() = WrightFisher()
moran() = Moran()
function viability_selection(; mates = 2, offspring_per_mate = 2)
    mates * offspring_per_mate >= 2 ||
        throw(ArgumentError("each female needs at least 2 offspring (mates x offspring_per_mate)"))
    ViabilitySelection(mates, offspring_per_mate)
end

# ---- Model -------------------------------------------------------------------------------

struct Model{A <: Architecture, M <: Mutation, R <: Recombination, L <: LifeCycle}
    architecture::A
    mutation::M
    recombination::R
    life_cycle::L
    env_sd::Float64
    init_sd::Float64
end

"""
    evomod_model(architecture, mutation; env_sd = 0, init_sd = 0,
                 recombination = free_recombination(), life_cycle = wright_fisher())

A simulation model, as in evomod2.
"""
function evomod_model(architecture::Architecture, mutation; env_sd = 0.0, init_sd = 0.0,
                      recombination = free_recombination(), life_cycle = wright_fisher())
    Model(architecture, make_mutation(architecture, mutation),
          make_recombination(recombination, nloci(architecture)), life_cycle,
          float(env_sd), float(init_sd))
end

ntraits(m::Model) = ntraits(m.architecture)

function Base.show(io::IO, m::Model)
    pers = hasproperty(m.mutation, :persistence) ? m.mutation.persistence : 1.0
    print(io, "evomod model: ", nameof(typeof(m.architecture)), " (", nloci(m.architecture),
          " loci, ", ntraits(m), " traits), ", nameof(typeof(m.mutation)),
          pers == 1 ? " (random walk)" : pers == 0 ? " (house of cards)" : " (persistence $pers)",
          ", ", nameof(typeof(m.recombination)), ", ", nameof(typeof(m.life_cycle)),
          ", env_sd = ", m.env_sd)
end

# ---- Regimes -----------------------------------------------------------------------------

abstract type Regime end
struct Drift <: Regime
    generations::Int
end
struct Stabilizing <: Regime
    generations::Int
    omega::Any
    theta::Any
end
struct Directional <: Regime
    generations::Int
    delta_theta::Vector{Float64}
    omega::Any
    theta::Any
end
struct Fluctuating <: Regime
    generations::Int
    every::Int
    step::Any
    omega::Any
    structure::Any
    bounds::Any
    theta::Any
end

"""
    drift(generations)
    stabilizing(generations; omega, theta = nothing)
    directional(generations; delta_theta, omega, theta = nothing)
    fluctuating(generations; every, step, omega, structure = :correlated, bounds = (-1, 1), theta = nothing)

Selection regimes, as in evomod2. `omega` is a covariance matrix or a number
(times the identity). Without `theta` a regime continues from the current
optimum. `fluctuating` moves the optimum by `+-step` after every `every`
generations, for all traits together (`:correlated`), each trait on its own
(`:independent`) or in blocks (a vector of trait index vectors); a jump that
would leave `bounds` is not taken (`bounds = nothing`: unbounded).
"""
drift(generations::Integer) = Drift(generations)
stabilizing(generations::Integer; omega, theta = nothing) = Stabilizing(generations, omega, theta)
directional(generations::Integer; delta_theta, omega, theta = nothing) =
    Directional(generations, collect(float.(delta_theta)), omega, theta)
function fluctuating(generations::Integer; every::Integer, step, omega, structure = :correlated,
                     bounds = (-1.0, 1.0), theta = nothing)
    bounds === nothing || bounds[1] < bounds[2] || throw(ArgumentError("bounds must be (lower, upper) with lower < upper"))
    structure isa Symbol && !(structure in (:correlated, :independent)) &&
        throw(ArgumentError("structure must be :correlated, :independent or a vector of trait blocks"))
    Fluctuating(generations, every, step, omega, structure, bounds, theta)
end

omega_matrix(omega, p) = omega isa Number ? omega * Matrix{Float64}(I, p, p) : Matrix{Float64}(omega)

function blocks_of(structure, p)
    structure == :correlated && return zeros(Int, p)
    structure == :independent && return collect(0:(p - 1))
    traits = reduce(vcat, structure)
    allunique(traits) && all(1 .<= traits .<= p) ||
        throw(ArgumentError("blocks must be disjoint sets of trait indices between 1 and $p"))
    block = zeros(Int, p)
    for (b, t) in enumerate(structure)
        block[t] .= b - 1
    end
    alone = setdiff(1:p, traits)
    block[alone] = length(structure) .+ (0:(length(alone) - 1))
    block
end

# Each block draws a direction and moves by step unless a trait would leave the bounds.
function fluctuate!(theta, step, block, lower, upper)
    for b in 0:maximum(block)
        traits = findall(==(b), block)
        isempty(traits) && continue
        d = rand() < 0.5 ? -1.0 : 1.0
        moved = theta[traits] .+ d .* step[traits]
        if all(moved .>= lower .- 1e-9) && all(moved .<= upper .+ 1e-9)
            theta[traits] = moved
        end
    end
end

# ---- Population and records -------------------------------------------------------------

mutable struct Records
    generation::Vector{Int}
    theta::Vector{Vector{Float64}}
    mean_x::Vector{Vector{Float64}}
    mean_z::Vector{Vector{Float64}}
    G::Vector{Matrix{Float64}}
    P::Vector{Matrix{Float64}}
    G_genic::Vector{Matrix{Float64}}
    mean_fitness::Vector{Float64}
end
Records() = Records(Int[], Vector{Float64}[], Vector{Float64}[], Vector{Float64}[],
                    Matrix{Float64}[], Matrix{Float64}[], Matrix{Float64}[], Float64[])

mutable struct Population
    model::Model
    n::Int
    genomes::Array{Float64, 3}
    x::Matrix{Float64}
    z::Matrix{Float64}
    generation::Int
    theta::Vector{Float64}
    record_every::Int
    records::Records
    regimes::Vector{Any}
end

Base.show(io::IO, p::Population) =
    print(io, "evomod population: ", p.n, " individuals at generation ", p.generation,
          ", ", length(p.records.generation), " records")

"Genic part of G: 2 x sum over loci of the covariance of allele effects (n - 1 scaling)."
function genic_covariance(arch::Architecture, genomes::Array{Float64, 3})
    n = size(genomes, 3)
    E0 = allele_effects(arch, view(genomes, :, :, 1))
    E0 === nothing && return nothing
    p, na = size(E0)
    genic = zeros(p, p)
    A = zeros(p, 2n)
    effects = [Matrix(allele_effects(arch, view(genomes, :, :, i))) for i in 1:n]
    for l in 1:(na ÷ 2)
        for i in 1:n
            A[:, 2i - 1] = effects[i][:, 2l - 1]
            A[:, 2i] = effects[i][:, 2l]
        end
        mu = vec(mean(A, dims = 2))
        genic .+= 2 .* (A * A' ./ (2n) .- mu * mu')
    end
    genic .* (n / (n - 1))
end

function record!(rec::Records, gen, theta, X, Z, arch, genomes, w)
    push!(rec.generation, gen)
    push!(rec.theta, copy(theta))
    push!(rec.mean_x, vec(mean(X, dims = 2)))
    push!(rec.mean_z, vec(mean(Z, dims = 2)))
    push!(rec.G, cov(X'))
    push!(rec.P, cov(Z'))
    g = genic_covariance(arch, genomes)
    g === nothing || push!(rec.G_genic, g)
    push!(rec.mean_fitness, w)
end

log_fitness(z::AbstractVector, theta, omega_inv) = (d = z .- theta; -0.5 * dot(d, omega_inv * d))

# ---- Weighted sampling --------------------------------------------------------------------

function sample_weighted(cumw::Vector{Float64})
    u = rand() * cumw[end]
    min(searchsortedlast(cumw, u) + 1, length(cumw))
end

# Fenwick tree of weights: O(log n) updates and weighted draws (Moran)
mutable struct Fenwick
    tree::Vector{Float64}
    n::Int
    top::Int
    total::Float64
end
function Fenwick(w::Vector{Float64})
    n = length(w)
    tree = zeros(n)
    for i in 1:n
        tree[i] += w[i]
        j = i + (i & -i)
        j <= n && (tree[j] += tree[i])
    end
    Fenwick(tree, n, prevpow(2, max(n, 1)), sum(w))
end
function add!(f::Fenwick, i::Int, delta)
    f.total += delta
    while i <= f.n
        f.tree[i] += delta
        i += i & -i
    end
end
function draw(f::Fenwick)
    u = rand() * f.total
    pos, step = 0, f.top
    while step > 0
        if pos + step <= f.n && f.tree[pos + step] <= u
            pos += step
            u -= f.tree[pos]
        end
        step >>= 1
    end
    min(pos + 1, f.n)
end

# ---- One generation of each life cycle ------------------------------------------------------

function phenotype!(X, Z, i, arch, g, env_sd)
    X[:, i] = genotypic_value(arch, g)
    Z[:, i] = X[:, i] .+ env_sd .* randn(size(X, 1))
end

function generation!(::WrightFisher, pop, buffer, w, selective, theta, omega_inv, shift)
    m = pop.model
    n = pop.n
    for i in 1:n
        mutate!(m.mutation, view(pop.genomes, :, :, i))
    end
    cumw = cumsum(w)
    sires = [sample_weighted(cumw) for _ in 1:n]
    dames = [sample_weighted(cumw) for _ in 1:n]
    for i in 1:n
        offspring!(m.recombination, view(pop.genomes, :, :, sires[i]), view(pop.genomes, :, :, dames[i]),
                   view(buffer, :, :, i))
    end
    pop.genomes, buffer = buffer, pop.genomes
    for i in 1:n
        phenotype!(pop.x, pop.z, i, m.architecture, view(pop.genomes, :, :, i), m.env_sd)
    end
    buffer
end

function generation!(::Moran, pop, buffer, w, selective, theta, omega_inv, shift)
    m = pop.model
    n = pop.n
    f = Fenwick(copy(w))
    kid = zeros(size(pop.genomes, 1), size(pop.genomes, 2))
    for _ in 1:n
        dead = rand(1:n)
        sire, dame = draw(f), draw(f)
        offspring!(m.recombination, view(pop.genomes, :, :, sire), view(pop.genomes, :, :, dame), kid)
        mutate!(m.mutation, kid)
        pop.genomes[:, :, dead] = kid
        phenotype!(pop.x, pop.z, dead, m.architecture, kid, m.env_sd)
        w_new = selective ? exp(log_fitness(view(pop.z, :, dead), theta, omega_inv) - shift) : 1.0
        add!(f, dead, w_new - w[dead])
        w[dead] = w_new
    end
    buffer
end

function generation!(lc::ViabilitySelection, pop, buffer, w, selective, theta, omega_inv, shift)
    m = pop.model
    n = pop.n
    females = n ÷ 2
    nz = females * lc.mates * lc.offspring_per_mate
    zyg = zeros(size(pop.genomes, 1), size(pop.genomes, 2), nz)
    zx, zz = zeros(size(pop.x, 1), nz), zeros(size(pop.x, 1), nz)
    survivors = Int[]
    k = 0
    for f in 1:females, _ in 1:lc.mates
        male = females + rand(1:(n - females))
        for _ in 1:lc.offspring_per_mate
            k += 1
            g = view(zyg, :, :, k)
            offspring!(m.recombination, view(pop.genomes, :, :, male), view(pop.genomes, :, :, f), g)
            mutate!(m.mutation, g)
            phenotype!(zx, zz, k, m.architecture, g, m.env_sd)
            wz = selective ? exp(log_fitness(view(zz, :, k), theta, omega_inv)) : 1.0
            rand() < wz && push!(survivors, k)
        end
    end
    s = length(survivors)
    s == 0 && error("no zygote survived viability selection")
    for i in 1:n
        if i <= s
            j = i + rand(0:(s - i))
            survivors[i], survivors[j] = survivors[j], survivors[i]
            pick = survivors[i]
        else
            pick = survivors[rand(1:s)]
        end
        pop.genomes[:, :, i] = view(zyg, :, :, pick)
        pop.x[:, i] = zx[:, pick]
        pop.z[:, i] = zz[:, pick]
    end
    buffer
end

# ---- evolve ---------------------------------------------------------------------------------

"""
    evolve(model, regimes; n, record_every = 1, seed = nothing, initial = nothing)
    evolve(pop, regimes; record_every = pop.record_every, seed = nothing)

Run a new population of `n` individuals (or continue `pop`, in place) through
a regime or a vector of regimes, as in evomod2. Statistics are recorded every
`record_every` generations (and at the start of a new population). Random
numbers come from Julia's default generator; `seed` calls `Random.seed!`.
"""
function evolve(model::Model, regimes; n::Integer, record_every::Integer = 1, seed = nothing,
                initial = nothing)
    seed === nothing || Random.seed!(seed)
    model.life_cycle isa ViabilitySelection && isodd(n) &&
        throw(ArgumentError("viability_selection() needs an even population size"))
    arch = model.architecture
    p = ntraits(model)
    genomes = initial === nothing ? init_genomes(arch, n, model.init_sd) : Array{Float64, 3}(initial)
    size(genomes) == (state_rows(arch), 2 * nloci(arch), n) ||
        throw(ArgumentError("initial must be $(state_rows(arch)) x $(2 * nloci(arch)) x $n"))
    x = zeros(p, n)
    for i in 1:n
        x[:, i] = genotypic_value(arch, view(genomes, :, :, i))
    end
    z = x .+ model.env_sd .* randn(p, n)
    pop = Population(model, n, genomes, x, z, 0, zeros(p), record_every, Records(), Any[])
    run_regimes!(pop, regimes, true)
end

function evolve(pop::Population, regimes; record_every::Integer = pop.record_every, seed = nothing)
    seed === nothing || Random.seed!(seed)
    pop.record_every = record_every
    run_regimes!(pop, regimes, false)
end

function run_regimes!(pop::Population, regimes, record_start::Bool)
    regimes = regimes isa Regime ? [regimes] : collect(regimes)
    m = pop.model
    p = ntraits(m)
    arch = m.architecture
    theta = copy(pop.theta)
    first_theta = (regimes[1] isa Drift || regimes[1].theta === nothing) ? theta : collect(float.(regimes[1].theta))
    if record_start
        r1 = regimes[1]
        w0 = r1 isa Drift ? NaN :
             mean(exp(log_fitness(view(pop.z, :, i), first_theta, inv(omega_matrix(r1.omega, p)))) for i in 1:pop.n)
        record!(pop.records, pop.generation, first_theta, pop.x, pop.z, arch, pop.genomes, w0)
    end
    buffer = similar(pop.genomes)
    w = ones(pop.n)
    for r in regimes
        selective = !(r isa Drift)
        if !(r isa Drift) && r.theta !== nothing
            length(r.theta) == p || throw(ArgumentError("theta must have length $p"))
            theta = collect(float.(r.theta))
        end
        omega = r isa Drift ? Matrix{Float64}(I, p, p) : omega_matrix(r.omega, p)
        size(omega) == (p, p) && issymmetric(omega) || throw(ArgumentError("omega must be a symmetric $p x $p matrix"))
        omega_inv = inv(omega)
        delta = r isa Directional ? r.delta_theta : zeros(p)
        length(delta) == p || throw(ArgumentError("delta_theta must have length $p"))
        if r isa Fluctuating
            step = r.step isa Number ? fill(float(r.step), p) : collect(float.(r.step))
            block = blocks_of(r.structure, p)
            lower, upper = r.bounds === nothing ? (-Inf, Inf) : r.bounds
            any((theta .< lower) .| (theta .> upper)) &&
                throw(ArgumentError("the optimum at the start of a fluctuating regime is outside bounds"))
        end
        for g in 0:(r.generations - 1)
            theta .+= delta
            if r isa Fluctuating && g > 0 && g % r.every == 0
                fluctuate!(theta, step, block, lower, upper)
            end
            shift = 0.0
            if selective
                logw = [log_fitness(view(pop.z, :, i), theta, omega_inv) for i in 1:pop.n]
                shift = maximum(logw)
                w = exp.(logw .- shift)
            else
                w = ones(pop.n)
            end
            buffer = generation!(m.life_cycle, pop, buffer, w, selective, theta, omega_inv, shift)
            pop.generation += 1
            if pop.generation % pop.record_every == 0
                wbar = selective ? mean(exp(log_fitness(view(pop.z, :, i), theta, omega_inv)) for i in 1:pop.n) : NaN
                record!(pop.records, pop.generation, theta, pop.x, pop.z, arch, pop.genomes, wbar)
            end
        end
        push!(pop.regimes, r)
    end
    pop.theta = theta
    pop
end
