# Discrete-generation population machinery shared by the BinaryB,
# Antagonistic and PleiotropicVectors models.
#
# This file is `include`d inside each model module. The including module must
# define a mutable struct `Individual` with fields `m` (number of alleles,
# 2 x loci), `p` (number of traits) and `z` (phenotype), plus:
#
#   RandomInd(base_ind)                      new random individual
#   mutation!(pop::Population)               mutate every individual in place
#   cross!(ind_1, ind_2, new_ind, a_1, a_2)  write offspring into new_ind
#   moments(pop::Population)                 Dict of population statistics

"""
Population of individuals evolving on a Gaussian fitness surface.

Fields: `n` size, `base_ind` individual holding the model parameters,
`generation`, `fitness` (normalised to sum to one), `surface`
(`MvNormal(theta, omega)`), `pop` the individuals and `moments`, the
statistics recorded during `run_pop`.
"""
mutable struct Population
    n::Int
    base_ind::Individual
    generation::Int
    fitness::Vector{Float64}
    surface::MvNormal
    pop::Vector{Individual}
    moments::Vector{Any}
end

function Population(n::Integer, base_ind::Individual)
    Population(n, base_ind, 0, fill(1 / n, n),
               MvNormal(zeros(base_ind.p), Matrix{Float64}(I, base_ind.p, base_ind.p)),
               Individual[], Any[])
end

function Population(n::Integer, base_ind::Individual, pop_array::Vector{Individual})
    pop = Population(n, base_ind)
    pop.pop = pop_array
    pop
end

"Fill `pop` with random individuals and reset its generation and moments."
function RandomPop!(pop::Population)
    pop.pop = [RandomInd(pop.base_ind) for _ in 1:pop.n]
    pop.generation = 0
    pop.moments = Any[]
    pop.fitness = fill(1 / pop.n, pop.n)
    pop
end

Base.string(pop::Population) =
    "a Population with $(pop.n) individuals, at generation $(pop.generation)"
Base.show(io::IO, pop::Population) = print(io, "This is ", string(pop))

Base.getindex(pop::Population, i::Integer) = pop.pop[i]
Base.getindex(pop::Population, s::AbstractUnitRange) =
    Population(length(s), pop.base_ind, pop.pop[s])

"Copy `source` into `sink`; individuals are deep-copied, never shared."
function Base.copy!(sink::Population, source::Population)
    sink.n = source.n
    sink.base_ind = source.base_ind
    sink.generation = source.generation
    sink.fitness = copy(source.fitness)
    sink.surface = source.surface
    sink.pop = deepcopy(source.pop)
    sink.moments = copy(source.moments)
    sink
end

Base.copy(source::Population) = copy!(Population(source.n, source.base_ind), source)

"Normalised Gaussian fitness of every individual."
function fitness!(pop::Population)
    logfit = [logpdf(pop.surface, ind.z) for ind in pop.pop]
    pop.fitness = exp.(logfit .- logsumexp(logfit))
end

changeSurface!(pop::Population, theta::AbstractVector, omega::AbstractMatrix) =
    pop.surface = MvNormal(collect(float.(theta)), Matrix{Float64}(omega))

"Fill `mates` with parent indices drawn with probability proportional to fitness."
function choose_mates!(pop::Population, mates::Vector{Int})
    matings = rand(Multinomial(pop.n, pop.fitness))
    l = 1
    for k in 1:pop.n, _ in 1:matings[k]
        mates[l] = k
        l += 1
    end
    shuffle!(mates)
end

"""
    next_generation!(pop, holder_pop, sires, dames, alele_1, alele_2; selective = true)

Selection, mutation and mating. Offspring are written into `holder_pop`,
then the two populations swap individuals, so parents are never overwritten
while they are still being drawn.
"""
function next_generation!(pop::Population, holder_pop::Population, sires, dames,
                          alele_1, alele_2; selective = true)
    holder_pop.surface = pop.surface
    if selective
        fitness!(pop)
    else
        fill!(pop.fitness, 1 / pop.n)
    end
    holder_pop.fitness = pop.fitness
    mutation!(pop)
    choose_mates!(pop, sires)
    choose_mates!(pop, dames)
    for i in 1:pop.n
        cross!(pop[sires[i]], pop[dames[i]], holder_pop.pop[i], alele_1, alele_2)
    end
    holder_pop.generation = pop.generation + 1
    pop.pop, holder_pop.pop = holder_pop.pop, pop.pop
    pop.generation = holder_pop.generation
    pop
end

"Record the current population statistics in `pop.moments`."
moments!(pop::Population) = push!(pop.moments, moments(pop))

"""
    run_pop(ind, n_e, selectionRegimes, regimeGenerations; theta, delta_theta, omega, thin = 100, progress = true)
    run_pop(pop, selectionRegimes, regimeGenerations; ...)

Run a population through a sequence of regimes, e.g.
`["Drift", "Stab", "Direct"]` for `[5000, 5000, 5000]` generations.
"Drift" has no selection; regimes starting with "Dire" move the optimum by
`delta_theta` every generation; anything else is stabilising selection.
Moments are recorded every `thin` generations. The first method starts from
a random population of `n_e` individuals; the second continues `pop` in place.
"""
function run_pop(ind::Individual, n_e::Integer,
                 selectionRegimes::AbstractVector{<:AbstractString},
                 regimeGenerations::AbstractVector{<:Integer};
                 theta = zeros(ind.p),
                 delta_theta = zeros(ind.p),
                 omega = Matrix{Float64}(I, ind.p, ind.p),
                 thin = 100, progress = true)
    pop = Population(n_e, ind)
    RandomPop!(pop)
    run_pop(pop, selectionRegimes, regimeGenerations;
            theta = theta, delta_theta = delta_theta, omega = omega,
            thin = thin, progress = progress)
end

function run_pop(pop::Population,
                 selectionRegimes::AbstractVector{<:AbstractString},
                 regimeGenerations::AbstractVector{<:Integer};
                 theta = mean(pop.surface),
                 delta_theta = zeros(pop.base_ind.p),
                 omega = cov(pop.surface),
                 thin = 100, progress = true)
    length(selectionRegimes) == length(regimeGenerations) ||
        throw(ArgumentError("one generation count per regime is needed"))
    theta = collect(float.(theta))
    changeSurface!(pop, theta, omega)
    fitness!(pop)

    holder_pop = copy(pop)
    sires = zeros(Int, pop.n)
    dames = zeros(Int, pop.n)
    alele_1 = zeros(Int, pop.base_ind.m ÷ 2)
    alele_2 = zeros(Int, pop.base_ind.m ÷ 2)

    for (regime, n_gen) in zip(selectionRegimes, regimeGenerations)
        meter = Progress(n_gen; dt = 2, desc = string(regime, " "), enabled = progress)
        for i in 1:n_gen
            if i % thin == 0
                moments!(pop)
            end
            if is_directional(regime)
                theta = theta .+ delta_theta
                changeSurface!(pop, theta, omega)
            end
            next_generation!(pop, holder_pop, sires, dames, alele_1, alele_2;
                             selective = !is_drift(regime))
            next!(meter)
        end
    end
    pop
end
