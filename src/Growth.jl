"""
Overlapping-generations model with juveniles and adults.

Genetics as in `BinaryB` (binary `b`, `z = b * y`, no environmental noise),
but individuals have an age and become adults when `age > maturity`. Each time
step:

1. fitness is computed on the adult surface for adults and on the juvenile
   surface for juveniles, normalised within each class;
2. individuals die with probability `1 - logistic(10 w / max(w) - 5)`, where
   `w / max(w)` is the fitness relative to the fittest member of their class;
3. every dead individual is replaced by an offspring of two surviving adults,
   drawn with probability proportional to fitness (mutation happens in the
   offspring);
4. everybody ages by one step.
"""
module Growth

using Distributions
using LinearAlgebra
using LogExpFunctions: logsumexp, logistic
using ProgressMeter
using Random
using Statistics
using ..Common

export Individual, Population, RandomInd, RandomPop!, run_pop, next_time_step!,
       fitness!, mortality!, mutation!, cross!, changeSurface!, moments, moments!

mutable struct Individual
    m::Int            # number of alleles (2 x loci)
    p::Int            # number of traits
    mu::Float64       # mutation rate of allelic effects, per allele
    mu_b::Float64     # mutation rate of b, per entry
    mu_sigma::Float64 # standard deviation of mutational effects
    maturity::Int     # age after which an individual is an adult
    y::Vector{Float64}
    b::Matrix{Int}
    z::Vector{Float64}
    age::Int
    Individual(m, p, mu, mu_b, mu_sigma, maturity) =
        new(2 * m, p, mu, mu_b, mu_sigma, maturity)
end

"""
    Individual(m, p, mu, mu_b, mu_sigma, maturity)

Parameter holder (`base_ind`) for `m` diploid loci and `p` traits.
"""
Individual

function RandomInd!(ind::Individual, y_mean::Float64 = 0.0, g_sigma::Float64 = ind.mu_sigma)
    ind.y = rand(Normal(y_mean, g_sigma), ind.m)
    ind.b = rand(0:1, ind.p, ind.m)
    ind.z = ind.b * ind.y
    ind.age = rand(1:100)
    ind
end

function RandomInd(base_ind::Individual, y_mean::Float64 = 0.0,
                   g_sigma::Float64 = base_ind.mu_sigma)
    new_ind = Individual(base_ind.m ÷ 2, base_ind.p, base_ind.mu, base_ind.mu_b,
                         base_ind.mu_sigma, base_ind.maturity)
    RandomInd!(new_ind, y_mean, g_sigma)
end

identity_matrix(p) = Matrix{Float64}(I, p, p)

mutable struct Population
    n::Int
    base_ind::Individual
    generation::Int
    fitness::Vector{Float64}
    a_surface::MvNormal
    j_surface::MvNormal
    pop::Vector{Individual}
    moments::Vector{Any}
    is_adult::Vector{Bool}
    the_departed::Vector{Int}
    a_p_dep::Float64
    j_p_dep::Float64
end

function Population(n::Integer, base_ind::Individual)
    surface = MvNormal(zeros(base_ind.p), identity_matrix(base_ind.p))
    Population(n, base_ind, 0, zeros(n), surface, surface, Individual[], Any[],
               falses(n), Int[], 0.0, 0.0)
end

update_adults!(pop::Population) =
    pop.is_adult = [ind.age > pop.base_ind.maturity for ind in pop.pop]

function RandomPop!(pop::Population, y_mean::Float64 = 0.0,
                    g_sigma::Float64 = pop.base_ind.mu_sigma)
    pop.pop = [RandomInd(pop.base_ind, y_mean, g_sigma) for _ in 1:pop.n]
    pop.generation = 0
    pop.moments = Any[]
    pop.fitness = zeros(pop.n)
    pop.the_departed = Int[]
    update_adults!(pop)
    pop
end

Base.string(pop::Population) =
    "a Population with $(pop.n) individuals, at time step $(pop.generation)"
Base.show(io::IO, pop::Population) = print(io, "This is ", string(pop))
Base.getindex(pop::Population, i::Integer) = pop.pop[i]

"Copy `source` into `sink`; individuals are deep-copied, never shared."
function Base.copy!(sink::Population, source::Population)
    for f in fieldnames(Population)
        setfield!(sink, f, f == :base_ind ? source.base_ind : deepcopy(getfield(source, f)))
    end
    sink
end

Base.copy(source::Population) = copy!(Population(source.n, source.base_ind), source)

function changeSurface!(pop::Population, a_theta, a_omega, j_theta, j_omega)
    pop.a_surface = MvNormal(collect(float.(a_theta)), Matrix{Float64}(a_omega))
    pop.j_surface = MvNormal(collect(float.(j_theta)), Matrix{Float64}(j_omega))
end

normalise_log(logfit) = isempty(logfit) ? logfit : logfit .- logsumexp(logfit)

"Fitness on the adult or juvenile surface, normalised within each class."
function fitness!(pop::Population)
    logfit = [logpdf(pop.is_adult[i] ? pop.a_surface : pop.j_surface, pop[i].z)
              for i in 1:pop.n]
    logfit[pop.the_departed] .= -1e11
    adults = pop.is_adult
    logfit[adults] = normalise_log(logfit[adults])
    logfit[.!adults] = normalise_log(logfit[.!adults])
    pop.fitness = exp.(logfit)
end

"Uniform fitness within each class (no selection)."
function neutral_fitness!(pop::Population)
    n_adults = count(pop.is_adult)
    pop.fitness = [a ? 1 / n_adults : 1 / (pop.n - n_adults) for a in pop.is_adult]
end

"Add the individuals that die this step to `the_departed`."
function mortality!(pop::Population)
    adults = pop.is_adult
    max_a = any(adults) ? maximum(pop.fitness[adults]) : 1.0
    max_j = any(.!adults) ? maximum(pop.fitness[.!adults]) : 1.0
    survival_prob = [logistic(pop.fitness[i] / (adults[i] ? max_a : max_j) * 10 - 5)
                     for i in 1:pop.n]
    dying = findall(rand(pop.n) .> survival_prob)
    pop.the_departed = union(pop.the_departed, dying)
    dead_adults = adults[pop.the_departed]
    pop.a_p_dep = count(dead_adults) / count(adults)
    pop.j_p_dep = count(.!dead_adults) / count(.!adults)
    pop
end

function mutation!(ind::Individual)
    for _ in 1:rand(Binomial(ind.m, ind.mu))
        i = rand(1:ind.m)
        ind.y[i] += rand(Normal(0, ind.mu_sigma))
    end
    for _ in 1:rand(Binomial(ind.m * ind.p, ind.mu_b))
        i = rand(1:ind.p)
        j = rand(1:ind.m)
        ind.b[i, j] = 1 - ind.b[i, j]
    end
end

"Offspring of `ind_1` and `ind_2` (free recombination, then mutation), age 0."
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
    mutation!(new_ind)
    new_ind.z = new_ind.b * new_ind.y
    new_ind.age = 0
    new_ind
end

"One time step: fitness, mortality, replacement of the dead, ageing."
function next_time_step!(pop::Population, alele_1, alele_2; selective = true)
    selective ? fitness!(pop) : neutral_fitness!(pop)
    adults = findall(pop.is_adult)
    mortality!(pop)

    # only adults that survived this step reproduce; their slots are not overwritten
    parents = setdiff(adults, pop.the_departed)
    if !isempty(parents)
        w = pop.fitness[parents]
        parent_dist = Categorical(sum(w) > 0 ? w / sum(w) : fill(1 / length(w), length(w)))
        for ind in pop.the_departed
            sire = parents[rand(parent_dist)]
            dam  = parents[rand(parent_dist)]
            cross!(pop[sire], pop[dam], pop[ind], alele_1, alele_2)
        end
        pop.the_departed = Int[]
    end

    for ind in pop.pop
        ind.age += 1
    end
    update_adults!(pop)
    pop.generation += 1
    pop
end

function moments(pop::Population)
    ys = reduce(vcat, (ind.y' for ind in pop.pop))
    zs = reduce(vcat, (ind.z' for ind in pop.pop))
    mean_b = sum(ind.b for ind in pop.pop) / pop.n
    n_adults = count(pop.is_adult) / pop.n
    Dict("mean_y" => vec(mean(ys, dims = 1)),
         "mean_b" => mean_b,
         "mean_z" => vec(mean(zs, dims = 1)),
         "zs" => zs,
         "theta" => copy(mean(pop.a_surface)),
         "gen" => pop.generation,
         "P" => cov(zs),
         "corrP" => cor(zs),
         "adults" => n_adults,
         "juveniles" => 1 - n_adults,
         "p_d_adults" => pop.a_p_dep,
         "p_d_juveniles" => pop.j_p_dep,
         "a_fitness" => std(pop.fitness[pop.is_adult]),
         "j_fitness" => std(pop.fitness[.!pop.is_adult]))
end

moments!(pop::Population) = push!(pop.moments, moments(pop))

"""
    run_pop(ind, n_e, selectionRegimes, regimeGenerations; a_theta, a_delta_theta, a_omega, j_theta, j_delta_theta, j_omega, thin = 100, progress = true)
    run_pop(pop, selectionRegimes, regimeGenerations; ...)

As `BinaryB.run_pop`, with separate adult (`a_`) and juvenile (`j_`) fitness
surfaces. Generations are time steps.
"""
function run_pop(ind::Individual, n_e::Integer,
                 selectionRegimes::AbstractVector{<:AbstractString},
                 regimeGenerations::AbstractVector{<:Integer};
                 a_theta = zeros(ind.p), a_delta_theta = zeros(ind.p),
                 a_omega = identity_matrix(ind.p),
                 j_theta = zeros(ind.p), j_delta_theta = zeros(ind.p),
                 j_omega = identity_matrix(ind.p),
                 thin = 100, progress = true)
    pop = Population(n_e, ind)
    RandomPop!(pop)
    run_pop(pop, selectionRegimes, regimeGenerations;
            a_theta, a_delta_theta, a_omega, j_theta, j_delta_theta, j_omega,
            thin, progress)
end

function run_pop(pop::Population,
                 selectionRegimes::AbstractVector{<:AbstractString},
                 regimeGenerations::AbstractVector{<:Integer};
                 a_theta = mean(pop.a_surface), a_delta_theta = zeros(pop.base_ind.p),
                 a_omega = cov(pop.a_surface),
                 j_theta = mean(pop.j_surface), j_delta_theta = zeros(pop.base_ind.p),
                 j_omega = cov(pop.j_surface),
                 thin = 100, progress = true)
    length(selectionRegimes) == length(regimeGenerations) ||
        throw(ArgumentError("one generation count per regime is needed"))
    a_theta = collect(float.(a_theta))
    j_theta = collect(float.(j_theta))
    changeSurface!(pop, a_theta, a_omega, j_theta, j_omega)
    fitness!(pop)

    alele_1 = zeros(Int, pop.base_ind.m ÷ 2)
    alele_2 = zeros(Int, pop.base_ind.m ÷ 2)

    for (regime, n_gen) in zip(selectionRegimes, regimeGenerations)
        meter = Progress(n_gen; dt = 2, desc = string(regime, " "), enabled = progress)
        for i in 1:n_gen
            if i % thin == 0
                moments!(pop)
            end
            if is_directional(regime)
                a_theta = a_theta .+ a_delta_theta
                j_theta = j_theta .+ j_delta_theta
                changeSurface!(pop, a_theta, a_omega, j_theta, j_omega)
            end
            next_time_step!(pop, alele_1, alele_2; selective = !is_drift(regime))
            next!(meter)
        end
    end
    pop
end

end
