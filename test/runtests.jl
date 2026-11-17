using EvoMod
using LinearAlgebra
using Random
using Statistics
using Test

const B = EvoMod.BinaryB
const A = EvoMod.Antagonistic
const PV = EvoMod.PleiotropicVectors
const GR = EvoMod.Growth
const BM = EvoMod.BMatrix
const ML = EvoMod.Multilinear

@testset "EvoMod" begin

@testset "Common" begin
    omega = twoModuleMatrix(fill(2.0, 4), 0.8, 0.5)
    @test omega ≈ [2 1.6 0 0; 1.6 2 0 0; 0 0 2 1; 0 0 1 2]
    c = [1 0.8 0.1 0.1; 0.8 1 0.1 0.1; 0.1 0.1 1 0.6; 0.1 0.1 0.6 1]
    avg_plus, avg_minus = AVGcorr(c)
    @test avg_plus ≈ [0.8, 0.6]
    @test avg_minus ≈ 0.1
    @test AVGratio(c) ≈ 7
    @test lowerTri(c) == [0.8, 0.1, 0.1, 0.1, 0.1, 0.6]
    @test EvoMod.Common.is_drift(" Drift") && !EvoMod.Common.is_drift("Stab")
    @test EvoMod.Common.is_directional("Direct")
end

# Full heritable state of allele `a`: effect and its column of b, or the
# effect vector (Multilinear)
allele(ind, a) = hasproperty(ind, :b) ? (ind.y[a], ind.b[:, a]) : ind.y[:, a]

# Generic checks for the discrete-generation models
function check_generations(M, base_ind; n = 60)
    Random.seed!(1)
    pop = M.Population(n, base_ind)
    M.RandomPop!(pop)

    # copy never shares individuals
    c = copy(pop)
    @test all(c.pop[i] !== pop.pop[i] for i in 1:n)
    @test all(c.pop[i].y == pop.pop[i].y for i in 1:n)

    # offspring alleles come from the right parent and locus
    a1 = zeros(Int, base_ind.m ÷ 2); a2 = similar(a1)
    kid = deepcopy(pop[3])
    M.cross!(pop[1], pop[2], kid, a1, a2)
    for l in 1:(base_ind.m ÷ 2)
        @test allele(kid, 2l - 1) == allele(pop[1], 2l - 1 + a1[l])
        @test allele(kid, 2l) == allele(pop[2], 2l - 1 + a2[l])
    end

    # a generation step does not write offspring into the parents
    holder = copy(pop)
    parents_before = deepcopy(pop.pop)
    parent_objects = copy(pop.pop)
    sires = zeros(Int, n); dames = zeros(Int, n)
    M.changeSurface!(pop, zeros(base_ind.p), Matrix(1.0I, base_ind.p, base_ind.p))
    M.next_generation!(pop, holder, sires, dames, a1, a2)
    @test pop.generation == 1
    @test isempty(intersect(Set(objectid.(pop.pop)), Set(objectid.(parent_objects))))
    # offspring alleles are alleles of the same locus in the right parent
    # (exact unless the parent's allele mutated this generation)
    matches = [let parent = parents_before[isodd(k) ? sires[i] : dames[i]], l = (k + 1) ÷ 2
                   allele(pop[i], k) in (allele(parent, 2l - 1), allele(parent, 2l))
               end for i in 1:n, k in 1:base_ind.m]
    @test mean(matches) > 0.95

    # directional selection drags the population towards the moving optimum
    Random.seed!(2)
    p = base_ind.p
    delta = [fill(0.05, p ÷ 2); zeros(p - p ÷ 2)]
    run = M.run_pop(base_ind, 200, ["Drift", "Stab", "Direct"], [5, 5, 60];
                    delta_theta = delta, omega = twoModuleMatrix(fill(2.0, p), 0.5, 0.5),
                    thin = 1, progress = false)
    @test run.generation == 70
    @test length(run.moments) == 70
    @test run.moments[end]["gen"] == 69
    z = run.moments[end]["mean_z"]
    @test mean(z[1:(p ÷ 2)]) - mean(z[(p ÷ 2 + 1):end]) > 1
    # moments are recorded before the optimum moves in that generation
    @test run.moments[end]["theta"] ≈ 59 .* delta
    run
end

@testset "BinaryB" begin
    base = B.Individual(20, 4, 0.8, 0.01, 0.001, 0.1)
    @test base.m == 40
    run = check_generations(B, base)
    mom = run.moments[end]
    @test all(x -> x in (0, 1), run[1].b)
    @test size(mom["G"]) == (4, 4)
    @test all(0 .< mom["h2"] .< 1)
    @test run[1].x == run[1].b * run[1].y

    # B mutation flips entries
    Random.seed!(3)
    ind = B.RandomInd(base)
    b0 = copy(ind.b)
    B.mutation!(ind, B.Binomial(0, 0.0), B.Binomial(1, 1.0))
    @test sum(ind.b .!= b0) == 1
end

@testset "Antagonistic" begin
    base = A.Individual(20, 4, 0.01, 0.001, 0.1)
    run = check_generations(A, base)
    @test all(x -> x in (-1, 0, 1), run[1].b)
    @test run[1].z == run[1].b * run[1].y
    @test sum(run.moments[end]["count_b"].count) == base.m
    @test A.classifyBj([1, -1, 0, 0]) == "Modular"   # classes use |b| first
    @test A.classifyBj([1, 1, -1, -1]) == "Antagonistic"
    @test A.classifyBj([0, 1, 0, -1]) == "Antagonistic"
    @test A.classifyBj([1, 1, 0, 0]) == "Modular"
    @test A.classifyBj([0, 0, -1, 0]) == "Local"
    @test A.classifyBj([1, 1, 1, 1]) == "Integrated"
    @test A.classifyBj([0, 0, 0, 0]) == "Neutral"
end

@testset "PleiotropicVectors" begin
    base = PV.Individual(20, 4, 0.01, 0.1, 0.1)
    run = check_generations(PV, base)
    @test run[1].z ≈ vec(sum(run[1].b, dims = 2))
    @test sum(run.moments[end]["count_b"].count) == base.m
    @test PV.classifyBj([1.0, 1.0, 0.0, 0.0]) == "Modular"
    @test PV.classifyBj([0.0, 0.0, 0.0, 0.0]) == "Neutral"

    # mutation keeps the pleiotropic direction, also for negative effects
    Random.seed!(4)
    ind = PV.RandomInd(PV.Individual(5, 4, 1.0, 1e-6, 1e-6))
    ind.y[1] = -0.5
    u = [1.0, 2.0, 0.0, 0.0] / sqrt(5)
    ind.b[:, 1] = u * ind.y[1]
    for _ in 1:20
        PV.mutation!(ind, PV.Binomial(1, 1.0))
    end
    @test ind.b[:, 1] / ind.y[1] ≈ u atol = 1e-3

    # epistasis: fast phenotype equals the explicit double sum
    Random.seed!(5)
    e = PV.random_epistasis(5, 4, 1.0)
    epi = PV.Individual(5, 4, 0.01, 0.1, 0.1, e)
    ind = PV.RandomInd(epi)
    z = vec(sum(ind.b, dims = 2))
    for k in 1:4, i in 1:10, j in 1:(i - 1)
        z[k] += e[i, j, k] * ind.b[k, i] * ind.b[k, j]
    end
    @test ind.z ≈ z
    @test_throws ArgumentError PV.Individual(5, 4, 0.01, 0.1, 0.1, rand(10, 10, 4))
    pop = PV.Population(10, epi); PV.RandomPop!(pop)
    @test copy(pop)[1].e === e   # epistasis array stays shared
    Random.seed!(6)
    run = PV.run_pop(epi, 50, ["Stab"], [3]; progress = false)
    @test run.generation == 3
end

@testset "Growth" begin
    Random.seed!(7)
    base = GR.Individual(10, 4, 0.01, 0.001, 0.1, 50)
    pop = GR.Population(200, base)
    GR.RandomPop!(pop)
    @test pop.is_adult == [ind.age > 50 for ind in pop.pop]
    c = copy(pop)
    @test all(c.pop[i] !== pop.pop[i] for i in 1:200)

    GR.changeSurface!(pop, zeros(4), Matrix(1.0I, 4, 4), zeros(4), Matrix(10.0I, 4, 4))
    GR.fitness!(pop)
    @test sum(pop.fitness[pop.is_adult]) ≈ 1
    @test sum(pop.fitness[.!pop.is_adult]) ≈ 1

    # newborns replace exactly the dead, and only survivors reproduce
    a1 = zeros(Int, 10); a2 = similar(a1)
    for _ in 1:20
        GR.next_time_step!(pop, a1, a2)
        @test isempty(pop.the_departed)
    end
    @test pop.generation == 20
    @test any(ind.age <= 20 for ind in pop.pop)

    Random.seed!(8)
    run = GR.run_pop(base, 200, ["Drift", "Stab", "Direct"], [10, 10, 30];
                     a_delta_theta = fill(0.02, 4), thin = 5, progress = false)
    @test run.generation == 50
    @test length(run.moments) == 10
    @test 0 < run.moments[end]["adults"] < 1
    @test run.moments[end]["theta"] ≈ fill(0.58, 4)  # recorded before the 30th move
end

@testset "Multilinear" begin
    Random.seed!(10)
    p, loci = 3, 6
    eps = ML.random_epsilon(p; sd = 0.5)
    @test all(k -> issymmetric(eps[k, :, :]), 1:p)
    @test_throws ArgumentError ML.Params(loci, p; mu = 0.1, mutation_cov = 0.1, epsilon = rand(p, p, p))
    @test_throws ArgumentError ML.Params(loci, p; mu = 0.1, mutation_cov = 0.1, epsilon = zeros(p, p, p + 1))
    @test_throws ArgumentError ML.Params(loci, p; mu = 0.1, mutation_cov = zeros(p, p))

    add = ML.Params(loci, p; mu = 0.1, mutation_cov = 0.1)
    epi = ML.Params(loci, p; mu = 0.1, mutation_cov = 0.1, epsilon = eps)
    zero_eps = ML.Params(loci, p; mu = 0.1, mutation_cov = 0.1, epsilon = zeros(p, p, p))
    y = randn(p, 2loci)
    Y = ML.locus_effects(y)
    @test Y[:, 2] == y[:, 3] + y[:, 4]

    # additive model: sum of allele vectors; epsilon = 0 gives the same
    @test ML.genotypic_value(add, y) ≈ vec(sum(y, dims = 2))
    @test ML.genotypic_value(zero_eps, y) ≈ ML.genotypic_value(add, y)

    # fast pairwise sum equals the explicit sum over pairs of loci
    naive = vec(sum(Y, dims = 2))
    for i in 1:loci, j in (i + 1):loci, k in 1:p
        naive[k] += Y[:, i]' * eps[k, :, :] * Y[:, j]
    end
    @test ML.genotypic_value(epi, y) ≈ naive

    # univariate case: x = sum(Y) + ε sum_{i<j} Y_i Y_j
    uni = ML.Params(loci, 1; mu = 0.1, mutation_cov = 0.1, epsilon = fill(0.3, 1, 1, 1))
    y1 = randn(1, 2loci); Y1 = vec(ML.locus_effects(y1))
    @test ML.genotypic_value(uni, y1)[1] ≈ sum(Y1) + 0.3 * sum(Y1[i] * Y1[j] for i in 1:loci for j in (i + 1):loci)

    # local effects: removing locus i changes x by its local effect
    L = ML.local_effects(epi, y)
    y0 = copy(y); y0[:, 5:6] .= 0
    @test ML.genotypic_value(epi, y) - ML.genotypic_value(epi, y0) ≈ L[:, 3]
    @test ML.local_effects(add, y) ≈ Y

    # mutation adds MvNormal(0, M) steps
    M = [1.0 0.5 0; 0.5 1 0; 0 0 0.2]
    par = ML.Params(1, p; mu = 1.0, mutation_cov = M)
    steps = reduce(hcat, begin
        ind = ML.RandomInd(ML.Individual(par)); y_before = copy(ind.y)
        ML.mutation!(ind, ML.Binomial(1, 1.0), ML.MvNormal(zeros(p), par.mutation_cov))
        vec(sum(ind.y - y_before, dims = 2))
    end for _ in 1:20_000)
    @test cov(steps') ≈ M atol = 0.05

    # parameters are shared, not copied
    pop = ML.Population(10, epi); ML.RandomPop!(pop)
    @test copy(pop)[1].par === epi
    @test all(ind -> all(iszero, ind.y), pop.pop)   # init_sigma = 0: reference genotype

    # generic generation checks, additive and epistatic
    for par in (ML.Params(20, 4; mu = 0.01, mutation_cov = 0.1, v_e = 0.5, init_sigma = 0.1),
                ML.Params(20, 4; mu = 0.01, mutation_cov = 0.1, v_e = 0.5, init_sigma = 0.1,
                          epsilon = ML.random_epsilon(4; sd = 0.05)))
        run = check_generations(ML, ML.Individual(par))
        @test run[1].x ≈ ML.genotypic_value(par, run[1].y)
    end
end

@testset "BMatrix" begin
    Random.seed!(9)
    c = BM.random_corr(5)
    @test issymmetric(c)
    @test diag(c) ≈ ones(5)
    @test isposdef(c)
    @test BM.pcasimilarity(c, c) ≈ 1
    target = abs.(BM.random_corr(4))
    b, d = BM.fit_b_matrix(target; n_cols = 30, t_steps = 20_000, max_rounds = 2)
    proj = ones(4, 4); BM.project!(b, proj)
    @test d ≈ BM.matrix_distance(target, proj)
    start = ones(4, 4); BM.project!(rand(0:1, 4, 30), start)
    @test d < BM.matrix_distance(target, start)
end

end
include("sim.jl")
