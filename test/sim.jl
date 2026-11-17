# Tests of EvoMod.Sim, ported from the evomod2 R test suite.
using EvoMod.Sim
using EvoMod.Sim: locus_effects, ntraits, mutate!, mutate_one!, Fenwick, draw, add!
using LinearAlgebra, Random, Statistics, Test

const S = EvoMod.Sim

naive_pairwise(y, eps) = begin
    Y = locus_effects(y); p, L = size(Y)
    x = vec(sum(Y, dims = 2))
    for i in 1:(L - 1), j in (i + 1):L, k in 1:p
        x[k] += Y[:, i]' * eps[k, :, :, i, j] * Y[:, j]
    end
    x
end

@testset "Sim" begin

@testset "architectures" begin
    Random.seed!(1)
    y = randn(3, 12)
    @test genotypic_value(additive(6, 3), y) ≈ vec(sum(y, dims = 2))
    e = random_epsilon(3; sd = 0.5)
    Y = locus_effects(y)
    naive = vec(sum(Y, dims = 2))
    for i in 1:6, j in (i + 1):6, k in 1:3
        naive[k] += Y[:, i]' * e[k, :, :] * Y[:, j]
    end
    @test genotypic_value(multilinear(6, 3, e), y) ≈ naive
    pe = pairwise_epsilon(6, 3; sd = 0.5)
    @test genotypic_value(multilinear(6, 3, pe), y) ≈ naive_pairwise(y, pe)
    # identical tensors for every pair = shared model
    same = zeros(3, 3, 3, 6, 6)
    for i in 1:5, j in (i + 1):6
        same[:, :, :, i, j] = e
    end
    @test genotypic_value(multilinear(6, 3, same), y) ≈ genotypic_value(multilinear(6, 3, e), y)
    # local effects: change from removing a locus
    for arch in (multilinear(6, 3, e), multilinear(6, 3, pe), additive(6, 3))
        L = local_effects(arch, y)
        for l in 1:6
            y0 = copy(y); y0[:, (2l - 1):(2l)] .= 0
            @test genotypic_value(arch, y) - genotypic_value(arch, y0) ≈ L[:, l]
        end
    end
    # binary B: x = B y
    g = [randn(1, 20); Float64.(rand(4, 20) .< 0.5)]
    @test genotypic_value(binary_b(10, 4), g) ≈ g[2:end, :] * g[1, :]
    # contribution matrix: x = C a
    a = [0.5, -2.0]; C = [1 0.2; -1 0; 3 0.5]
    gc = reshape(repeat([a; vec(C)] ./ 2, inner = 2), 1, :)
    arch = contribution_matrix(3, 2; init_contributions = Matrix(1.0I, 3, 2))
    @test genotypic_value(arch, gc) ≈ C * a
    Lc = local_effects(arch, gc)
    for l in 1:8
        g0 = copy(gc); g0[:, (2l - 1):(2l)] .= 0
        @test genotypic_value(arch, gc) - genotypic_value(arch, g0) ≈ Lc[:, l]
    end
    @test_throws ArgumentError multilinear(6, 3, rand(3, 3, 3))
    @test_throws ArgumentError evomod_model(binary_b(5, 2), mutation(0.1, 0.1))
    @test_throws ArgumentError evomod_model(additive(5, 2), binary_b_mutation(0.1, 0.1, 0.1))
    @test_throws ArgumentError evomod_model(contribution_matrix(2), mutation(0.1, 0.1))
end

@testset "mutation" begin
    Random.seed!(2)
    M = [1 0.5 0; 0.5 1 0; 0 0 0.2]
    m = evomod_model(additive(1, 3), mutation(1.0, M))
    steps = reduce(hcat, [begin g = zeros(3, 2); mutate_one!(m.mutation, g); vec(sum(g, dims = 2)) end for _ in 1:20000])
    @test cov(steps') ≈ M atol = 0.05
    # house of cards forgets the old effect; persistence regresses
    h = evomod_model(additive(500, 2), hoc_mutation(1.0, 0.5))
    g = fill(100.0, 2, 1000); mutate!(h.mutation, g)
    changed = g[:, vec(any(g .!= 100, dims = 1))]
    @test abs(mean(changed)) < 0.05
    @test std(changed) ≈ 0.5 atol = 0.03
    pm = evomod_model(additive(100000, 1), mutation(0.01, 0.1; persistence = 0.5))
    g = fill(10.0, 1, 200000); mutate!(pm.mutation, g)
    @test median(g[g .!= 10]) ≈ 5 atol = 0.05
    # binary B: flip rate
    bb = binary_b_mutation(0.1, 0.5, 0.1; b_mutation = :flip)
    g = zeros(5, 20000); mutate!(bb, g)
    # entries hit twice flip back: the fraction of 1s is (1 - exp(-2 rate)) / 2
    @test mean(g[2:end, :]) ≈ (1 - exp(-0.2)) / 2 atol = 0.004
    @test all(x -> x in (0.0, 1.0), g[2:end, :])
    @test Sim.expected_mutations(bb, g) ≈ 0.1 * 20000 + 0.1 * 20000 * 4
    @test_throws ArgumentError mutation(0.1, 0.1; persistence = 2)
end

@testset "recombination" begin
    m = evomod_model(additive(6, 1), mutation(0.0, 0.1); recombination = genetic_map(2, 0.5))
    d = 0.5 / 2
    @test m.recombination.rates ≈ [fill((1 - exp(-2d)) / 2, 2); 0.5; fill((1 - exp(-2d)) / 2, 2)]
    coupled(L, n) = (g = zeros(1, 2L, n); g[:, 1:2:(2L), :] .= 1; g)
    rates = [0.05, 0.3, 0.5, 0.0]
    m = evomod_model(additive(5, 1), mutation(0.0, 0.1); recombination = linkage(rates))
    pop = evolve(m, drift(1); n = 20000, initial = coupled(5, 20000), seed = 3)
    g = genomes(pop)
    for l in 1:4
        sw = mean([g[1, 2l - 1, :] .!= g[1, 2l + 1, :]; g[1, 2l, :] .!= g[1, 2l + 2, :]])
        @test abs(sw - rates[l]) < 0.01
    end
    # LD decay: D1 = 0.25 (1 - 2r) from double heterozygotes, then (1 - r) per generation
    r = 0.1; n = 5000
    m = evomod_model(additive(2, 1), mutation(0.0, 0.1); recombination = linkage([r]))
    pop = evolve(m, drift(1); n = n, initial = coupled(2, n), seed = 4)
    D(g) = (a = [g[1, 1, :]; g[1, 2, :]]; b = [g[1, 3, :]; g[1, 4, :]]; mean(a .* b) - mean(a) * mean(b))
    @test D(genomes(pop)) ≈ 0.25 * (1 - 2r) atol = 0.01
    for _ in 2:10
        evolve(pop, drift(1))
    end
    @test D(genomes(pop)) ≈ 0.25 * (1 - 2r) * (1 - r)^9 rtol = 0.1
end

@testset "life cycles" begin
    Random.seed!(5)
    w = [1, 0, 3, 0.5, 2, 0, 1.5]
    f = Fenwick(Float64.(w))
    counts = zeros(7)
    for _ in 1:70000
        counts[draw(f)] += 1
    end
    @test counts ./ 70000 ≈ w ./ sum(w) atol = 0.01
    add!(f, 3, -3.0)
    @test all(draw(f) != 3 for _ in 1:5000)
    # neutral drift: genic variance decays by (1 - 1/(2 Ne)) per generation
    n, t = 100, 40
    decay(lc) = mean(begin
        m = evomod_model(additive(200, 1), mutation(0.0, 0.1); init_sd = 1, life_cycle = lc)
        pop = evolve(m, drift(t); n = n, record_every = t)
        g = g_matrices(pop; component = :genic)
        g[t][1, 1] / g[0][1, 1]
    end for _ in 1:8)
    Random.seed!(6)
    @test decay(wright_fisher()) ≈ (1 - 1 / (2n))^t rtol = 0.05
    @test decay(viability_selection()) ≈ (1 - 1 / (2n))^t rtol = 0.05
    @test decay(moran()) ≈ (1 - 1 / n)^t rtol = 0.07
    # viability selection: mothers from the first half, fathers from the second
    g = zeros(1, 4, 40); g[1, 1:2, :] .= repeat(collect(1.0:40)', 2)
    m = evomod_model(additive(2, 1), mutation(0.0, 0.1); life_cycle = viability_selection())
    kids = genomes(evolve(m, drift(1); n = 40, initial = g, seed = 7))
    @test all(kids[1, 1, :] .> 20) && all(kids[1, 2, :] .<= 20)
    @test_throws ArgumentError evolve(m, drift(1); n = 41)
    # Moran overlap: about exp(-1) of individuals survive a generation
    m = evomod_model(additive(5, 1), mutation(0.0, 0.1); init_sd = 1, life_cycle = moran())
    pop = evolve(m, drift(1); n = 2000, seed = 8)
    before = Set(eachslice(copy(genomes(pop)), dims = 3))
    evolve(pop, drift(1))
    @test mean(s in before for s in eachslice(genomes(pop), dims = 3)) ≈ exp(-1) atol = 0.03
end

@testset "regimes and records" begin
    m = evomod_model(additive(10, 2), mutation(0.01, 0.1); env_sd = 0.5, init_sd = 0.2)
    pop = evolve(m, [drift(10), directional(10; delta_theta = [0.5, -0.5], omega = 5), stabilizing(10; omega = 5)];
                 n = 40, record_every = 5, seed = 9)
    tm = trait_means(pop)
    @test unique(tm.generation) == 0:5:30
    @test tm.theta[tm.trait .== "trait1"] ≈ [0, 0, 0, 2.5, 5, 5, 5]
    @test pop.theta ≈ [5, -5]
    evolve(pop, stabilizing(5; omega = 5, theta = [1, 1]))
    @test pop.theta == [1.0, 1.0]
    # fluctuating: jumps after every `every` generations, structures, bounds
    m4 = evomod_model(additive(5, 4), mutation(0.01, 0.1); env_sd = 0.5, init_sd = 0.1)
    pop = evolve(m4, fluctuating(200; every = 10, step = 0.2, omega = 1, bounds = nothing); n = 20, seed = 10)
    th = reduce(hcat, pop.records.theta)
    changes = findall(vec(sum(abs.(diff(th, dims = 2)), dims = 1)) .> 0)
    @test all(changes .% 10 .== 1)
    @test all(th[1, :] .≈ th[4, :])
    pop = evolve(m4, fluctuating(3000; every = 10, step = 0.2, omega = 1, bounds = nothing,
                                 structure = [[1, 2], [3, 4]]); n = 20, record_every = 10, seed = 11)
    th = reduce(hcat, pop.records.theta)
    @test th[1, :] ≈ th[2, :] && th[3, :] ≈ th[4, :]
    @test abs(cor(diff(th[1, :]), diff(th[3, :]))) < 0.15
    pop = evolve(m4, fluctuating(20000; every = 10, step = 0.2, omega = 1, structure = :independent);
                 n = 10, record_every = 10, seed = 12)
    th = reduce(hcat, pop.records.theta)
    @test all(-1 - 1e-9 .<= th .<= 1 + 1e-9)
    x = th[1, :]; top = findall(abs.(x[1:(end - 1)] .- 1) .< 1e-9)
    @test mean(abs.(x[top .+ 1] .- 1) .< 1e-9) ≈ 0.5 atol = 0.08
    # reproducibility and continuation
    m = evomod_model(binary_b(10, 3), binary_b_mutation(0.01, 0.2, 0.005); env_sd = 0.5, init_sd = 0.2,
                     life_cycle = moran())
    regs = [stabilizing(10; omega = 5), fluctuating(20; every = 5, step = 0.2, omega = 5)]
    one = evolve(m, regs; n = 60, record_every = 5, seed = 13)
    two = evolve(evolve(m, regs[1:1]; n = 60, record_every = 5, seed = 13), regs[2:2])
    @test one.genomes == two.genomes
    @test one.records.G == two.records.G
    @test mean_fitness(one).mean_fitness[end] > 0
end

@testset "analysis" begin
    m = evomod_model(additive(10, 2), mutation(0.02, 0.3); init_sd = 0.4, recombination = linkage(fill(0.01, 9)))
    pop = evolve(m, stabilizing(200; omega = 2); n = 200, seed = 14)
    exact = g_decomposition(pop)
    @test exact.G ≈ exact.genic + exact.ld
    Random.seed!(15)
    shuf = g_decomposition(pop; method = :shuffle, shuffles = 400)
    @test shuf.genic ≈ exact.genic rtol = 0.05
    @test g_matrices(pop; component = :genic)[200] ≈ exact.genic
    a = additive_covariance(pop)
    @test a.additive ≈ a.G atol = 1e-8
    Random.seed!(16)
    me = evomod_model(multilinear(10, 2, pairwise_epsilon(10, 2; sd = 2)), mutation(0.01, 0.2); env_sd = 1, init_sd = 0.3)
    pe = evolve(me, stabilizing(50; omega = 20); n = 300, seed = 17)
    a = additive_covariance(pe)
    @test all(diag(a.additive) .< diag(a.G))
    @test_throws ArgumentError g_decomposition(pe)
    # multilinear mutation effects scale as (1 + eps (S - Y_l))
    L, v, eps, s = 10, 0.1, 0.3, 0.05
    mm = evomod_model(multilinear(L, 1, fill(eps, 1, 1, 1)), mutation(0.01, s))
    p1 = evolve(mm, drift(1); n = 20, initial = fill(v, 1, 2L, 20), seed = 18)
    p1.genomes .= v
    Random.seed!(19)
    mc = mutational_covariance(p1; n = 40000)
    @test mc.per_mutation[1, 1] ≈ s^2 * (1 + eps * 2v * (L - 1))^2 rtol = 0.03
    # neutral house of cards equilibrium
    n, mu, loci, sd = 200, 0.00125, 100, 0.5
    th = 4 * n * mu
    h = evomod_model(additive(loci, 1), hoc_mutation(mu, sd); init_sd = sd * sqrt(th / (1 + th)))
    ph = evolve(h, drift(3000); n = n, record_every = 100, seed = 20)
    gg = g_matrices(ph; component = :genic)
    @test mean(gg[t][1, 1] for t in 600:100:3000) ≈ 2 * loci * sd^2 * th / (1 + th) rtol = 0.1
    # matrix statistics
    G = [2 0.8 0.3; 0.8 1 0.1; 0.3 0.1 0.5]
    @test autonomy(G, eigvecs(G)) ≈ ones(3)
    @test sum(eigen_variance(G)) ≈ 1
    c = [1 .8 .1 .1; .8 1 .1 .1; .1 .1 1 .6; .1 .1 .6 1]
    ac = avg_corr(c)
    @test ac.within ≈ [0.8, 0.6] && ac.between ≈ 0.1 && ac.avg_ratio ≈ 7
end

end
