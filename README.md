EvoMod
======

Individual-based simulations of the evolution of modularity in complex traits,
following Melo, D., & Marroig, G. (2015). Directional selection can drive the
evolution of modularity in complex traits. *PNAS* 112 (2) 470-475,
doi:10.1073/pnas.1322632112.

The original simulations of the paper are the C code in
[lem-usp/evomod](https://github.com/lem-usp/evomod). This repository is a Julia
package with that model and several extensions:

| Module | Model |
|---|---|
| `EvoMod.BinaryB` | Binary ontogenetic matrix `b`, `z = b * y + e` (as the C model) |
| `EvoMod.Antagonistic` | Entries of `b` in {-1, 0, 1}; classification of effect vectors |
| `EvoMod.PleiotropicVectors` | Continuous pleiotropic vectors, optional pairwise epistasis |
| `EvoMod.Growth` | Overlapping generations: juveniles, adults, mortality |
| `EvoMod.BMatrix` | Annealing search for a `b` whose row correlations match a target |
| `EvoMod.Multilinear` | Allelic effect vectors; additive, or pairwise multilinear epistasis (Hansen & Wagner 2001) |

Each model module documents its assumptions (`?EvoMod.BinaryB` in the REPL).

This package is used to prototype models and to cross-check the R package
[evomod2](https://github.com/diogro/evomod2).

### EvoMod.Sim: the evomod2 engine in Julia

`EvoMod.Sim` has everything evomod2 has, with the same function names:

| Component | Functions |
|---|---|
| Architectures | `additive`, `multilinear` (with `random_epsilon` or `pairwise_epsilon`), `binary_b`, `contribution_matrix` |
| Mutation | `mutation`, `hoc_mutation`, `binary_b_mutation`, `contribution_mutation` (`persistence` keyword) |
| Recombination | `free_recombination`, `genetic_map`, `linkage` |
| Life cycles | `wright_fisher`, `moran`, `viability_selection` |
| Regimes | `drift`, `stabilizing`, `directional`, `fluctuating` |
| Results | `trait_means`, `variances`, `g_matrices` (`component = :genic` / `:ld`), `p_matrices`, `mean_fitness`, `genomes` |
| Analysis | `g_decomposition` (`method = :shuffle`), `shuffle_genomes`, `additive_covariance`, `mutational_covariance`, `genotypic_values`, `local_effects`, `avg_corr`, `autonomy`, `mean_autonomy`, `eigen_variance` |

```julia
julia> using EvoMod.Sim, LinearAlgebra
julia> model = evomod_model(binary_b(50, 4), binary_b_mutation(0.0005, 0.2, 0.0001);
                            env_sd = 0.8, init_sd = 0.2, life_cycle = moran())
julia> pop = evolve(model, [drift(500), stabilizing(2000; omega = 10),
                            fluctuating(2000; every = 10, step = 0.2, omega = 10, structure = [[1, 2], [3, 4]])];
                    n = 1000, record_every = 50, seed = 1)
julia> trait_means(pop); g_matrices(pop); mutational_covariance(pop)
```

Keyword arguments replace R's named arguments (`stabilizing(2000; omega = 10)`),
symbols replace strings (`structure = :independent`, `method = :shuffle`),
and recorded matrices are `Dict`s keyed by generation. Random numbers come
from Julia's generator, so results agree with evomod2 in distribution, not
number for number; `validation/cross_check_sim.R` in evomod2 compares the
two.

### Other models

The original models of this package are submodules of their own. The multilinear model, for example:

```julia
julia> using EvoMod, EvoMod.Multilinear, LinearAlgebra
julia> eps = random_epsilon(2; sd = 0.05)          # p x p x p, symmetric in the last two indices
julia> par = Params(50, 2; mu = 0.005, mutation_cov = 0.05, v_e = 1.0, epsilon = eps)
julia> pop = run_pop(par, 1000, ["Stab", "Direct"], [2000, 2000];
                     delta_theta = [0.005, 0.0], omega = 20 * Matrix(1.0I, 2, 2), thin = 20)
```

Requirements
------------

Julia 1.10 or later for the package, 1.11 or later for the notebooks
(install with [juliaup](https://github.com/JuliaLang/juliaup)).

Using the package
-----------------

```julia
julia> ]activate .
julia> ]instantiate
julia> using EvoMod, EvoMod.BinaryB, LinearAlgebra
julia> ind = Individual(100, 8, 0.8, 0.0005, 0.0001, 0.02)  # loci, traits, v_e, mu, mu_b, mu_sigma
julia> p = 8; delta = 0.02 * [ones(4); -ones(4)]
julia> pop = run_pop(ind, 1000, ["Drift", "Stab", "Direct"], [500, 1500, 2000];
                     delta_theta = delta, omega = 10 * Matrix(1.0I, p, p), thin = 100)
julia> AVGcorr(pop.moments[end]["corrG"])   # (within-module correlations, between-module)
```

`run_pop` takes a list of regimes ("Drift": no selection; names starting with
"Dire": the optimum moves by `delta_theta` each generation; anything else:
stabilizing selection) and records `moments(pop)` every `thin` generations in
`pop.moments`.

Run the tests with `]test`.

Notebooks
---------

`notebooks/` has one notebook per model, with the parameters used in the
original explorations. They use their own environment (`notebooks/Project.toml`,
which loads this package from `..`) and Gadfly for plots; figures are written to
`notebooks/figures/`. To run them in Jupyter, install the Julia kernel once:

```julia
using Pkg; Pkg.add("IJulia"); using IJulia; installkernel("Julia")
```

The default parameters are for full runs (thousands of individuals and
generations) and can take a long time; reduce `n_e` and the generation counts
to explore.
