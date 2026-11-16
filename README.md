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
evomod2. The multilinear model, for example:

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
