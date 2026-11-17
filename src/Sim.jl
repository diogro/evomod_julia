"""
The evomod2 engine in Julia: the same models, regimes, life cycles,
statistics and analysis tools as the R package evomod2
(https://github.com/diogro/evomod2), with the same function names.

    using EvoMod.Sim
    model = evomod_model(additive(50, 4), mutation(0.0005, 0.2); env_sd = 0.8)
    pop = evolve(model, [drift(500), stabilizing(2000; omega = 10)]; n = 1000, record_every = 50, seed = 1)
    trait_means(pop); g_matrices(pop)

Components: architectures `additive`, `multilinear` (with `random_epsilon` or
`pairwise_epsilon`), `binary_b`, `contribution_matrix`; mutation `mutation`,
`hoc_mutation`, `binary_b_mutation`, `contribution_mutation`; recombination
`free_recombination`, `genetic_map`, `linkage`; life cycles `wright_fisher`,
`moran`, `viability_selection`; regimes `drift`, `stabilizing`,
`directional`, `fluctuating`. Results: `trait_means`, `variances`,
`g_matrices`, `p_matrices`, `mean_fitness`, `genomes`. Analysis:
`g_decomposition`, `shuffle_genomes`, `additive_covariance`,
`mutational_covariance`, `genotypic_values`, `local_effects`, `avg_corr`,
`autonomy`, `mean_autonomy`, `eigen_variance`.

Random numbers come from Julia's default generator, so results agree with
evomod2 in distribution, not number for number.
"""
module Sim

using DataFrames
using Distributions
using LinearAlgebra
using Random
using Statistics

export additive, multilinear, binary_b, contribution_matrix, random_epsilon, pairwise_epsilon,
       mutation, hoc_mutation, binary_b_mutation, contribution_mutation,
       free_recombination, genetic_map, linkage,
       wright_fisher, moran, viability_selection,
       evomod_model, drift, stabilizing, directional, fluctuating, evolve,
       trait_means, variances, mean_fitness, g_matrices, p_matrices, genomes,
       genotypic_values, genotypic_value, local_effects, g_decomposition, shuffle_genomes,
       additive_covariance, mutational_covariance, avg_corr, autonomy, mean_autonomy,
       eigen_variance

include("sim/components.jl")
include("sim/evolve.jl")
include("sim/analysis.jl")

end
