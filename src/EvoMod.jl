"""
Individual-based simulations of the evolution of modularity
(Melo & Marroig 2015, PNAS) and their extensions.

Each model is a submodule with the same interface (`Individual`,
`Population`, `run_pop`, `moments`, ...):

- `EvoMod.BinaryB`: binary ontogenetic matrix with environmental noise (as the C model)
- `EvoMod.Antagonistic`: ontogenetic matrix entries in {-1, 0, 1}
- `EvoMod.PleiotropicVectors`: continuous pleiotropic vectors, optional epistasis
- `EvoMod.Growth`: overlapping generations with juveniles and adults
- `EvoMod.BMatrix`: annealing search for B matrices matching a correlation matrix
- `EvoMod.Multilinear`: multivariate allelic effects, additive or with pairwise multilinear epistasis

Load one model at a time with `using EvoMod.BinaryB`; shared helpers
(`twoModuleMatrix`, `AVGcorr`, ...) are in `EvoMod.Common`.
"""
module EvoMod

include("common.jl")
include("BinaryB.jl")
include("Antagonistic.jl")
include("PleiotropicVectors.jl")
include("Growth.jl")
include("BMatrix.jl")
include("Multilinear.jl")

using .Common
export Common, BinaryB, Antagonistic, PleiotropicVectors, Growth, BMatrix, Multilinear,
       twoModuleMatrix, lowerTri, AVGcorr, AVGratio

end
