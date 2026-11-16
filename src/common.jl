"""
Helpers shared by all EvoMod models: fitness surfaces, modularity statistics
and selection-regime parsing.
"""
module Common

using LinearAlgebra
using Statistics

export twoModuleMatrix, lowerTri, AVGcorr, AVGratio, is_drift, is_directional

"""
    twoModuleMatrix(vars, cor1, cor2)

Covariance matrix with two modules (first and second half of the traits),
within-module correlations `cor1` and `cor2`, no correlation between modules,
and variances `vars`.
"""
function twoModuleMatrix(vars::AbstractVector{<:Real}, cor1, cor2)
    n = length(vars)
    n_1 = n ÷ 2
    module_1 = [ones(n_1); zeros(n - n_1)]
    module_2 = [zeros(n_1); ones(n - n_1)]
    omega = (module_1 * module_1') * cor1 + (module_2 * module_2') * cor2
    omega[diagind(omega)] .= 1
    omega .* (sqrt.(vars) * sqrt.(vars)')
end

"Upper-triangle elements (i < j) of a square matrix, row by row."
function lowerTri(mat)
    p = size(mat, 1)
    [mat[i, j] for i in 1:p for j in 1:p if i < j]
end

default_modules(p) = [collect(1:(p ÷ 2)), collect((p ÷ 2 + 1):p)]

"""
    AVGcorr(mat; modules)

Average within-module correlation for each module and the average
between-module correlation. Returns `(avg_plus::Vector, avg_minus)`.
Modules default to the first and second half of the traits.
"""
function AVGcorr(mat; modules = default_modules(size(mat, 1)))
    p = size(mat, 1)
    avg_plus = [mean(lowerTri(mat[mod, mod])) for mod in modules]
    in_module = falses(p, p)
    for mod in modules
        in_module[mod, mod] .= true
    end
    avg_plus, mean(mat[.!in_module])
end

"AVG ratio: mean within-module correlation over between-module correlation."
function AVGratio(mat; modules = default_modules(size(mat, 1)))
    avg_plus, avg_minus = AVGcorr(mat; modules = modules)
    mean(avg_plus) / avg_minus
end

regime_name(regime) = lowercase(strip(regime))

"Regime without selection (\"Drift\")."
is_drift(regime) = regime_name(regime) == "drift"

"Regime with a moving optimum (any name starting with \"Dire\", e.g. \"Direct\")."
is_directional(regime) = startswith(regime_name(regime), "dire")

end
