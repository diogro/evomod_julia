"""
Search for binary ontogenetic matrices whose row correlations match a target
correlation matrix (simulated annealing), plus random correlation matrices and
matrix comparison helpers.
"""
module BMatrix

using LinearAlgebra
using Random

export random_corr, pcasimilarity, matrix_distance, project!, fit_b_matrix

"""
    random_corr(num_traits; ke = 1e-3)

Random correlation matrix (method of Numpacharoen & Atsawarungruangkit),
with traits in random order.
"""
function random_corr(num_traits::Integer; ke = 1e-3)
    n = num_traits
    random_corr = zeros(n, n)
    b = Float64[i >= j for i in 1:n, j in 1:n]
    random_corr[2:n, 1] = 2 .* rand(n - 1) .- 1
    b[2:n, 1] = random_corr[2:n, 1]
    for i in 2:n
        b[i, 2:i] .= sqrt(1 - random_corr[i, 1]^2)
    end
    for i in 3:n, j in 2:(i - 1)
        b1 = dot(b[j, 1:(j - 1)], b[i, 1:(j - 1)])
        b2 = b[j, j] * b[i, j]
        z = b1 + b2
        y = b1 - b2
        random_corr[i, j] = b2 < ke ? b1 : y + (z - y) * rand()
        cosinv = (random_corr[i, j] - b1) / b2
        if isfinite(cosinv)
            if cosinv > 1
                b[i, (j + 1):n] .= 0
            elseif cosinv < -1
                b[i, j] = -b[i, j]
                b[i, (j + 1):n] .= 0
            else
                b[i, j] *= cosinv
                b[i, (j + 1):n] .*= sqrt(1 - cosinv^2)
            end
        end
    end
    random_corr = random_corr + random_corr' + I
    perm = shuffle(1:n)
    random_corr[perm, perm]
end

"PCA similarity between two covariance matrices (eigenvalue-weighted)."
function pcasimilarity(cov_x::AbstractMatrix, cov_y::AbstractMatrix)
    eg_x = eigen(Symmetric(cov_x))
    eg_y = eigen(Symmetric(cov_y))
    total_var = dot(eg_x.values, eg_y.values)
    sum((eg_x.values * eg_y.values') .* (eg_x.vectors' * eg_y.vectors) .^ 2) / total_var
end

"Sum of absolute element-wise differences."
matrix_distance(cov_x, cov_y) = sum(abs.(cov_x - cov_y))

"Row correlation of `b_matrix` (cosine of rows) written into `b_projection`."
function project!(b_matrix, b_projection)
    b_projection .= b_matrix * b_matrix'
    d = diag(b_projection)
    b_projection ./= sqrt.(d * d')
end

"""
    fit_b_matrix(target_matrix; n_cols = 100, t_init = 1e-3, t_end = 1e-7,
                 t_steps = 1_000_000, tol = 2, max_rounds = 100)

Simulated annealing over binary matrices `b` (traits x `n_cols`) so that the
row correlations of `b` approach `target_matrix`. One round cools the
temperature linearly from `t_init` to `t_end` in `t_steps` single-entry flips;
rounds repeat until the distance is at most `tol` or `max_rounds` is reached.
Returns `(b, distance)`.
"""
function fit_b_matrix(target_matrix::AbstractMatrix; n_cols = 100, t_init = 1e-3,
                      t_end = 1e-7, t_steps = 1_000_000, tol = 2, max_rounds = 100)
    p = size(target_matrix, 1)
    b_matrix = rand(0:1, p, n_cols)
    b_projection = ones(p, p)
    project!(b_matrix, b_projection)
    dist = matrix_distance(target_matrix, b_projection)
    rounds = 0
    while !(dist <= tol) && rounds < max_rounds
        rounds += 1
        for t in range(t_init, t_end, length = t_steps)
            i = rand(1:p)
            j = rand(1:n_cols)
            b_matrix[i, j] = 1 - b_matrix[i, j]
            project!(b_matrix, b_projection)
            new_dist = matrix_distance(target_matrix, b_projection)
            # Metropolis rule: always accept improvements, accept a worse
            # matrix with probability exp(-increase / t)
            if new_dist < dist || rand() < exp((dist - new_dist) / t)
                dist = new_dist
            else
                b_matrix[i, j] = 1 - b_matrix[i, j]
            end
        end
    end
    project!(b_matrix, b_projection)
    b_matrix, matrix_distance(target_matrix, b_projection)
end

end
