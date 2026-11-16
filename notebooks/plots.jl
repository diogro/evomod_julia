# Plotting helpers shared by the notebooks (Gadfly).
# They take any EvoMod population whose `moments` were recorded by `run_pop`.

using Gadfly
using DataFrames
using LinearAlgebra
using Statistics
using Distributions: MvNormal, pdf
using EvoMod: AVGcorr
import Cairo, Fontconfig  # PDF and PNG output

"Heatmap of a correlation matrix (diagonal blanked), coloured from -1 to 1."
function plotCorrHeatmap(mat)
    a = copy(mat)
    a[diagind(a)] .= NaN
    spy(a, Scale.ContinuousColorScale(Scale.lab_gradient(colorant"lightblue",
                                                         colorant"white",
                                                         colorant"red"),
                                      minvalue = -1, maxvalue = 1))
end

moment_range(pop, start) = start:length(pop.moments)
moment_gens(pop, range) = [pop.moments[i]["gen"] for i in range]

"Average correlation within each module and between modules through time."
function plotCorrelations(pop, start = 1; key = "corrG")
    r = moment_range(pop, start)
    avgs = [AVGcorr(pop.moments[i][key]) for i in r]
    df = DataFrame(W_1 = [a[1][1] for a in avgs], W_2 = [a[1][2] for a in avgs],
                   B = [a[2] for a in avgs], G = moment_gens(pop, r))
    plot(df, layer(y = :W_1, x = :G, Geom.line, Theme(default_color = colorant"red")),
             layer(y = :W_2, x = :G, Geom.line, Theme(default_color = colorant"green")),
             layer(y = :B, x = :G, Geom.line),
         Guide.xlabel("Generation"), Guide.ylabel("Average correlation"),
         Guide.manual_color_key("Legend", ["Within 1", "Within 2", "Between"],
                                ["red", "green", "deepskyblue"]))
end

"AVG ratio (mean within-module over between-module correlation) through time."
function plotAVGRatio(pop, start = 1; key = "corrG")
    r = moment_range(pop, start)
    df = DataFrame(W = [(a = AVGcorr(pop.moments[i][key]); mean(a[1]) / a[2]) for i in r],
                   G = moment_gens(pop, r))
    plot(df, layer(y = :W, x = :G, Geom.line),
         Guide.xlabel("Generation"), Guide.ylabel("AVG ratio"))
end

"Mean trait value and optimum of each module through time."
function plotTraits(pop, start = 1)
    r = moment_range(pop, start)
    p2 = pop.base_ind.p ÷ 2
    df = DataFrame(mod1   = [mean(pop.moments[i]["mean_z"][1:p2])       for i in r],
                   theta1 = [mean(pop.moments[i]["theta"][1:p2])        for i in r],
                   mod2   = [mean(pop.moments[i]["mean_z"][(p2 + 1):end]) for i in r],
                   theta2 = [mean(pop.moments[i]["theta"][(p2 + 1):end])  for i in r],
                   gen    = moment_gens(pop, r))
    plot(df, layer(y = :mod1, x = :gen, Geom.line, Theme(default_color = colorant"red")),
             layer(y = :theta1, x = :gen, Geom.line, Theme(default_color = colorant"darkred")),
             layer(y = :theta2, x = :gen, Geom.line, Theme(default_color = colorant"darkblue")),
             layer(y = :mod2, x = :gen, Geom.line),
         Guide.xlabel("Generation"), Guide.ylabel("Trait value"),
         Guide.manual_color_key("Legend",
                                ["Trait Module 1", "Trait Module 2", "Theta Module 1", "Theta Module 2"],
                                ["red", "deepskyblue", "darkred", "darkblue"]))
end

"Number of effect vectors in each pleiotropy class through time."
function plotPleiotropy(pop, start = 1)
    df = reduce(vcat, [pop.moments[i]["count_b"] for i in moment_range(pop, start)])
    plot(df, x = :gen, y = :count, color = :class, Geom.line,
         Guide.xlabel("Generation"), Guide.ylabel("Count"))
end

"Mean of a scalar moment (e.g. \"h2\") through time."
function plotMoment(pop, key, start = 1; f = mean)
    r = moment_range(pop, start)
    df = DataFrame(value = [f(pop.moments[i][key]) for i in r], gen = moment_gens(pop, r))
    plot(df, x = :gen, y = :value, Geom.line, Guide.xlabel("Generation"), Guide.ylabel(key))
end

"""
    plotSurfacePop(pop, surface; gen = length(pop.moments))

Recorded phenotypes at moment `gen` projected on the two main axes of the
fitness `surface`, over the contours of the surface. Points are coloured by
the current fitness.
"""
function plotSurfacePop(pop, surface; gen = length(pop.moments))
    zs = pop.moments[gen]["zs"]
    theta = pop.moments[gen]["theta"]
    eg = eigen(Symmetric(Matrix(surface.Σ)))
    V = eg.vectors[:, (end - 1):end]
    zs_eig = zs * V
    s_theta = V' * theta
    s_omega = Diagonal(eg.values[(end - 1):end])
    zs_df = sort(DataFrame(x = zs_eig[:, 1], y = zs_eig[:, 2], fit = pop.fitness), :fit)
    lim(k) = (min(s_theta[k] - 2sqrt(s_omega[k, k]), minimum(zs_eig[:, k])),
              max(s_theta[k] + 2sqrt(s_omega[k, k]), maximum(zs_eig[:, k])))
    xs = range(lim(1)..., length = 150)
    ys = range(lim(2)..., length = 150)
    dens = MvNormal(s_theta, Matrix(s_omega))
    Z = [pdf(dens, [x, y]) for x in xs, y in ys]
    plot(layer(z = Z, x = xs, y = ys, Geom.contour),
         layer(zs_df, x = :x, y = :y, color = :fit, Geom.point))
end

"Correlations, traits and pleiotropy classes stacked in one PDF in `dir`."
function all_plots(pop, file; start = 1, dir = "figures")
    mkpath(dir)
    plots = [plotCorrelations(pop, start), plotTraits(pop, start)]
    haskey(pop.moments[end], "count_b") && push!(plots, plotPleiotropy(pop, start))
    draw(PDF(joinpath(dir, file), 6inch, 2inch * length(plots)), vstack(plots...))
end
