module AdaptiveTransportMap

using DocStringExtensions
using LinearAlgebra
using OrdinaryDiffEq
using SpecialFunctions
using ProgressMeter
using BenchmarkTools
using ForwardDiff
using JLD
using Polynomials
using Distributions
using QuadGK
using TensorOperations
using LoopVectorization
using Optim
using NLsolve
using LineSearches
using MLDataUtils
using ThreadPools
using Statistics
using PlotUtils
using RecipesBase
using ColorTypes
using LaTeXStrings

include("tools/get.jl")
include("tools/parallel.jl")
include("tools/ADtools.jl")
include("tools/metric.jl")
include("tools/product.jl")
include("tools/tools.jl")
include("tools/normal.jl")
include("tools/clenshaw_curtis.jl")
# Tools to apply a linear transformation
include("tools/transform.jl")
# Tools for Banana distribution
include("tools/banana.jl")
# Tools for mixture of Gaussian distributions
include("tools/mixture.jl")
include("tools/view.jl")

# Tools for state-space model
include("statespace/system.jl")

# Tools for data assimilation
include("DA/inflation.jl")
include("DA/model.jl")
include("DA/gaspari.jl")
include("DA/seqfilter.jl")
include("DA/seqassim.jl")
include("DA/postprocess.jl")

# Tools for EnKF
include("enkf/senkf.jl")
include("enkf/etkf.jl")

# Tools for lorenz63
include("lorenz63/lorenz63.jl")

# Tools for lorenz96
include("lorenz96/lorenz96.jl")


# Rectifier tools
include("hermitemap/rectifier.jl")

# Hermite Polynomials
include("hermitefunction/hermite.jl")
include("hermitefunction/polyhermite.jl")
include("hermitefunction/phypolyhermite.jl")
include("hermitefunction/propolyhermite.jl")


# Hermite Functions
include("hermitefunction/phyhermite.jl")
include("hermitefunction/prohermite.jl")


# Uni and Multi Basis function
include("hermitemap/basis.jl")
include("hermitemap/multibasis.jl")
include("hermitemap/multifunction.jl")
include("hermitemap/expandedfunction.jl")
include("hermitemap/expandedfunction2.jl")
include("hermitemap/storage.jl")

# Integrated positive function
include("hermitemap/integratedfunction.jl")

# ReducedMargin
include("margin/reducedmargin.jl")
include("margin/totalorder.jl")


# KR-rearrangement and TransportMap structure
include("hermitemap/hermitemapcomponent.jl")
include("hermitemap/linhermitemapcomponent.jl")
# Tools for fast inversion
include("hermitemap/inverse.jl")
include("hermitemap/hybridinverse.jl")
# Tools for greedyfit and optimization
include("hermitemap/qr.jl")
include("hermitemap/qraccelerated.jl")
include("hermitemap/precond.jl")
include("hermitemap/greedyfit.jl")
include("hermitemap/optimize.jl")

include("hermitemap/hermitemap.jl")
include("hermitemap/totalordermap.jl")
include("hermitemap/stochmapfilter.jl")


# Tools for radial maps
# include("radialmap/function.jl")
# include("radialmap/separablecomponent.jl")
# include("radialmap/mapcomponent.jl")
# include("radialmap/sparsemapcomponent.jl")
# include("radialmap/map.jl")
# include("radialmap/weights.jl")
# include("radialmap/quantile.jl")
# include("radialmap/cost.jl")
# include("radialmap/solver.jl")
# include("radialmap/optimize.jl")
# include("radialmap/iterativeoptimize.jl")
# include("radialmap/invert.jl")
# include("radialmap/TMap.jl")


include("tools/plot_recipes.jl")

end # module
