module AdaptiveTransportMap

using OrdinaryDiffEq
using LinearAlgebra, SpecialFunctions
using Random
using ProgressMeter
using BenchmarkTools
using ForwardDiff
using JLD
using Polynomials
using Distributions
using QuadGK
using TensorOperations
using LoopVectorization
using FastGaussQuadrature
using Optim
using NLsolve
using LineSearches
using MLDataUtils
# using MLDataPattern
using ThreadPools
using QRupdate

using PlotUtils
using RecipesBase
using ColorTypes
using LaTeXStrings
using Quadrature


include("tools/parallel.jl")
include("tools/ADtools.jl")
include("tools/metric.jl")
include("tools/product.jl")
include("tools/tools.jl")
include("tools/normal.jl")
include("tools/clenshaw_curtis.jl")
include("tools/adaptiveCC.jl")
# Tools to apply a linear transformation
include("tools/transform.jl")
# Tools for Banana distribution
include("tools/banana.jl")
# Tools for mixture of Gaussian distributions
include("tools/mixture.jl")
include("tools/view.jl")

# Tools for state-space model
include("dynamic/system.jl")

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
include("mapcomponent/rectifier.jl")

# Hermite Polynomials
include("hermitefunction/hermite.jl")
include("hermitefunction/polyhermite.jl")
include("hermitefunction/phypolyhermite.jl")
include("hermitefunction/propolyhermite.jl")


# Hermite Functions
include("hermitefunction/phyhermite.jl")
include("hermitefunction/prohermite.jl")


# Uni and Multi Basis function
include("mapcomponent/basis.jl")
include("mapcomponent/multibasis.jl")
include("mapcomponent/multifunction.jl")
include("mapcomponent/expandedfunction.jl")
include("mapcomponent/parametric.jl")
include("mapcomponent/storage.jl")

# Integrated positive function
include("mapcomponent/integratedfunction.jl")

# ReducedMargin
include("margin/reducedmargin.jl")
include("margin/totalorder.jl")


# KR-rearrangement and TransportMap structure
include("mapcomponent/hermitemapcomponent.jl")
include("mapcomponent/linhermitemapcomponent.jl")
# Tools for fast inversion
include("mapcomponent/inverse.jl")
# Tools for greedyfit and optimization
include("mapcomponent/qraccelerated.jl")
include("mapcomponent/precond.jl")
include("mapcomponent/greedyfit.jl")
include("mapcomponent/optimize.jl")

include("mapcomponent/hermitemap.jl")
include("mapcomponent/assimilate.jl")

include("tools/plot_recipes.jl")

end # module
