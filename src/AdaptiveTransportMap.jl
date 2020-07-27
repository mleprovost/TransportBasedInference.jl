module AdaptiveTransportMap

using LinearAlgebra, SpecialFunctions
using Random
using ProgressMeter
using BenchmarkTools
using ForwardDiff
# using StaticUnivariatePolynomials
using Polynomials
using TransportMap
using DiffResults
using Distributions
using QuadGK
using TensorOperations
using LoopVectorization
using FastGaussQuadrature
using Optim
# using LinearMaps
using NLsolve
# using SparseArrays
using MLDataUtils
using MLBase

include("tools/product.jl")
include("tools/tools.jl")
include("tools/normal.jl")
include("tools/clenshaw_curtis.jl")
include("tools/adaptiveCC.jl")
# Tools to apply a linear transformation
include("tools/transform.jl")
# Tools for mixture of Gaussian distributions
include("tools/mixture.jl")

include("mapcomponent/rectifier.jl")



# Hermite Polynomials
include("hermitefunction/phypolyhermite.jl")
include("hermitefunction/propolyhermite.jl")
include("hermitefunction/polyhermite.jl")

# Hermite Functions
include("hermitefunction/hermite.jl")
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

# Tools for fast inversion
include("mapcomponent/inverse.jl")

# ReducedMargin
include("margin/reducedmargin.jl")
include("margin/totalorder.jl")


# KR-rearrangement and TransportMap structure
include("mapcomponent/precond.jl")
include("mapcomponent/hermitemapcomponent.jl")
include("mapcomponent/linhermitemapcomponent.jl")
include("mapcomponent/hermitemap.jl")
include("mapcomponent/greedyfit.jl")
include("mapcomponent/optimize.jl")



end # module
