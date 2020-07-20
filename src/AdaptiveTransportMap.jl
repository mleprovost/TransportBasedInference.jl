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

include("tools/product.jl")
include("tools/tools.jl")
include("tools/normal.jl")
include("tools/clenshaw_curtis.jl")
include("tools/adaptiveCC.jl")
# Tools to apply a linear transformation
include("tools/scale.jl")

include("rectifier.jl")

# Hermite Polynomials
include("phypolyhermite.jl")
include("propolyhermite.jl")

include("polyhermite.jl")

# Hermite Functions
include("hermite.jl")
include("phyhermite.jl")
include("prohermite.jl")


# Uni and Multi Basis function
include("basis.jl")
include("multibasis.jl")
include("multifunction.jl")
include("expandedfunction.jl")
include("parametric.jl")
include("storage.jl")

# Integrated positive function
include("integratedfunction.jl")

# ReducedMargin
include("margin/reducedmargin.jl")
include("margin/totalorder.jl")


# KR-rearrangement and TransportMap structure
include("hermitemapcomponent.jl")
include("hermitemap.jl")
include("greedyfit.jl")



end # module
