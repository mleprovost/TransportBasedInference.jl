using Test

using LinearAlgebra, Statistics
using TransportMap
using AdaptiveTransportMap
using AdaptiveTransportMap: evaluate
using ForwardDiff
using FastGaussQuadrature
using SpecialFunctions
using QuadGK
using Random
using Distributions
using Optim
using NLsolve
using MLDataUtils

# # Tools: double factorial, adaptive integration
# include("tools/tools.jl")
# include("tools/normal.jl")
# include("tools/clenshaw_curtis.jl")
# include("tools/adaptiveCC.jl")
# include("tools/transform.jl")
#
# # Functions to manage margins
# include("margin/reducedmargin.jl")
# include("margin/totalorder.jl")
#
# include("hermitefunction/phypolyhermite.jl")
# include("hermitefunction/propolyhermite.jl")
#
# include("hermitefunction/phyhermite.jl")
# include("hermitefunction/prohermite.jl")
# #
# # # Test tools for Basis, MultiBasis, ExpandedFunction
# include("mapcomponent/rectifier.jl")
# include("mapcomponent/basis.jl")
# include("mapcomponent/expandedfunction.jl")
# include("mapcomponent/parametric.jl")
#
# # Test tools for integrated function
# include("mapcomponent/integratedfunction.jl")
# include("mapcomponent/storage.jl")

# Test tools for HermiteMap component
include("mapcomponent/hermitemapcomponent.jl")
include("mapcomponent/linhermitemapcomponent.jl")

# Test greedy procedure
include("mapcomponent/greedyfit.jl")

# Test optimization of HermiteMap component
include("mapcomponent/qraccelerated.jl")
include("mapcomponent/optimize.jl")

include("mapcomponent/inverse.jl")

include("mapcomponent/hermitemap.jl")
