using Test

using LinearAlgebra, Statistics
using TransportMap
using AdaptiveTransportMap
using ForwardDiff
using FastGaussQuadrature
using SpecialFunctions
using QuadGK
using Random
using Distributions
using Optim


# Tools: double factorial, adaptive integration
include("tools/tools.jl")
include("tools/normal.jl")
include("tools/clenshaw_curtis.jl")
include("tools/adaptiveCC.jl")
include("tools/transform.jl")

# Functions to manage margins
include("margin/reducedmargin.jl")
include("margin/totalorder.jl")


include("rectifier.jl")
include("phypolyhermite.jl")
include("propolyhermite.jl")

include("phyhermite.jl")
include("prohermite.jl")


# Test tools for Basis, MultiBasis, ExpandedFunction
include("basis.jl")
include("expandedfunction.jl")
include("parametric.jl")

# Test tools for integrated function
include("integratedfunction.jl")

include("storage.jl")


# Test tools for HermiteMap component
include("hermitemapcomponent.jl")

# Test greedy procedure
include("greedyfit.jl")
