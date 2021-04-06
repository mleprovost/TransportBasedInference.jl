using Test

using LinearAlgebra, Statistics
using OrdinaryDiffEq
using AdaptiveTransportMap
using AdaptiveTransportMap: evaluate
using ForwardDiff
using FastGaussQuadrature
using SpecialFunctions
using QuadGK
using Distributions
using Optim
using NLsolve
using MLDataUtils
using Quadrature
using Cubature
using FiniteDiff
using Random

# Tools: double factorial, adaptive integration
include("tools/tools.jl")
include("tools/normal.jl")
include("tools/transform.jl")

# Tools for data assimilation
include("DA/inflation.jl")


# Tools for state-space models
include("statespace/system.jl")


# Functions to manage margins
include("margin/reducedmargin.jl")
include("margin/totalorder.jl")

include("hermitefunction/phypolyhermite.jl")
include("hermitefunction/propolyhermite.jl")

include("hermitefunction/phyhermite.jl")
include("hermitefunction/prohermite.jl")

# Test tools for Basis, MultiBasis, ExpandedFunction
include("mapcomponent/rectifier.jl")
include("mapcomponent/basis.jl")
include("mapcomponent/expandedfunction.jl")
include("mapcomponent/reduced.jl")
include("mapcomponent/parametric.jl")

# Test tools for integrated function
include("mapcomponent/integratedfunction.jl")
include("mapcomponent/storage.jl")

# Test tools for HermiteMap component
include("mapcomponent/hermitemapcomponent.jl")
include("mapcomponent/linhermitemapcomponent.jl")

# Test greedy procedure
include("mapcomponent/greedyfit.jl")

# Test optimization of HermiteMap component
include("mapcomponent/qr.jl")
# include("mapcomponent/qraccelerated.jl")
include("mapcomponent/optimize.jl")

include("mapcomponent/inverse.jl")
include("mapcomponent/hybridinverse.jl")
include("mapcomponent/hermitemap.jl")
