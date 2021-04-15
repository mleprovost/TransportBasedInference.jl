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

Test tools for Basis, MultiBasis, ExpandedFunction
include("hermitemap/rectifier.jl")
include("hermitemap/basis.jl")
include("hermitemap/expandedfunction.jl")
include("hermitemap/reduced.jl")
include("hermitemap/parametric.jl")

# Test tools for integrated function
include("hermitemap/integratedfunction.jl")
include("hermitemap/storage.jl")

# Test tools for HermiteMap component
include("hermitemap/hermitemapcomponent.jl")
include("hermitemap/linhermitemapcomponent.jl")

# Test greedy procedure
include("hermitemap/greedyfit.jl")

# Test optimization of HermiteMap component
include("hermitemap/qr.jl")
# include("hermitemap/qraccelerated.jl")
include("hermitemap/optimize.jl")

include("hermitemap/inverse.jl")
include("hermitemap/hybridinverse.jl")
include("hermitemap/hermitemap.jl")
