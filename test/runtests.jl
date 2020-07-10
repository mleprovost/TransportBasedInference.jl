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


# Tools: double factorial, adaptive integration
include("tools/tools.jl")
include("tools/normal.jl")
include("tools/clenshaw_curtis.jl")
include("tools/adaptiveCC.jl")
include("tools/scale.jl")

# Functions to manage margins
include("margin/reducedmargin.jl")



include("rectifier.jl")
include("phypolyhermite.jl")
include("propolyhermite.jl")

include("phyhermite.jl")
include("prohermite.jl")

include("basis.jl")
