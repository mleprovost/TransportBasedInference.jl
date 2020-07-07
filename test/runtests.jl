using Test

using LinearAlgebra, Statistics
using TransportMap
using AdaptiveTransportMap
using ForwardDiff
using FastGaussQuadrature
using SpecialFunctions


include("tools.jl")
include("rectifier.jl")
include("phypolyhermite.jl")
include("propolyhermite.jl")

include("phyhermite.jl")
include("prohermite.jl")


include("scale.jl")
