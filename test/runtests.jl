using Test

using LinearAlgebra, Statistics
using TransportMap
using AdaptiveTransportMap
using ForwardDiff
using FastGaussQuadrature
using SpecialFunctions

include("rectifier.jl")
include("phypolyhermite.jl")
# include("propolyhermite.jl")


include("scale.jl")