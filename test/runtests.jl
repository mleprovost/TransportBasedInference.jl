using Test

using LinearAlgebra, Statistics
using TransportMap
using AdaptiveTransportMap
using ForwardDiff
using FastGaussQuadrature

include("rectifier.jl")
include("phyhermite.jl")
include("prohermite.jl")


include("scale.jl")
