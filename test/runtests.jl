using Test

using LinearAlgebra, Statistics
using TransportMap
using AdaptiveTransportMap
using ForwardDiff
using FastGaussQuadrature
using SpecialFunctions
using QuadGK


# Tools: double factorial, adaptive integration
include("tools/tools.jl")
include("tools/clenshaw_curtis.jl")
include("tools/adaptiveCC.jl")


include("rectifier.jl")
include("phypolyhermite.jl")
include("propolyhermite.jl")

include("phyhermite.jl")
include("prohermite.jl")

include("basis.jl")


include("scale.jl")
