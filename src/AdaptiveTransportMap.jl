module AdaptiveTransportMap


using LinearAlgebra, SpecialFunctions
using Random
using ProgressMeter
using BenchmarkTools
using ForwardDiff
# using StaticUnivariatePolynomials
using Polynomials
using TransportMap



include("rectifier.jl")

# Hermite Polynomials
include("hermite.jl")
include("phyhermite.jl")
# include("prohermite.jl")

# Tools to apply a linear transformation
include("scale.jl")


end # module
