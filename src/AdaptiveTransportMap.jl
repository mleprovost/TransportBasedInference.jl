module AdaptiveTransportMap


using LinearAlgebra, SpecialFunctions
using Random
using ProgressMeter
using BenchmarkTools
using ForwardDiff
# using StaticUnivariatePolynomials
using Polynomials
using TransportMap
using DiffResults

include("tools.jl")

include("rectifier.jl")

# Hermite Polynomials
include("polyhermite.jl")
include("phypolyhermite.jl")
include("propolyhermite.jl")


# Hermite Functions
include("hermite.jl")
include("phyhermite.jl")
include("prohermite.jl")


# Uni and Multi Basis function
include("basis.jl")
include("multibasis.jl")
include("multifunction.jl")
include("expandedfunction.jl")

# Integrated positive function
include("integratedfunction.jl")


# Tools to apply a linear transformation
include("scale.jl")


end # module
