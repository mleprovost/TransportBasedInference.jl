
export Mixture, rand, logpdf

import Base: rand
import Distributions: logpdf


struct Mixture{Nψ, Nx}
    # Array of Nψ dimensinos
    dist::Array{MvNormal,1}

    # Vector of weights for each mode
    w::Array{Float64,1}
end


function Mixture(Nx::Int64)

    @assert Nx == 3 "Method is not implemented for Nx != 3 "
    # Define number of modes
    Nψ = 2^Nx

    # Define weights, mean and covariance
    w   = rand(Nψ)
    w ./= sum(w)

    dist = []

    loc = [-4 -4.0 -4; -4 4 -4; 4 -4 -4; 4 4 -4; -4 -4 4; -4 4 4; 4 -4 4; 4 4 4]

    @assert size(loc) == (Nψ, Nx)
    collect(Iterators.product(-4:4, -4:4))

    @inbounds for i=1:Nψ
        # Use identity for the covariance matric
        push!(dist, MvNormal(loc[i,:], 1.0))
    end

    return Mixture{Nψ, Nx}(dist, w)
end


function rand(M::Mixture{Nψ, Nx}, Ne::Int64) where {Nψ, Nx}
    out = zeros(Nx, Ne)
    @inbounds for (i,di) in enumerate(M.dist)
        out += M.w[i]*rand(di,Ne)
    end
    return out
end


function logpdf(M::Mixture{Nψ, Nx}, X::Array{Float64,2}) where {Nψ, Nx}
    NxX, Ne = size(X)
    @assert NxX == Nx "Dimension of the sample is wrong"
    pdfX = zeros(Ne)
    @inbounds for (i,di) in enumerate(M.dist)
        pdfX += M.w[i]*pdf(di, X)
    end

    return log.(pdfX)
end
