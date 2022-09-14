
export Mixture, sample_mixture, log_pdf_mixture


struct Mixture
    "Number of Gaussian kernels"
    Nψ::Int64

    "Dimension of the state"
    Nx::Int64

    "Array of Nψ dimensions"
    dist::Array{MvNormal,1}

    "Vector of weights for each mode"
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

    return Mixture(Nψ, Nx, dist, w)
end


function sample_mixture(M::Mixture, Ne::Int64)
    out = zeros(M.Nx, Ne)
    @inbounds for (i,di) in enumerate(M.dist)
        out += M.w[i]*rand(di,Ne)
    end
    return out
end


function log_pdf_mixture(M::Mixture, X::Array{Float64,2})
    NxX, Ne = size(X)
    @assert NxX == M.Nx "Dimension of the sample is wrong"
    logpdfX = zeros(Ne)
    @inbounds for (i,di) in enumerate(M.dist)
        logpdfX += M.w[i]*pdf(di, X)
    end
    @. logpdfX = log(logpdfX)

    return logpdfX
end
