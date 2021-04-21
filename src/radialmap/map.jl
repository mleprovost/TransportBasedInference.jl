export KRmap, evaluate, SparseKRmap


import Base: size, show

#### Structure for the lower triangular map U

struct KRmap
        k::Int64
        p::Int64
        U::Array{Uk, 1}

        # Optimization parameters
        γ::Float64
        λ::Float64
        δ::Float64
        κ::Float64
end


function KRmap(k::Int64, p::Int64; γ::Float64=2.0, λ::Float64=0.1, δ::Float64=1e-8, κ::Float64=10.0)
        U = Uk[]
        for i=1:k
        @inbounds push!(U, Uk(i,p))
        end
        return KRmap(k, p, U, γ, λ, δ, κ)
end

function Base.show(io::IO, U::KRmap)
println(io,"K-R rearrangement of dimension k = $(U.k) and order p = $(U.p)
with parameters (γ, λ, δ, κ) = ($(U.γ), $(U.λ), $(U.λ), $(U.δ), $(U.κ))")

end


size(U::KRmap) = (U.k, U.p)

# Evaluate the map Uk at z = (z1,...,zk)
function (U::KRmap)(z; start::Int64=1)
        k = U.k
        @assert k==size(z)[1] "Incorrect length of the input vector"
        out = zeros(k-start+1)

        for i=start:k
        # @inbounds out[i] = U.U[i](z[1:i])
        @inbounds out[i] = U.U[i](view(z,1:i))
        end
        return out
end

function (U::KRmap)(z::EnsembleState; start::Int64=1)
        k = U.k
        @assert k==size(z)[1] "Incorrect length of the input vector"
        out = zeros(k-start+1)

        for i=start:k
        # @inbounds out[i] = U.U[i](z[1:i])
        @inbounds out[i] = U.U[i](view(z.S,1:i))
        end
        return out
end

function evaluate!(out, U::KRmap, z; start::Int64=1)
        k = U.k
        @assert k==size(z,1) "Incorrect length of the input vector"
        out = zeros(k-start+1)

        for i=start:k
        # @inbounds out[i] = U.U[i](z[1:i])
        @inbounds out[i] = U.U[i](view(z,1:i))
        end
        return out
end

# Evaluate the map Uk at z = (z1,...,zk)
function (U::KRmap)(X::AbstractMatrix{Float64}; start::Int64=1)
        k = U.k
        out = zeros(k-start+1,Ne)
        # out = SharedArray{Float64}(k-start+1,Ne)
                @inbounds for i=1:Ne
                col = view(X,:,i)
                out[:,i] .= U(col, start=start)
                end
        return out
end

# Evaluate the map Uk at z = (z1,...,zk)
function evaluate(U::KRmap, X::AbstractMatrix{Float64}; start::Int64=1)
        @get U (k, p)
        out = zeros(k-start+1,Ne)
        @inbounds Threads.@threads for i=1:Ne
                col = view(X,:,i)
                out[:,i] .= U(col, start=start)
        end
        return out
end


#### Sparse Structure for the lower triangular map U

struct SparseKRmap
        k::Int64
        p::Array{Array{Int64,1}}
        U::Array{SparseUk, 1}

        # Optimization parameters
        γ::Float64
        λ::Float64
        δ::Float64
        κ::Float64
end


function SparseKRmap(k::Int64, p::Array{Array{Int64,1}}; γ::Float64=2.0, λ::Float64=0.1, δ::Float64=1e-8, κ::Float64=10.0)
        @assert k==size(p,1) "Wrong dimension of the SparseKRmap"
        U = SparseUk[]
        for i=1:k
        @inbounds push!(U, SparseUk(i,p[i]))
        end
        return SparseKRmap(k, p, U, γ, λ, δ, κ)
end

function SparseKRmap(k::Int64, p::Array{Int64,1}; γ::Float64=2.0, λ::Float64=0.1, δ::Float64=1e-8, κ::Float64=10.0)
        @assert k==size(p,1) "Wrong dimension of the SparseKRmap"
        U = SparseUk[]
        for i=1:k
        @inbounds push!(U, SparseUk(i,p[i]))
        end
        return SparseKRmap(k, [fill(p[i],i) for i=1:k], U, γ, λ, δ, κ)
end

function Base.show(io::IO, U::SparseKRmap)
println(io,"Sparse K-R rearrangement of dimension k = $(U.k) and order p = $(U.p)
with parameters (γ, λ, δ, κ) = ($(U.γ), $(U.λ), $(U.λ), $(U.δ), $(U.κ))")

end


size(U::SparseKRmap) = (U.k, U.p)

# Evaluate the map Sparse Uk at z = (z1,...,zk)
function (U::SparseKRmap)(z; start::Int64=1)
        k = U.k
        @assert k==size(z,1) "Incorrect length of the input vector"
        out = zeros(k-start+1)
        @inbounds for i=start:k
        if allequal(U.U[i].p,-1)
        out[i] = z[i]
        else
         out[i] = U.U[i](view(z,1:i))
        end
        end
        return out
end

# Evaluate the map Sparse Uk at z = (z1,...,zk)
function (U::SparseKRmap)(X::AbstractMatrix{Float64}; start::Int64=1)
        k = U.k
        out = zeros(k-start+1,Ne)
        # out = SharedArray{Float64}(k-start+1,Ne)
                @inbounds for i=1:Ne
                col = view(X,:,i)
                out[:,i] .= U(col, start=start)
                end
        return out
end

# Evaluate the map SparseUk at z = (z1,...,zk)
function evaluate(U::SparseKRmap, X::AbstractMatrix{Float64}; start::Int64=1)
        @get U (k, p)
        out = zeros(k-start+1,Ne)
        @inbounds Threads.@threads for i=1:Ne
                        tmp_ens = view(ens.S,:,i)
                        out[:,i] .= U(tmp_ens, start=start)
        end
        return out
end
