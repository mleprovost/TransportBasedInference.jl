export RadialMap, evaluate, SparseRadialMap


import Base: size, show

#### Structure for the lower triangular map U

struct RadialMap
        Nx::Int64
        p::Int64
        U::Array{RadialMapComponent, 1}

        # Optimization parameters
        γ::Float64
        λ::Float64
        δ::Float64
        κ::Float64
end


function RadialMap(Nx::Int64, p::Int64; γ::Float64=2.0, λ::Float64=0.1, δ::Float64=1e-8, κ::Float64=10.0)
        U = RadialMapComponent[]
        for i=1:Nx
        @inbounds push!(U, RadialMapComponent(i,p))
        end
        return RadialMap(Nx, p, U, γ, λ, δ, κ)
end

function Base.show(io::IO, U::RadialMap)
println(io,"Radial Map of dimension Nx = $(U.Nx) and order p = $(U.p)
with parameters (γ, λ, δ, κ) = ($(U.γ), $(U.λ), $(U.λ), $(U.δ), $(U.κ))")

end


size(U::RadialMap) = (U.Nx, U.p)

# Evaluate the map RadialMapComponent at z = (z1,...,zNx)
function (U::RadialMap)(z; start::Int64=1)
        Nx = U.Nx
        @assert Nx==size(z,1) "Incorrect length of the input vector"
        out = zeros(Nx-start+1)

        for i=start:Nx
        # @inbounds out[i] = U.U[i](z[1:i])
        @inbounds out[i] = U.U[i](view(z,1:i))
        end
        return out
end

function evaluate!(out, U::RadialMap, z; start::Int64=1)
        Nx = U.Nx
        @assert Nx==size(z,1) "Incorrect length of the input vector"
        out = zeros(Nx-start+1)

        for i=start:Nx
        # @inbounds out[i] = U.U[i](z[1:i])
        @inbounds out[i] = U.U[i](view(z,1:i))
        end
        return out
end

# Evaluate the map RadialMapComponent at z = (z1,...,zNx)
function (U::RadialMap)(X::AbstractMatrix{Float64}; start::Int64=1)
        Nx = U.Nx
        out = zeros(Nx-start+1,Ne)
        # out = SharedArray{Float64}(Nx-start+1,Ne)
                @inbounds for i=1:Ne
                col = view(X,:,i)
                out[:,i] .= U(col, start=start)
                end
        return out
end

# Evaluate the map RadialMapComponent at z = (z1,...,zNx)
function evaluate(U::RadialMap, X::AbstractMatrix{Float64}; start::Int64=1)
        @get U (Nx, p)
        out = zeros(Nx-start+1,Ne)
        @inbounds Threads.@threads for i=1:Ne
                col = view(X,:,i)
                out[:,i] .= U(col, start=start)
        end
        return out
end


#### Sparse Structure for the lower triangular map U

struct SparseRadialMap
        Nx::Int64
        p::Array{Array{Int64,1}}
        U::Array{SparseRadialMapComponent, 1}

        # Optimization parameters
        γ::Float64
        λ::Float64
        δ::Float64
        κ::Float64
end


function SparseRadialMap(Nx::Int64, p::Array{Array{Int64,1}}; γ::Float64=2.0, λ::Float64=0.1, δ::Float64=1e-8, κ::Float64=10.0)
        @assert Nx==size(p,1) "Wrong dimension of the SparseRadialMap"
        U = SparseRadialMapComponent[]
        for i=1:Nx
        @inbounds push!(U, SparseRadialMapComponent(i,p[i]))
        end
        return SparseRadialMap(Nx, p, U, γ, λ, δ, κ)
end

function SparseRadialMap(Nx::Int64, p::Array{Int64,1}; γ::Float64=2.0, λ::Float64=0.1, δ::Float64=1e-8, κ::Float64=10.0)
        @assert Nx==size(p,1) "Wrong dimension of the SparseRadialMap"
        U = SparseRadialMapComponent[]
        for i=1:Nx
        @inbounds push!(U, SparseRadialMapComponent(i,p[i]))
        end
        return SparseRadialMap(Nx, [fill(p[i],i) for i=1:Nx], U, γ, λ, δ, κ)
end

function Base.show(io::IO, U::SparseRadialMap)
println(io,"Sparse Radial Map of dimension Nx = $(U.Nx) and order p = $(U.p)
with parameters (γ, λ, δ, κ) = ($(U.γ), $(U.λ), $(U.λ), $(U.δ), $(U.κ))")

end


size(U::SparseRadialMap) = (U.Nx, U.p)

# Evaluate the map Sparse RadialMapComponent at z = (z1,...,zNx)
function (U::SparseRadialMap)(z; start::Int64=1)
        Nx = U.Nx
        @assert Nx==size(z,1) "Incorrect length of the input vector"
        out = zeros(Nx-start+1)
        @inbounds for i=start:Nx
        if allequal(U.U[i].p,-1)
        out[i] = z[i]
        else
         out[i] = U.U[i](view(z,1:i))
        end
        end
        return out
end

# Evaluate the map Sparse RadialMapComponent at z = (z1,...,zNx)
function (U::SparseRadialMap)(X::AbstractMatrix{Float64}; start::Int64=1)
        Nx = U.Nx
        out = zeros(Nx-start+1,Ne)
        # out = SharedArray{Float64}(Nx-start+1,Ne)
                @inbounds for i=1:Ne
                col = view(X,:,i)
                out[:,i] .= U(col, start=start)
                end
        return out
end

# Evaluate the map SparseRadialMapComponent at z = (z1,...,zNx)
function evaluate(U::SparseRadialMap, X::AbstractMatrix{Float64}; start::Int64=1)
        @get U (Nx, p)
        out = zeros(Nx-start+1,Ne)
        @inbounds Threads.@threads for i=1:Ne
                        tmp_ens = view(ens.S,:,i)
                        out[:,i] .= U(tmp_ens, start=start)
        end
        return out
end
