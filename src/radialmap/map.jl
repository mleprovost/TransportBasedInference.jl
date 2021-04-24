export RadialMap, evaluate!, evaluate, SparseRadialMap


import Base: size, show

#### Structure for the lower triangular map M

struct RadialMap
        Nx::Int64
        p::Int64
        C::Array{RadialMapComponent, 1}

        # Optimization parameters
        γ::Float64
        λ::Float64
        δ::Float64
        κ::Float64
end


function RadialMap(Nx::Int64, p::Int64; γ::Float64=2.0, λ::Float64=0.1, δ::Float64=1e-8, κ::Float64=10.0)
        C = RadialMapComponent[]
        for i=1:Nx
        @inbounds push!(C, RadialMapComponent(i,p))
        end
        return RadialMap(Nx, p, C, γ, λ, δ, κ)
end

function Base.show(io::IO, M::RadialMap)
println(io,"Radial Map of dimension Nx = $(M.Nx) and order p = $(M.p)
with parameters (γ, λ, δ, κ) = ($(M.γ), $(M.λ), $(M.δ), $(M.κ))")

end


size(M::RadialMap) = (M.Nx, M.p)

# Evaluate the map RadialMapComponent at z = (z1,...,zNx)
function evaluate!(out, M::RadialMap, z::AbstractVector{Float64}; start::Int64=1)
        Nx = M.Nx
        @assert Nx == size(z,1) "Incorrect length of the input vector"

        for i=start:Nx
        # @inbounds out[i] = M.C[i](z[1:i])
        @inbounds out[i] = M.C[i](view(z,1:i))
        end
        return out
end

evaluate(M::RadialMap, z::AbstractVector{Float64}; start::Int64=1) = evaluate!(zeros(M.Nx-start+1), M, z; start = start)

(M::RadialMap)(z::AbstractVector{Float64}; start::Int64=1) =  evaluate(M, z; start = start)

# Evaluate in-place the RadialMap `M`
function evaluate!(out, M::RadialMap, X::AbstractMatrix{Float64}; start::Int64=1)
        @get M (Nx, p)
        NxX, Ne = size(X)
        @assert NxX == Nx "Wrong dimension of the ensemble matrix `X`"
        @inbounds Threads.@threads for i=1:Ne
                col = view(X,:,i)
                evaluate!(view(out,:,i), M, col; start =start)
                # out[:,i] .= M(col, start=start)
        end
        return out
end

evaluate(M::RadialMap, X::AbstractMatrix{Float64}; start::Int64=1) = evaluate!(zero(size(X)), M, X; start = start)

(M::RadialMap)(X::AbstractMatrix{Float64}; start::Int64=1) = evaluate(M::RadialMap, X::AbstractMatrix{Float64}; start = start)

#### Sparse Structure for the lower triangular map M

struct SparseRadialMap
        Nx::Int64
        p::Array{Array{Int64,1}}
        C::Array{SparseRadialMapComponent, 1}

        # Optimization parameters
        γ::Float64
        λ::Float64
        δ::Float64
        κ::Float64
end


function SparseRadialMap(Nx::Int64, p::Array{Array{Int64,1}}; γ::Float64=2.0, λ::Float64=0.1, δ::Float64=1e-8, κ::Float64=10.0)
        @assert Nx==size(p,1) "Wrong dimension of the SparseRadialMap"
        C = SparseRadialMapComponent[]
        for i=1:Nx
        @inbounds push!(C, SparseRadialMapComponent(i,p[i]))
        end
        return SparseRadialMap(Nx, p, C, γ, λ, δ, κ)
end

function SparseRadialMap(Nx::Int64, p::Array{Int64,1}; γ::Float64=2.0, λ::Float64=0.1, δ::Float64=1e-8, κ::Float64=10.0)
        @assert Nx==size(p,1) "Wrong dimension of the SparseRadialMap"
        C = SparseRadialMapComponent[]
        for i=1:Nx
        @inbounds push!(C, SparseRadialMapComponent(i,p[i]))
        end
        return SparseRadialMap(Nx, [fill(p[i],i) for i=1:Nx], C, γ, λ, δ, κ)
end

function Base.show(io::IO, M::SparseRadialMap)
        println(io,"Sparse Radial Map of dimension Nx = $(M.Nx) and order p = $(M.p)
        with parameters (γ, λ, δ, κ) = ($(M.γ), $(M.λ), $(M.δ), $(M.κ))")
end

@propagate_inbounds Base.getindex(M::SparseRadialMap, i::Int) = getindex(M.C,i)
@propagate_inbounds Base.setindex!(M::SparseRadialMap, C::SparseRadialMapComponent, i::Int) = setindex!(M.C,C,i)


size(M::SparseRadialMap) = (M.Nx, M.p)

# Evaluate the map Sparse RadialMap at z = (z1,...,zNx)

function evaluate!(out, M::SparseRadialMap, z::AbstractVector{Float64}; start::Int64=1)
        Nx = M.Nx
        @assert Nx==size(z,1) "Incorrect length of the input vector"
        @inbounds for i=start:Nx
                if allequal(M.C[i].p,-1)
                out[i] = z[i]
                else
                out[i] = M.C[i](view(z,1:i))
                end
        end
        return out
end

evaluate(M::SparseRadialMap, z::AbstractVector{Float64}; start::Int64=1) = evaluate!(zeros(M.Nx-start+1), M, z; start = start)

(M::SparseRadialMap)(z::AbstractVector{Float64}; start::Int64=1) =  evaluate(M, z; start = start)

# Evaluate the map SparseRadialMapComponent at z = (z1,...,zNx)
function evaluate!(out, M::SparseRadialMap, X::AbstractMatrix{Float64}; start::Int64=1)
        @get M (Nx, p)
        NxX, Ne = size(X)
        @assert NxX == Nx "Wrong dimension of the ensemble matrix `X`"
        @assert size(out) == (Nx, Ne) "Wrong dimension of the output matrix `out`"
        @inbounds for i=1:Ne
                col = view(X,:,i)
                evaluate!(view(out,:,i), M, col; start = start)
        end
        return out
end

evaluate(M::SparseRadialMap, X::AbstractMatrix{Float64}; start::Int64=1) = evaluate!(zero(X), M, X; start = start)

(M::SparseRadialMap)(X::AbstractMatrix{Float64}; start::Int64=1) =  evaluate(M, X; start = start)
