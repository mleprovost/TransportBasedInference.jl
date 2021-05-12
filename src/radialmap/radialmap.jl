export RadialMap, evaluate!, evaluate, SparseRadialMap


import Base: size, show

#### Structure for the lower triangular map M

struct RadialMap
        Nx::Int64
        p::Int64
        L::LinearTransform
        C::Array{RadialMapComponent, 1}

        # Optimization parameters
        γ::Float64
        λ::Float64
        δ::Float64
        κ::Float64
end


function RadialMap(Nx::Int64, p::Int64; γ::Float64=2.0, λ::Float64=0.1, δ::Float64=1e-8, κ::Float64=10.0)
        C = RadialMapComponent[]
        @inbounds for i=1:Nx
                push!(C, RadialMapComponent(i,p))
        end
        return RadialMap(Nx, p, LinearTransform(Nx), C, γ, λ, δ, κ)
end

function RadialMap(X::Array{Float64,2}, p::Int64; γ::Float64=2.0, λ::Float64=0.1, δ::Float64=1e-8, κ::Float64=10.0)
        L = LinearTransform(X; diag = true)
        Nx, Ne = size(X)

        C = RadialMapComponent[]
        @inbounds for i=1:Nx
                push!(C, RadialMapComponent(i,p))
        end
        return RadialMap(Nx, p, LinearTransform(Nx), C, γ, λ, δ, κ)
end

function Base.show(io::IO, M::RadialMap)
println(io,"Radial Map of dimension Nx = $(M.Nx) and order p = $(M.p)
with parameters (γ, λ, δ, κ) = ($(M.γ), $(M.λ), $(M.δ), $(M.κ))")

end

@propagate_inbounds Base.getindex(M::RadialMap, i::Int) = getindex(M.C,i)
@propagate_inbounds Base.setindex!(M::RadialMap, C::RadialMapComponent, i::Int) = setindex!(M.C,C,i)

size(M::RadialMap) = (M.Nx, M.p)

# Evaluate the map RadialMapComponent at z = (z1,...,zNx)
function evaluate!(out, M::RadialMap, z::AbstractVector{Float64}; apply_rescaling::Bool=true, start::Int64=1)
        Nx = M.Nx
        @assert Nx == size(z,1) "Incorrect length of the input vector"

        # We can apply the rescaling to all the components once
        if apply_rescaling == true
                transform!(M.L, z)
        end

        @inbounds  for i=start:Nx
                out[i] = M.C[i](view(z,1:i))
        end

        if apply_rescaling == true
                itransform!(M.L, z)
        end

        return out
end

evaluate(M::RadialMap, z::AbstractVector{Float64}; apply_rescaling::Bool=true, start::Int64=1) = evaluate!(zeros(M.Nx-start+1), M, z; apply_rescaling = apply_rescaling, start = start)

(M::RadialMap)(z::AbstractVector{Float64}; apply_rescaling::Bool=true, start::Int64=1) =  evaluate(M, z; apply_rescaling = apply_rescaling, start = start)

# Evaluate in-place the RadialMap `M`
function evaluate!(out, M::RadialMap, X::AbstractMatrix{Float64}; apply_rescaling::Bool=true, start::Int64=1)
        @get M (Nx, p)
        NxX, Ne = size(X)
        @assert NxX == Nx "Wrong dimension of the ensemble matrix `X`"

        if apply_rescaling == true
                transform!(M.L, X)
        end

        @inbounds Threads.@threads for i=1:Ne
                col = view(X,:,i)
                evaluate!(view(out,:,i), M, col; apply_rescaling = false, start =start)
                # out[:,i] .= M(col, start=start)
        end

        if apply_rescaling == true
                itransform!(M.L, X)
        end
        return out
end

evaluate(M::RadialMap, X::AbstractMatrix{Float64}; apply_rescaling::Bool=true, start::Int64=1) = evaluate!(zero(X), M, X; apply_rescaling = apply_rescaling, start = start)

(M::RadialMap)(X::AbstractMatrix{Float64}; apply_rescaling::Bool=true, start::Int64=1) = evaluate(M, X; apply_rescaling = apply_rescaling, start = start)

#### Sparse Structure for the lower triangular map M

struct SparseRadialMap
        Nx::Int64
        p::Array{Array{Int64,1}}
        L::LinearTransform
        C::Array{SparseRadialMapComponent, 1}

        # Optimization parameters
        γ::Float64
        λ::Float64
        δ::Float64
        κ::Float64
end


function SparseRadialMap(Nx::Int64, p::Array{Array{Int64,1}}; γ::Float64=2.0, λ::Float64=0.1, δ::Float64=1e-8, κ::Float64=10.0)
        @assert Nx==size(p,1) "Wrong dimension of the SparseRadialMap"
        L = LinearTransform(Nx)
        C = SparseRadialMapComponent[]
        @inbounds  for i=1:Nx
                push!(C, SparseRadialMapComponent(i,p[i]))
        end
        return SparseRadialMap(Nx, p, L, C, γ, λ, δ, κ)
end

function SparseRadialMap(Nx::Int64, p::Array{Int64,1}; γ::Float64=2.0, λ::Float64=0.1, δ::Float64=1e-8, κ::Float64=10.0)
        @assert Nx==size(p,1) "Wrong dimension of the SparseRadialMap"
        L = LinearTransform(Nx)
        C = SparseRadialMapComponent[]
        @inbounds for i=1:Nx
                push!(C, SparseRadialMapComponent(i,p[i]))
        end
        return SparseRadialMap(Nx, [fill(p[i],i) for i=1:Nx], L, C, γ, λ, δ, κ)
end

function SparseRadialMap(Nx::Int64, p::Int64; γ::Float64=2.0, λ::Float64=0.1, δ::Float64=1e-8, κ::Float64=10.0)
        L = LinearTransform(Nx)
        C = SparseRadialMapComponent[]
        @inbounds for i=1:Nx
                push!(C, SparseRadialMapComponent(i,p))
        end
        return SparseRadialMap(Nx, [fill(p,i) for i=1:Nx], L, C, γ, λ, δ, κ)
end

# Methods based on an ensemble matrix X
function SparseRadialMap(X::Array{Float64,2}, p::Array{Array{Int64,1}}; γ::Float64=2.0, λ::Float64=0.1, δ::Float64=1e-8, κ::Float64=10.0)
        Nx, Ne = size(X)
        @assert Nx==size(p,1) "Wrong dimension of the SparseRadialMap"
        L = LinearTransform(X)
        C = SparseRadialMapComponent[]
        @inbounds  for i=1:Nx
                push!(C, SparseRadialMapComponent(i,p[i]))
        end
        return SparseRadialMap(Nx, p, L, C, γ, λ, δ, κ)
end

function SparseRadialMap(X::Array{Float64,2}, p::Array{Int64,1}; γ::Float64=2.0, λ::Float64=0.1, δ::Float64=1e-8, κ::Float64=10.0)
        Nx, Ne = size(X)
        @assert Nx==size(p,1) "Wrong dimension of the SparseRadialMap"
        L = LinearTransform(X)
        C = SparseRadialMapComponent[]
        @inbounds for i=1:Nx
                push!(C, SparseRadialMapComponent(i,p[i]))
        end
        return SparseRadialMap(Nx, [fill(p[i],i) for i=1:Nx], L, C, γ, λ, δ, κ)
end

function SparseRadialMap(X::Array{Float64,2}, p::Int64; γ::Float64=2.0, λ::Float64=0.1, δ::Float64=1e-8, κ::Float64=10.0)
        Nx, Ne = size(X)
        L = LinearTransform(X)
        C = SparseRadialMapComponent[]
        @inbounds for i=1:Nx
                push!(C, SparseRadialMapComponent(i,p))
        end
        return SparseRadialMap(Nx, [fill(p,i) for i=1:Nx], L, C, γ, λ, δ, κ)
end

function Base.show(io::IO, M::SparseRadialMap)
        println(io,"Sparse Radial Map of dimension Nx = $(M.Nx) and order p = $(M.p)
        with parameters (γ, λ, δ, κ) = ($(M.γ), $(M.λ), $(M.δ), $(M.κ))")
end

@propagate_inbounds Base.getindex(M::SparseRadialMap, i::Int) = getindex(M.C,i)
@propagate_inbounds Base.setindex!(M::SparseRadialMap, C::SparseRadialMapComponent, i::Int) = setindex!(M.C,C,i)


size(M::SparseRadialMap) = (M.Nx, M.p)

# Evaluate the map Sparse RadialMap at z = (z1,...,zNx)

function evaluate!(out, M::SparseRadialMap, z::AbstractVector{Float64}; apply_rescaling::Bool=true, start::Int64=1)
        Nx = M.Nx
        @assert Nx==size(z,1) "Incorrect length of the input vector"

        if apply_rescaling == true
                transform!(M.L, z)
        end

        @inbounds for i=start:Nx
                if allequal(M.C[i].p,-1)
                        out[i] = z[i]
                else
                        out[i] = M.C[i](view(z,1:i))
                end
        end

        if apply_rescaling == true
                itransform!(M.L, z)
        end

        return out
end

evaluate(M::SparseRadialMap, z::AbstractVector{Float64}; apply_rescaling::Bool=true, start::Int64=1) = evaluate!(zeros(M.Nx-start+1), M, z; apply_rescaling = apply_rescaling, start = start)

(M::SparseRadialMap)(z::AbstractVector{Float64}; apply_rescaling::Bool=true, start::Int64=1) =  evaluate(M, z; apply_rescaling = apply_rescaling, start = start)

# Evaluate the map SparseRadialMapComponent at z = (z1,...,zNx)
function evaluate!(out, M::SparseRadialMap, X::AbstractMatrix{Float64}; apply_rescaling::Bool=true, start::Int64=1)
        @get M (Nx, p)
        NxX, Ne = size(X)
        @assert NxX == Nx "Wrong dimension of the ensemble matrix `X`"
        @assert size(out) == (Nx, Ne) "Wrong dimension of the output matrix `out`"

        if apply_rescaling == true
                transform!(M.L, X)
        end

        @inbounds for i=1:Ne
                col = view(X,:,i)
                evaluate!(view(out,:,i), M, col; apply_rescaling = false, start = start)
        end

        if apply_rescaling == true
                itransform!(M.L, X)
        end

        return out
end

evaluate(M::SparseRadialMap, X::AbstractMatrix{Float64}; apply_rescaling::Bool=true, start::Int64=1) = evaluate!(zero(X), M, X; apply_rescaling = apply_rescaling, start = start)

(M::SparseRadialMap)(X::AbstractMatrix{Float64}; apply_rescaling::Bool=true, start::Int64=1) =  evaluate(M, X; apply_rescaling = apply_rescaling, start = start)
