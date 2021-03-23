export StateSpace, propagate, observe

struct StateSpace
    "Propagatation f"
    f::Function

    "Observation h"
    h::Function
end

function propagate(f::Function, t, X, Ny::Int64, Nx::Int64; P::Parallel=serial)
    Nypx, Ne = size(X)
    @assert Nypx == Ny + Nx "Wrong dimension of Ny or Nx"
    if typeof(P)==Serial
        @inbounds for i=1:Ne
        x = view(X, Ny+1:Nypx, i)
        x .= f(t, x)[2]
        end
    else
        @inbounds Threads.@threads for i=1:Ne
        x = view(X, Ny+1:Nypx, i)
        x .= f(t, x)[2]
        end
    end
end


function observe(h::Function, t, X, Ny::Int64, Nx::Int64; P::Parallel=serial)
    Nypx, Ne = size(X)
    @assert Nypx == Ny + Nx "Wrong dimension of Ny or Nx"
    if typeof(P)==Serial
        @inbounds for i=1:Ne
        x = view(X, Ny+1:Nypx, i)
        y = view(X, 1:Ny, i)

        y .= h(t, x)
        end
    else
        @inbounds Threads.@threads for i=1:Ne
        x = view(X, Ny+1:Nypx, i)
        y = view(X, 1:Ny, i)

        y .= h(t, x)
        end
    end
end

observe(F::StateSpace, t, x::Array{Float64,1}) = F.h(t, x)
