export StateSpace, propagate, observe


"""
$(TYPEDEF)

An immutable structure representing the dynamical `f` and observation `h` operators.
The dynamical model is provided by the right hand side of the ODE to solve.
For a system of ODEs, we will prefer an in-place syntax `f(du, u, p, t)`, where `p` are parameters of the model.
We rely on `OrdinaryDiffEq` to integrate the dynamical system with the Tsitouras 5/4 Runge-Kutta method adaptive time marching.
`h` must be a function of the form `h(u, t)`, where `u` is the state vector and `t` is the time.

## Fields
$(TYPEDFIELDS)

"""
struct StateSpace
    "Propagatation f"
    f::Function

    "Observation h"
    h::Function
end

# To remove
function propagate(f::Function, X, t::Float64, Ny::Int64, Nx::Int64; P::Parallel=serial)
    Nypx, Ne = size(X)
    @assert Nypx == Ny + Nx "Wrong dimension of Ny or Nx"
    if typeof(P)==Serial
        @inbounds for i=1:Ne
        x = view(X, Ny+1:Nypx, i)
        x .= f(x, t)[2]
        end
    else
        @inbounds Threads.@threads for i=1:Ne
        x = view(X, Ny+1:Nypx, i)
        x .= f(x, t)[2]
        end
    end
end

"""
$(TYPEDSIGNATURES)

Evaluate the function `h` for the different state vectors of the `X` at time `t`, and store the results in the first `Ny` columns of `X`.
`X` is an ensemble matrix that contains the observation vectors in the first `Ny` lines, and the state vectors in the lines `Ny+1` to `Ny+Nx`.
The code can run in serial or with multithreading.
"""
function observe(h::Function, X, t::Float64, Ny::Int64, Nx::Int64; P::Parallel=serial)
    Nypx, Ne = size(X)
    @assert Nypx == Ny + Nx "Wrong dimension of Ny or Nx"
    if typeof(P)==Serial
        @inbounds for i=1:Ne
        x = view(X, Ny+1:Nypx, i)
        y = view(X, 1:Ny, i)

        y .= h(x, t)
        end
    elseif typeof(P)==Thread
        @inbounds Threads.@threads for i=1:Ne
        x = view(X, Ny+1:Nypx, i)
        y = view(X, 1:Ny, i)

        y .= h(x, t)
        end
    end
end

"""
$(TYPEDSIGNATURES)

Evaluate the function `h` for the different state vectors of the `X` at time `t`, and store the results in observation vectors of `X`.
`X` is an EnsembleStateMeas that contains the states in the field `state` and the observations in the field `meas`.
The code can run in serial or with multithreading.
"""
function observe(h::Function, X::EnsembleStateMeas, t::Float64; P::Parallel=serial)
    Nx, Ny, Ne = size(X)
    if typeof(P)==Serial
        @inbounds for i=1:Ne
        #x = view(X, Ny+1:Nypx, i)
        x = view(X.state.S, :, i)
        #y = view(X, 1:Ny, i)
        y = view(X.meas.S, :, i)

        y .= h(x, t)
        end
    elseif typeof(P)==Thread
        @inbounds Threads.@threads for i=1:Ne
        #x = view(X, Ny+1:Nypx, i)
        x = view(X.state.S, :, i)
        #y = view(X, 1:Ny, i)
        y = view(X.meas.S, :, i)

        y .= h(x, t)
        end
    end
end

"""
$(TYPEDSIGNATURES)

Apply the observation operator of the `StateSpace` `F` to the ensemble matrix `X` at time `t`.
"""
observe(F::StateSpace, x::Array{Float64,1}, t::Float64) = F.h(x, t)
