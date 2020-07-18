export Storage, update_storage!

# Create a structure that will hold evaluation of the basis functions,
# as well as their derivative and second derivative


struct Storage{m, Nψ, Nx}

    # Parametric function
    f::ParametricFunction{m, Nψ, Nx}

    # Off-diagonal basis evaluation
    ψoff::Array{Float64,2}

    # Diagonal basis evaluation
    ψd::Array{Float64,2}

    # Diagonal basis evaluation at x = 0
    ψd0::Array{Float64,2}

    # Evaluate ∂_xd ψ(x_1,...,x_d)
    dψxd::Array{Float64,2}

    # Cache for ∂_c ∂_xd(f(x_{1:d-1},t)
    cache_dcψxdt::Array{Float64,2}

    # Cache for ∂_xd(f(x_{1:d-1},t)
    cache_dψxd::Array{Float64,1}

    # Cache integration for J and dJ
    cache_integral::Array{Float64,1}

    # Cache for g(∂k(f(x_{1:k})))
    cache_g::Array{Float64,1}
end

function Storage(f::ParametricFunction{m, Nψ, Nx}, X::Array{Float64,2}) where {m, Nψ, Nx}
        NxX, Ne = size(X)
        @assert NxX == Nx
        ψoff = evaluate_offdiagbasis(f, X)
        ψd   = evaluate_diagbasis(f, X)
        ψd0  = repeated_evaluate_basis(f.f, zeros(Ne))
        dψxd = repeated_grad_xk_basis(f.f, X[Nx,:])

        # Cache variable
        cache_dcψxdt = zero(dψxd)
        cache_dψxd = zeros(Ne)
        cache_integral = zeros(Ne + Ne*Nψ)
        cache_g = zeros(Ne)


        return Storage{m, Nψ, Nx}(f, ψoff, ψd, ψd0, dψxd, cache_dcψxdt, cache_dψxd, cache_integral, cache_g)
end


function update_storage!(S::Storage{m, Nψ, k}, X::Array{Float64,2}, newidx::Array{Int64,2}) where {m, Nψ, k}

end
