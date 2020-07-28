export Storage, update_storage

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

# function Storage(f::ParametricFunction{m, Nψ, Nx}, X::Array{Float64,2}; hess::Bool = false) where {m, Nψ, Nx}

function Storage(f::ParametricFunction{m, Nψ, Nx}, X; hess::Bool = false) where {m, Nψ, Nx}
        NxX, Ne = size(X)
        @assert NxX == Nx
        ψoff = evaluate_offdiagbasis(f, X)
        ψd   = evaluate_diagbasis(f, X)
        ψd0  = repeated_evaluate_basis(f.f, zeros(Ne))
        dψxd = repeated_grad_xk_basis(f.f, X[Nx,:])

        if hess == false
            # Cache variable
            cache_dcψxdt = zero(dψxd)
            cache_dψxd = zeros(Ne)
            cache_integral = zeros(Ne + Ne*Nψ)
            cache_g = zeros(Ne)
        else
            # Cache variable
            cache_dcψxdt = zero(dψxd)
            cache_dψxd = zeros(Ne)
            cache_integral = zeros(Ne + Ne*Nψ + Ne*Nψ*Nψ)
            cache_g = zeros(Ne)
        end


        return Storage{m, Nψ, Nx}(f, ψoff, ψd, ψd0, dψxd, cache_dcψxdt, cache_dψxd, cache_integral, cache_g)
end

# function update_storage(S::Storage{m, Nψ, k}, X::Array{Float64,2}, addedidx::Array{Int64,2}) where {m, Nψ, k}

function update_storage(S::Storage{m, Nψ, k}, X, addedidx::Array{Int64,2}) where {m, Nψ, k}
    NxX, Ne = size(X)

    @assert NxX == k "Wrong dimension of the sample X"
    addednψ = size(addedidx,1)
    newNψ = addednψ + Nψ

    fnew = ParametricFunction(ExpandedFunction(S.f.f.B, vcat(S.f.f.idx, addedidx), vcat(S.f.f.coeff, zeros(addednψ))))

    # Update off-diagonal component
    addedψoff = evaluate_offdiagbasis(fnew, X, addedidx)

    # Update ψd
    addedψd = evaluate_diagbasis(fnew, X, addedidx)

    # Update ψd0
    addedψd0  = repeated_evaluate_basis(fnew.f, zeros(Ne), addedidx)

    # Update dψxd
    addedψxd = repeated_grad_xk_basis(fnew.f, X[k,:], addedidx)


    return Storage{m, newNψ, k}(fnew, hcat(S.ψoff, addedψoff), hcat(S.ψd, addedψd),
                                hcat(S.ψd0, addedψd0), hcat(S.dψxd, addedψxd),
                                hcat(S.cache_dcψxdt, zeros(Ne, addednψ)),
                                S.cache_dψxd,
                                vcat(S.cache_integral, zeros(Ne*addednψ)),
                                S.cache_g)
    # @show S
end
