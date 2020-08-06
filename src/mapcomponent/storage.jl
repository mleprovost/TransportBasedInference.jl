
export Storage, update_storage

# Create a structure that will hold evaluation of the basis functions,
# as well as their derivative and second derivative


struct Storage

    m::Int64
    Nψ::Int64
    Nx::Int64

    # Parametric function
    f::ParametricFunction

    # Off-diagonal basis evaluation
    ψoff::Array{Float64,2}

    # Off-diagonal basis evaluation ⊗ Diagonal basis evaluation at x = 0
    ψoffψd::Array{Float64,2}

    # Off-diagonal basis evaluation ⊗ Diagonal basis evaluation at x = 0
    ψoffψd0::Array{Float64,2}

    # Off-diagonal basis evaluation ⊗ ∂_xd ψ(x_1,...,x_d)
    ψoffdψxd::Array{Float64,2}

    # Store the norm of each column of ψoffψd
    ψnorm::Array{Float64,1}

    # Cache for ∂_c ∂_xd(f(x_{1:d-1},t)
    cache_dcψxdt::Array{Float64,2}

    cache_gradxd::Array{Float64,2}

    # Cache for ∂_xd(f(x_{1:d-1},t)
    cache_dψxd::Array{Float64,1}

    # Cache integration for J and dJ
    cache_integral::Array{Float64,1}

    # Cache for g(∂k(f(x_{1:k})))
    cache_g::Array{Float64,1}
end

# function Storage(f::ParametricFunction{m, Nψ, Nx}, X::Array{Float64,2}; hess::Bool = false) where {m, Nψ, Nx}

function Storage(f::ParametricFunction, X)#, hess::Bool = false)
        m = f.f.m
        Nψ = f.f.Nψ
        Nx = f.f.Nx
        NxX, Ne = size(X)
        @assert NxX == Nx
        ψoff = evaluate_offdiagbasis(f, X)
        ψoffψd = evaluate_diagbasis(f, X)
        ψoffψd0  = repeated_evaluate_basis(f.f, zeros(Ne))
        ψoffdψxd = repeated_grad_xk_basis(f.f, X[Nx,:])

        @avx for j=1:Nψ
            for i=1:Ne
                ψoffij = ψoff[i,j]
                ψoffψd[i,j] *= ψoffij
                ψoffψd0[i,j] *= ψoffij
                ψoffdψxd[i,j] *= ψoffij
            end
        end

        ψnorm = zeros(Nψ)
        @inbounds for i=1:Nψ
            ψnorm[i] = norm(view(ψoffψd,:,i))
        end

        rmul!(ψnorm, 1/sqrt(Ne))

        # Cache variable
        cache_dcψxdt = zero(ψoff)
        cache_gradxd = zeros(Ne, maximum(f.f.idx[:,end])+1)
        cache_dψxd = zeros(Ne)
        cache_integral = zeros(Ne + Ne*Nψ)
        cache_g = zeros(Ne)


        return Storage(m, Nψ, Nx, f, ψoff, ψoffψd, ψoffψd0, ψoffdψxd, ψnorm, cache_dcψxdt, cache_gradxd, cache_dψxd, cache_integral, cache_g)
end

# function update_storage(S::Storage{m, Nψ, k}, X::Array{Float64,2}, addedidx::Array{Int64,2}) where {m, Nψ, k}

function update_storage(S::Storage, X, addedidx::Array{Int64,2})
    NxX, Ne = size(X)
    Nψ = S.Nψ

    @assert NxX == S.Nx "Wrong dimension of the sample X"
    addedNψ = size(addedidx,1)
    newNψ = addedNψ + Nψ

    fnew = ParametricFunction(ExpandedFunction(S.f.f.B, vcat(S.f.f.idx, addedidx), vcat(S.f.f.coeff, zeros(addedNψ))))

    oldmaxj = maximum(S.f.f.idx[:,end])
    newmaxj = maximum(fnew.f.idx[:,end])

    @assert newmaxj >= oldmaxj "Error in the adaptive procedure, the set is not downward closed"


    # Update off-diagonal component
    addedψoff = evaluate_offdiagbasis(fnew, X, addedidx)

    # Update ψd
    addedψoffψd = evaluate_diagbasis(fnew, X, addedidx)

    # Update ψd0
    addedψoffψd0  = repeated_evaluate_basis(fnew.f, zeros(Ne), addedidx)

    # Update dψxd
    addedψoffdψxd = repeated_grad_xk_basis(fnew.f, X[S.Nx,:], addedidx)


    @avx for j=1:addedNψ
        for i=1:Ne
            addedψoffij = addedψoff[i,j]
            addedψoffψd[i,j] *= addedψoffij
            addedψoffψd0[i,j] *= addedψoffij
            addedψoffdψxd[i,j] *= addedψoffij
        end
    end

    addedψnorm = zeros(addedNψ)
     for i=1:addedNψ
        addedψnorm[i] = norm(view(addedψoffψd,:,i))
    end

    rmul!(addedψnorm, 1/sqrt(Ne))

    return Storage(S.m, newNψ, S.Nx, fnew, hcat(S.ψoff, addedψoff),
                                           hcat(S.ψoffψd, addedψoffψd),
                                           hcat(S.ψoffψd0, addedψoffψd0),
                                           hcat(S.ψoffdψxd, addedψoffdψxd),
                                           vcat(S.ψnorm, addedψnorm),
                                           hcat(S.cache_dcψxdt, zeros(Ne, addedNψ)),
                                           hcat(S.cache_gradxd, zeros(Ne,newmaxj-oldmaxj)),
                                           S.cache_dψxd,
                                           vcat(S.cache_integral, zeros(Ne*addedNψ)),
                                           S.cache_g)
end
