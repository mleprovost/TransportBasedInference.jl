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

    # Diagonal basis evaluation
    ψd::Array{Float64,2}

    # Diagonal basis evaluation at x = 0
    ψd0::Array{Float64,2}

    # Evaluate ∂_xd ψ(x_1,...,x_d)
    dψxd::Array{Float64,2}

    # Norm of ψd ⊗ ψoff per feature
    ψnorm::Array{Float64,1}

    R::UpperTriangular{Float64,Array{Float64,2}}

    Rinv::UpperTriangular{Float64,Array{Float64,2}}

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

function Storage(f::ParametricFunction, X; isrescaled::Bool=true)
        m = f.f.m
        Nψ = f.f.Nψ
        Nx = f.f.Nx
        NxX, Ne = size(X)
        @assert NxX == Nx
        ψoff = evaluate_offdiagbasis(f, X)
        ψd   = evaluate_diagbasis(f, X)
        ψd0  = repeated_evaluate_basis(f.f, zeros(Ne))
        dψxd = repeated_grad_xk_basis(f.f, X[Nx,:])

        # Cache variable
        cache_dcψxdt = zero(dψxd)
        cache_gradxd = zeros(Ne, maximum(f.f.idx[:,end])+1)
        cache_dψxd = zeros(Ne)
        cache_integral = zeros(Ne + Ne*Nψ)
        cache_g = zeros(Ne)

        # R and R^{-1} matrices
        # This avoid to allocate a new array
        @. cache_dcψxdt = ψoff * ψd
        ψnorm = norm.(eachcol(cache_dcψxdt))

        @inbounds for col in eachcol(cache_dcψxdt)
            normalize!(col)
        end

        # F = UpdatableQR(cache_dcψxdt)
        # Rinv = inv(F.R1)
        R = UpperTriangular(zeros(5,5))
        Rinv = UpperTriangular(zeros(5,5))

        if isrescaled == true
            ψoff ./= ψnorm'
            ψd   ./= ψnorm'
            ψd0  ./= ψnorm'
            dψxd ./= ψnorm'
        end

        fill!(cache_dcψxdt, 0.0)

        # else (if we add the hessian)
            # Cache variable
            # cache_dcψxdt = zero(dψxd)
            # cache_gradxd = zeros(Ne, maximum(f.f.idx[:,end])+1)
            # cache_dψxd = zeros(Ne)
            # cache_integral = zeros(Ne + Ne*Nψ + Ne*Nψ*Nψ)
            # cache_g = zeros(Ne)
        # end
        return Storage(m, Nψ, Nx, f, ψoff, ψd, ψd0, dψxd, ψnorm, R, Rinv, cache_dcψxdt, cache_gradxd, cache_dψxd, cache_integral, cache_g)
end

# function update_storage(S::Storage{m, Nψ, k}, X::Array{Float64,2}, addedidx::Array{Int64,2}) where {m, Nψ, k}

function update_storage(S::Storage, X, addedidx::Array{Int64,2}; isrescaled::Bool=true)
    NxX, Ne = size(X)
    Nψ = S.Nψ

    @assert NxX == S.Nx "Wrong dimension of the sample X"
    addedNψ = size(addedidx,1)
    newNψ = addedNψ + Nψ

    fnew = ParametricFunction(ExpandedFunction(S.f.f.B, vcat(S.f.f.idx, addedidx), vcat(S.f.f.coeff, zeros(addedNψ))))

    # Update off-diagonal component
    addedψoff = evaluate_offdiagbasis(fnew, X, addedidx)

    # Update ψd
    addedψd = evaluate_diagbasis(fnew, X, addedidx)

    # Update ψd0
    addedψd0  = repeated_evaluate_basis(fnew.f, zeros(Ne), addedidx)

    # Update dψxd
    addeddψxd = repeated_grad_xk_basis(fnew.f, X[S.Nx,:], addedidx)

    oldmaxj = maximum(S.f.f.idx[:,end])
    newmaxj = maximum(fnew.f.idx[:,end])

    @assert newmaxj >= oldmaxj "Error in the adaptive procedure, the set is not downward closed"

    # This avoid to allocate a new array

    addedcache_dcψxdt = zeros(Ne, addedNψ)
    @. addedcache_dcψxdt = addedψoff * addedψd

    ψnorm_new = norm.(eachcol(addedcache_dcψxdt))

    @inbounds for col in eachcol(addedcache_dcψxdt)
        normalize!(col)
    end

    # Add recursively the different multi indices
    # @inbounds for i=1:addedNψ
    #     add_column_householder!(S.F, addedcache_dcψxdt[:,i])
    # end

    if isrescaled == true
        addedψoff ./= ψnorm_new'
        addedψd   ./= ψnorm_new'
        addedψd0  ./= ψnorm_new'
        addedψxd ./= ψnorm_new'
    end

    fill!(addedcache_dcψxdt, 0.0)

    return Storage(S.m, newNψ, S.Nx, fnew,  hcat(S.ψoff, addedψoff),
                                            hcat(S.ψd, addedψd),
                                            hcat(S.ψd0, addedψd0),
                                            hcat(S.dψxd, addedψxd),
                                            ψnormnew,
                                            UpperTriangular(zeros(5,5)),
                                            UpperTriangular(zeros(5,5)),
                                            hcat(S.cache_dcψxdt, addedcache_dcψxdt),
                                            hcat(S.cache_gradxd, zeros(Ne,newmaxj-oldmaxj)),
                                            S.cache_dψxd,
                                            vcat(S.cache_integral, zeros(Ne*addedNψ)),
                                            S.cache_g)
end
