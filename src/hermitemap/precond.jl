import LinearAlgebra: ldiv!, dot

export Preconditioner, InvPreconditioner, precond!


"""
$(TYPEDEF)

An immutable structure to hold a preconditioner and its Cholesky factorization.

## Fields
$(TYPEDFIELDS)

"""
struct Preconditioner
    P::Symmetric{Float64}

    F::Cholesky{Float64, Matrix{Float64}}
end

function Preconditioner(P::Matrix{Float64})
    return Preconditioner(Symmetric(P), cholesky(Symmetric(P)))
end

ldiv!(x, P::Preconditioner, b) = copyto!(x, P.F \ b)
dot(A::Array, P::Preconditioner, B::Vector) = dot(A, P.P, B)


"""
    InvPreconditioner

An immutable structure to hold the inverse of the preconditioner.
For instance, this structure can be used to hold the estimate of the inverse of the Hessian in the BFGS algorithm.
"""
struct InvPreconditioner
    InvP::Symmetric{Float64}

    F::Cholesky{Float64, Matrix{Float64}}
end

function InvPreconditioner(InvP::Matrix{Float64})
    return InvPreconditioner(Symmetric(InvP), cholesky(Symmetric(InvP)))
end

ldiv!(x, P::InvPreconditioner, b) = copyto!(x, P.InvP * b)
dot(A::Array, P::InvPreconditioner, B::Vector) = ldiv!(A, P.F, B)

"""
$(TYPEDSIGNATURES)

Computes in-place a Hessian preconditioner based on the Gauss-Newton approximation (outer-product of gradient of the cost function).
"""
function precond!(P, coeff, S::Storage, C::HermiteMapComponent, X)
    Nψ = C.Nψ
    NxX, Ne = size(X)
    @assert NxX == C.Nx "Wrong dimension of the sample X"
    @assert size(S.ψoff, 1) == Ne
    @assert size(S.ψoff, 2) == Nψ

    # Output objective, gradient
    xlast = view(X,NxX,:)#)

    fill!(S.cache_integral, 0)

    # Integrate at the same time for the objective, gradient
    function integrand!(v::Vector{Float64}, t::Float64)
        repeated_grad_xk_basis!(S.cache_dcψxdt, S.cache_gradxd, C.I.f, t*xlast)

        # @avx @. S.cache_dψxd = (S.cache_dcψxdt .* S.ψoff) *ˡ coeff
        @avx @. S.cache_dcψxdt *= S.ψoff
        mul!(S.cache_dψxd, S.cache_dcψxdt, coeff)

        # Integration for J
        vJ = view(v,1:Ne)
        evaluate!(vJ, C.I.g, S.cache_dψxd)

        # Integration for dcJ

        grad_x!(S.cache_dψxd, C.I.g, S.cache_dψxd)

        # @avx for j=2:Nψ+1
        #     for i=1:Ne
        #         v[(j-1)*Ne+i] = S.cache_dψxd[i]*S.cache_dcψxdt[i,j-1]
        #     end
        # end

        v[Ne+1:Ne+Ne*Nψ] .= reshape(S.cache_dψxd .* S.cache_dcψxdt , (Ne*Nψ))
        # v[Ne+1:Ne+Ne*Nψ] .= reshape(grad_x(C.I.g, S.cache_dψxd) .* S.cache_dcψxdt , (Ne*Nψ))
    end

    quadgk!(integrand!, S.cache_integral, 0.0, 1.0)#; rtol = 1e-3)#; order = 9, rtol = 1e-10)

    # Multiply integral by xk (change of variable in the integration)
    @inbounds for j=1:Nψ+1
        @. S.cache_integral[(j-1)*Ne+1:j*Ne] *= xlast
    end
    # @avx for j=1:Nψ+1
    #     for i=1:Ne
    #         S.cache_integral[(j-1)*Ne+i] *= xlast[i]
    #     end
    # end

    # Add f(x_{1:d-1},0) i.e. (S.ψoff .* S.ψd0)*coeff to S.cache_integral
    @avx for i=1:Ne
        f0i = zero(Float64)
        for j=1:Nψ
            f0i += S.ψoffψd0[i,j]*coeff[j]
        end
        S.cache_integral[i] += f0i
    end

    # Store g(∂_{xk}f(x_{1:k})) in S.cache_g
    @avx for i=1:Ne
        prelogJi = zero(Float64)
        for j=1:Nψ
            prelogJi += S.ψoffdψxd[i,j]*coeff[j]
        end
        S.cache_g[i] = prelogJi
    end

    reshape_cacheintegral = reshape(S.cache_integral[Ne+1:Ne+Ne*Nψ], (Ne, Nψ))

    fill!(P, 0.0)
    @inbounds for l=1:Ne
        # Exploit symmetry of the Hessian
        for i=1:Nψ
            for j=i:Nψ
            # P[i,j] +=  reshape2_cacheintegral[l,i,j]*S.cache_integral[l]
            P[i,j] +=  (reshape_cacheintegral[l,i] + S.ψoffψd0[l,i]) * (reshape_cacheintegral[l,j] + S.ψoffψd0[l,j])
            P[i,j] -=  ( (S.ψoffdψxd[l,i]) * (S.ψoffdψxd[l,j])*(
                            hess_x(C.I.g, S.cache_g[l]) * C.I.g(S.cache_g[l]) -
                            grad_x(C.I.g, S.cache_g[l])^2))/C.I.g(S.cache_g[l])^2

            P[j,i] = P[i,j]
            end
        end
    end
    rmul!(P, 1/Ne)
    # Add derivative of the L2 penalty term ∂^2_c α ||c||^2 = 2 *α *I
    @inbounds for i=1:Nψ
        P[i,i] += 2*C.α*I
    end
    return P
end

"""
$(TYPEDSIGNATURES)

Computes in-place a Hessian preconditioner based on the Gauss-Newton approximation (outer-product of gradient of the cost function).
"""
precond!(S::Storage, C::HermiteMapComponent, X) = (P, coeff) -> precond!(P, coeff, S, C, X)


"""
$(TYPEDSIGNATURES)

Computes in-place a Hessian preconditioner based on the diagonal entries of the Gauss-Newton approximation (outer-product of gradient of the cost function).
"""
function diagprecond!(P, coeff, S::Storage, C::HermiteMapComponent, X::Array{Float64,2})
    Nψ = C.Nψ
    Nx = C.Nx
    NxX, Ne = size(X)
    @assert NxX == Nx "Wrong dimension of the sample X"
    @assert size(S.ψoff, 1) == Ne
    @assert size(S.ψoff, 2) == Nψ

    # Output objective, gradient
    xlast = view(X, Nx,:)#)

    fill!(S.cache_integral, 0)

    # Integrate at the same time for the objective, gradient
    function integrand!(v::Vector{Float64}, t::Float64)
        S.cache_dcψxdt .= repeated_grad_xk_basis(C.I.f, t*xlast)

        # @avx @. S.cache_dψxd = (S.cache_dcψxdt .* S.ψoff) *ˡ coeff
        S.cache_dcψxdt .*= S.ψoff
        mul!(S.cache_dψxd, S.cache_dcψxdt, coeff)

        # Integration for J
        vJ = view(v,1:Ne)
        evaluate!(vJ, C.I.g, S.cache_dψxd)
        # v[1:Ne] .= C.I.g(S.cache_dψxd)

        # Integration for dcJ
        v[Ne+1:Ne+Ne*Nψ] .= reshape(grad_x(C.I.g, S.cache_dψxd) .* S.cache_dcψxdt , (Ne*Nψ))
    end

    quadgk!(integrand!, S.cache_integral, 0.0, 1.0)#; rtol = 1e-3)#; order = 9, rtol = 1e-10)

    # Multiply integral by xk (change of variable in the integration)
    @inbounds for j=1:Nψ+1
        @. S.cache_integral[(j-1)*Ne+1:j*Ne] *= xlast
    end

    # Add f(x_{1:d-1},0) i.e. (S.ψoff .* S.ψd0)*coeff to S.cache_integral
    @avx for i=1:Ne
        f0i = zero(Float64)
        for j=1:Nψ
            f0i += S.ψoffψd0[i,j]*coeff[j]
        end
        S.cache_integral[i] += f0i
    end


    # Store g(∂_{xk}f(x_{1:k})) in S.cache_g
    @avx for i=1:Ne
        prelogJi = zero(Float64)
        for j=1:Nψ
            prelogJi += S.ψoffdψxd[i,j]*coeff[j]
        end
        S.cache_g[i] = prelogJi
    end


    reshape_cacheintegral = reshape(S.cache_integral[Ne+1:Ne+Ne*Nψ], (Ne, Nψ))
    # reshape2_cacheintegral = reshape(S.cache_integral[Ne + Ne*Nψ + 1: Ne + Ne*Nψ + Ne*Nψ*Nψ], (Ne, Nψ, Nψ))
    fill!(P, 0.0)
    @inbounds for l=1:Ne
        # Exploit symmetry of the Hessian
        for i=1:Nψ
            # P[i,j] +=  reshape2_cacheintegral[l,i,j]*S.cache_integral[l]
            P[i] +=  (reshape_cacheintegral[l,i] + S.ψoffψd0[l,i])^2# * (reshape_cacheintegral[l,j] + S.ψoff[l,j]*S.ψd0[l,j])
            P[i] -=  ( (S.ψoffdψxd[l,i])^2*(
                            hess_x(C.I.g, S.cache_g[l]) * C.I.g(S.cache_g[l]) -
                            grad_x(C.I.g, S.cache_g[l])^2))/C.I.g(S.cache_g[l])^2
        end
    end
    rmul!(P, 1/Ne)
    # Add derivative of the L2 penalty term ∂^2_c α ||c||^2 = 2 *α *I
    @inbounds for i=1:Nψ
        P[i] += 2*C.α
    end
    return Diagonal(P)
end

"""
$(TYPEDSIGNATURES)

Computes in-place a Hessian preconditioner based on the diagonal entries of the Gauss-Newton approximation (outer-product of gradient of the cost function).
"""
diagprecond!(S::Storage, C::HermiteMapComponent, X::Array{Float64,2}) = (P, coeff) -> diagprecond!(P, coeff, S, C, X)
