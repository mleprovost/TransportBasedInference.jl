export QRscaling, qrnegative_log_likelihood!,
                  qrnegative_log_likelihood,
                  qrprecond!,
                  qrprecond



struct QRscaling
    R::UpperTriangular{Float64,Array{Float64,2}}
    Rinv::UpperTriangular{Float64,Array{Float64,2}}
    D::Diagonal{Float64, Array{Float64,1}}
    Dinv::Diagonal{Float64, Array{Float64,1}}
    U::UpperTriangular{Float64,Array{Float64,2}}
    Uinv::UpperTriangular{Float64,Array{Float64,2}}
    L2Uinv::Array{Float64,2}
end

function QRscaling(R, ψnorm)
    D = Diagonal(ψnorm)
    Dinv = inv(D)
    R = UpperTriangular(R)
    Rinv = inv(R)
    U = UpperTriangular(R*D)
    Uinv = Dinv*Rinv
    L2Uinv = Uinv'*Uinv
    return QRscaling(R, Rinv, D, Dinv, U, Uinv, L2Uinv)
end

function QRscaling(S::Storage)
    D = Diagonal(S.ψnorm)
    Dinv = inv(D)
    R = UpperTriangular(qr!(S.ψoffψd * Dinv).R)
    Rinv = inv(R)
    U = UpperTriangular(R*D)
    Uinv = Dinv*Rinv
    L2Uinv = Uinv'*Uinv
    return QRscaling(R, Rinv, D, Dinv, U, Uinv, L2Uinv)
end

function qrnegative_log_likelihood!(J̃, dJ̃, c̃oeff, F::QRscaling, S::Storage, C::MapComponent, X)
    # In this version, c̃oeff is expressed in the rescaled space
    NxX, Ne = size(X)
    m = C.m
    Nx = C.Nx
    Nψ = C.Nψ
    @assert NxX == Nx "Wrong dimension of the sample X"
    @assert size(S.ψoff, 1) == Ne
    @assert size(S.ψoff, 2) == Nψ

    # Output objective, gradient
    xlast = view(X,NxX,:)

    fill!(S.cache_integral, 0.0)

    # Integrate at the same time for the objective, gradient
    function integrand!(v::Vector{Float64}, t::Float64)
        repeated_grad_xk_basis!(S.cache_dcψxdt, S.cache_gradxd, C.I.f.f, t*xlast)

         # This computing is also reused in the computation of the gradient, no interest to skip it
        @avx @. S.cache_dcψxdt *= S.ψoff
        mul!(S.cache_dcψxdt, S.cache_dcψxdt, F.Uinv)
        mul!(S.cache_dψxd, S.cache_dcψxdt, c̃oeff)

        # This doesn't work here, if we use this line we need to add ψoff[i,j] in the last multiplication
        # @avx @. S.cache_dψxd = (S.cache_dcψxdt * S.ψoff) *ˡ coeff

        # Integration for J̃
        vJ = view(v,1:Ne)
        evaluate!(vJ, C.I.g, S.cache_dψxd)

        # Integration for dcJ̃
        grad_x!(S.cache_dψxd, C.I.g, S.cache_dψxd)

        v[Ne+1:Ne+Ne*Nψ] .= reshape(S.cache_dψxd .* S.cache_dcψxdt , (Ne*Nψ))
    end

    quadgk!(integrand!, S.cache_integral, 0.0, 1.0; rtol = 1e-3)#; order = 9, rtol = 1e-10)

    # Multiply integral by xlast (change of variable in the integration)
    @inbounds for j=1:Nψ+1
        @. S.cache_integral[(j-1)*Ne+1:j*Ne] *= xlast
    end

    # Add f(x_{1:d-1},0) i.e. (S.ψoff .* S.ψd0)*coeff to S.cache_integral
    @avx for i=1:Ne
        f0i = zero(Float64)
        for j=1:Nψ
            f0i += S.ψoffψd0[i,j]*c̃oeff[j]
        end
        S.cache_integral[i] += f0i
    end

    # Store g(∂_{xk}f(x_{1:k})) in S.cache_g
    @avx for i=1:Ne
        prelogJi = zero(Float64)
        for j=1:Nψ
            prelogJi += S.ψoffdψxd[i,j]*c̃oeff[j]
        end
        S.cache_g[i] = prelogJi
    end

    # Formatting to use with Optim.jl
    if dJ̃ != nothing
        reshape_cacheintegral = reshape(S.cache_integral[Ne+1:end], (Ne, Nψ))
        fill!(dJ̃, 0.0)
        @inbounds for i=1:Ne
            for j=1:Nψ
            dJ̃[j] += gradlog_pdf(S.cache_integral[i])*(reshape_cacheintegral[i,j] + S.ψoffψd0[i,j]) + # dsoftplus(S.cache_g[i])*S.ψoff[i,j]*S.dψxd[i,j]*(1/softplus(S.cache_g[i]))
                     grad_x(C.I.g, S.cache_g[i])*S.ψoffdψxd[i,j]/C.I.g(S.cache_g[i])
            end
        end
        # rmul!(dJ̃, -1/Ne)
        # Add derivative of the L2 penalty term ∂_c α ||U^{-1}c||^2 = 2 α U^{-1}c
        mul!(dJ̃, F.L2Uinv, c̃oeff, 2*C.α, -1/Ne)
        # dJ̃ .+= 2*C.α*F.L2Uinv*coeff
    end

    if J̃ != nothing
        J̃ = 0.0
        @avx for i=1:Ne
            J̃ += log_pdf(S.cache_integral[i]) + log(C.I.g(S.cache_g[i]))
        end
        J̃ *=(-1/Ne)
        J̃ += C.α*norm(F.Uinv*c̃oeff)^2
        return J̃
    end
end

qrnegative_log_likelihood(F::QRscaling, S::Storage, C::MapComponent, X) = (J̃, dJ̃, c̃oeff) -> qrnegative_log_likelihood!(J̃, dJ̃, c̃oeff, F, S, C, X)


function qrprecond!(P, coeff, F::QRscaling, S::Storage, C::MapComponent, X)
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
        repeated_grad_xk_basis!(S.cache_dcψxdt, S.cache_gradxd, C.I.f.f, t*xlast)

        # @avx @. S.cache_dψxd = (S.cache_dcψxdt .* S.ψoff) *ˡ coeff
        @avx @. S.cache_dcψxdt *= S.ψoff
        mul!(S.cache_dcψxdt, S.cache_dcψxdt, F.Uinv)
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

    quadgk!(integrand!, S.cache_integral, 0.0, 1.0; rtol = 1e-3)#; order = 9, rtol = 1e-10)

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
    # Add derivative of the L2 penalty term ∂^2_c̃ α ||Uinv c̃||^2 = ∂^2_c̃ (α c̃' Uinv' Uinv c̃) = 2*α Uinv'*Uinv

    P .+= 2*C.α*F.L2Uinv
    return P
end

qrprecond!(S::Storage, C::MapComponent, X) = (P, coeff) -> precond!(P, coeff, S, C, X)
