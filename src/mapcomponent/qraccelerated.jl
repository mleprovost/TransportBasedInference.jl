export qrnegative_log_likelihood!

function qrnegative_log_likelihood!(J, dJ, coeff, S::Storage, C::MapComponent, X)
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
        mul!(S.cache_dψxd, S.cache_dcψxdt, coeff)

        # This doesn't work here, if we sue this line we need to add ψoff[i,j] in the last multiplication
        # @avx @. S.cache_dψxd = (S.cache_dcψxdt * S.ψoff) *ˡ coeff

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

    # Multiply integral by xlast (change of variable in the integration)
    # @avx for j=1:Nψ+1
    #     for i=1:Ne
    #         S.cache_integral[(j-1)*Ne+i] *= xlast[i]
    #     end
    # end

    # Multiply integral by xlast (change of variable in the integration)
    @inbounds for j=1:Nψ+1
        @. S.cache_integral[(j-1)*Ne+1:j*Ne] *= xlast
    end

    # Add f(x_{1:d-1},0) i.e. (S.ψoff .* S.ψd0)*coeff to S.cache_integral
    @avx for i=1:Ne
        f0i = zero(Float64)
        for j=1:Nψ
            f0i += (S.ψoff[i,j] * S.ψd0[i,j])*coeff[j]
        end
        S.cache_integral[i] += f0i
    end

    # Store g(∂_{xk}f(x_{1:k})) in S.cache_g
    @avx for i=1:Ne
        prelogJi = zero(Float64)
        for j=1:Nψ
            prelogJi += (S.ψoff[i,j] * S.dψxd[i,j])*coeff[j]
        end
        S.cache_g[i] = prelogJi
    end

    # Formatting to use with Optim.jl
    if dJ != nothing
        reshape_cacheintegral = reshape(S.cache_integral[Ne+1:end], (Ne, Nψ))
        fill!(dJ, 0.0)
        @inbounds for i=1:Ne
            for j=1:Nψ
            dJ[j] += gradlog_pdf(S.cache_integral[i])*(reshape_cacheintegral[i,j] + S.ψoff[i,j]*S.ψd0[i,j]) + # dsoftplus(S.cache_g[i])*S.ψoff[i,j]*S.dψxd[i,j]*(1/softplus(S.cache_g[i]))
                     grad_x(C.I.g, S.cache_g[i])*S.ψoff[i,j]*S.dψxd[i,j]/C.I.g(S.cache_g[i])
            end
        end
        rmul!(dJ, -1/Ne)
        # Add derivative of the L2 penalty term ∂_c α ||c||^2 = 2 *α c
        dJ .+= 2*C.α*coeff
    end

    if J != nothing
        J = 0.0
        @avx for i=1:Ne
            J += log_pdf(S.cache_integral[i]) + log(C.I.g(S.cache_g[i]))
        end
        J *=(-1/Ne)
        J += C.α*norm(coeff)^2
        return J
    end
end
