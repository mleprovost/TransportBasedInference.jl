export HermiteMapk,
       log_pdf,
       negative_log_likelihood!,
       negative_log_likelihood


struct HermiteMapk{m, Nψ, k}
    # IntegratedFunction
    I::IntegratedFunction{m, Nψ, k}
    # Regularization parameter
    α::Float64

    function HermiteMapk(I::IntegratedFunction{m, Nψ, k}; α::Float64 = 1e-6) where {m, Nψ, k}
        return new{m, Nψ, k}(I, α)
    end
end

function log_pdf(Hk::HermiteMapk{m, Nψ, k}, X::Array{Float64,2}) where {m, Nψ, k, Nx}

    Sx = evaluate(Hk.I, ens)
    dxSx = grad_xd(Hk.I, ens)
    return log_pdf.(Normal())

end

function negative_log_likelihood(S::Storage{m, Nψ, k}, Hk::HermiteMapk{m, Nψ, k}, X::Array{Float64,2}) where {m, Nψ, k}
    NxX, Ne = size(X)
    @assert NxX == k "Wrong dimension of the sample X"
    @assert size(S.ψoff, 1) == Ne
    @assert size(S.ψoff, 2) == Nψ

    # Output objective, gradient
    xk = deepcopy(X[NxX,:])#)

    coeff = Hk.I.f.f.coeff


    dcdψ = zeros(Ne, Nψ)
    dψxd = zeros(Ne)
    J = 0.0
    dJ = zeros(Nψ)
    # Integrate at the same time for the objective, gradient
    function integrand!(v::Vector{Float64}, t::Float64)
        S.cache_dcψxdt.= repeated_grad_xk_basis(Hk.I.f.f,  0.5*(t+1)*xk) .* S.ψoff

        mul!(S.cache_dψxd, S.cache_dcψxdt, Hk.I.f.f.coeff)

        # Integration for J
        v[1:Ne] .= Hk.I.g(S.cache_dψxd)
        # vobj =

        # Integration for dcJ
        v[Ne+1:Ne+Ne*Nψ] .= reshape(grad_x(Hk.I.g, S.cache_dψxd) .* S.cache_dcψxdt , (Ne*Nψ))
    end
    S.cache_integral .= quadgk!(integrand!, S.cache_integral, -1, 1; rtol = 1e-3)[1]
    logdψk = log.(Hk.I.g(repeated_grad_xk_basis(Hk.I.f.f, xk) .* S.ψoff *Hk.I.f.f.coeff))
    quad  =  (S.ψoff .* S.ψd0)*Hk.I.f.f.coeff + 0.5*xk .* S.cache_integral[1:Ne]
    for i=1:Ne
        J += logpdf.(Normal(), quad[i]) +  logdψk[i]
    end

    J *=(-1/Ne)

    return J

    #  + 0.5*xk .* quadgk!(integrand!, cache, -1, 1)[1]

end


function negative_log_likelihood!(S::Storage{m, Nψ, k}, Hk::HermiteMapk{m, Nψ, k}, X::Array{Float64,2}) where {m, Nψ, k}
    NxX, Ne = size(X)
    @assert NxX == k "Wrong dimension of the sample X"
    @assert size(S.ψoff, 1) == Ne
    @assert size(S.ψoff, 2) == Nψ

    # Output objective, gradient
    xk = view(X,NxX,:)#)

    coeff = Hk.I.f.f.coeff

    J = 0.0
    dJ = zeros(Nψ)
    # Integrate at the same time for the objective, gradient
    function integrand!(v::Vector{Float64}, t::Float64)
        S.cache_dcψxdt .= repeated_grad_xk_basis(Hk.I.f.f, t*xk)
        S.cache_dcψxdt .*= S.ψoff

        mul!(S.cache_dψxd, S.cache_dcψxdt, coeff)

        # Integration for J
        v[1:Ne] .= Hk.I.g(S.cache_dψxd)

        # Integration for dcJ
        v[Ne+1:Ne+Ne*Nψ] .= reshape(grad_x(Hk.I.g, S.cache_dψxd) .* S.cache_dcψxdt , (Ne*Nψ))
    end

    quadgk!(integrand!, S.cache_integral, 0, 1; rtol = 1e-3)

    # Multiply integral by xk (change of variable in the integration)
    for j=1:Nψ+1
        @. S.cache_integral[(j-1)*Ne+1:j*Ne] *= xk
    end

    # Add f(x_{1:d-1},0) i.e. (S.ψoff .* S.ψd0)*coeff to S.cache_integral
    # Compute ∂_{xk}g(∂_{xk}f(x_{1:k}))
    @avx for i=1:Ne
        f0i = zero(Float64)
        prelogJi = zero(Float64)
        for j=1:Nψ
            f0i += (S.ψoff[i,j] * S.ψd0[i,j])*coeff[j]
            prelogJi += (S.ψoff[i,j] * S.dψxd[i,j])*coeff[j]
        end
        S.cache_integral[i] += f0i
        S.cache_integral[Ne+i] += prelogJi
    end

    @inbounds for i=1:Ne
        J += logpdf(Normal(), S.cache_integral[i]) + log(Hk.I.g(S.cache_integral[Ne+i]))
    end

    # for i=1:Nψ
    #     for
    # logdψk = log.(Hk.I.g(S.dψxd .* S.ψoff *Hk.I.f.f.coeff))
    # quad  =  (S.ψoff .* S.ψd0)*coeff + xk .* S.cache_integral
    #  + 0.5*xk .* quadgk!(integrand!, cache, -1, 1)[1]

    J *=(-1/Ne)

    return J
end
