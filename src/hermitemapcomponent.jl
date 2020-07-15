export HermiteMapk,
       log_pdf,
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

function log_pdf(Hk::HermiteMapk{m, Nψ, k}, ens::EnsembleState{k,Ne}) where {m, Nψ, k, Nx, Ne}

    Sx = evaluate(Hk.I, ens)
    dxSx = grad_xd(Hk.I, ens)
    return log_pdf.(Normal())

end

function negative_log_likelihood(Hk::HermiteMapk{m, Nψ, k}, ens::EnsembleState{k, Ne}) where {m, Nψ, k, Ne}
    # Output objective, gradient and hessian
    ψoff = evaluate_offdiagbasis(R.f, ens)
    xk = deepcopy(ens.S[end,:])
    cache = zeros(Ne + Ne*Nψ + Ne*Nψ*Nψ)

    dcdψ = zeros(Ne, Nψ)
    dcdψouter = zeros(Ne, Nψ, Nψ)
    dψxd = zeros(Ne)
    # Integrate at the same time for the objective, gradient and hessian
    function integrand!(v::Vector{Float64}, t::Float64)
        dcdψ .= repeatgrad_xk_basis(R.f.f,  0.5*(t+1)*xk) .* ψoff
        dψxd .= dcdψ*R.f.f.coeff

        vobj = view(v, 1:Ne)
        vobj =  R.g(dψxd)

        vgrad = view(v, Ne+1:Ne+Ne*Nψ)
        vgrad = reshape(grad_x(R.g, dψxd) .* dcdψ , (Ne*Nψ))

        vhess = view(v, Ne + Ne*Nψ + 1: Ne + Ne*Nψ + Ne*Nψ*Nψ)
        @inbounds for i=1:Nψ
            for j=1:Nψ
                dcdψouter[:,i,j] = dcdψ[:,i] .* dcdψ[:, j]
            end
        end
        vhess = reshape(hess_x(R.g, dψxd) .* dcdψouter, (Ne*Nψ*Nψ))
    end

    0.5*xk .* quadgk!(integrand!, cache, -1, 1)[1]

end
