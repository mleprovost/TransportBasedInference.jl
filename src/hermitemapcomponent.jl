export HermiteMapk,
       log_pdf,
       negative_log_likelihood!


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

function negative_log_likelihood!(Hk::HermiteMapk{m, Nψ, k}, X::Array{Float64,2}) where {m, Nψ, k}


    # Output objective, gradient
    ψoff = evaluate_offdiagbasis(R.f, ens)
    xk = deepcopy(ens.S[end,:])
    cache = zeros(Ne + Ne*Nψ)

    dcdψ = zeros(Ne, Nψ)
    dψxd = zeros(Ne)
    # Integrate at the same time for the objective, gradient
    function integrand!(v::Vector{Float64}, t::Float64)
        dcdψ .= repeatgrad_xk_basis(R.f.f,  0.5*(t+1)*xk) .* ψoff
        dψxd .= dcdψ*R.f.f.coeff

        vobj = view(v, 1:Ne)
        vobj =  R.g(dψxd)

        vgrad = view(v, Ne+1:Ne+Ne*Nψ)
        vgrad = reshape(grad_x(R.g, dψxd) .* dcdψ , (Ne*Nψ))
    end

    0.5*xk .* quadgk!(integrand!, cache, -1, 1)[1]

end
