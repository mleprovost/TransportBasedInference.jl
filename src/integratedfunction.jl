

export IntegratedFunction, grad_xd,
       grad_coeff_grad_xd,
       hess_coeff_grad_xd,
       integrate_xd,
       evaluate


struct IntegratedFunction{m, Nψ, Nx}
    g::Rectifier
    f::ParametricFunction{m, Nψ, Nx}
end

function IntegratedFunction(f::ParametricFunction{m, Nψ, Nx}) where {m, Nψ, Nx}
    return IntegratedFunction{m, Nψ, Nx}(Rectifier("softplus"), f)
end


function (R::IntegratedFunction{m, Nψ, Nx})(x::Array{T,1}) where {m, Nψ, Nx, T<:Real}
    return R.f.f(vcat(x[1:end-1], 0.0)) + quadgk(t->R.g(ForwardDiff.gradient(y->R.f.f(y), vcat(x[1:end-1],t))[end]), 0, x[end])[1]
end

# Compute g(∂ₖf(x_{1:k}))
function grad_xd(R::IntegratedFunction{m, Nψ, Nx}, ens::EnsembleState{Nx, Ne}) where {m, Nψ, Nx, Ne}
    dψ = grad_xd(R.f, ens)
    gdψ = R.g.(dψ)
    return gdψ
end

# Compute ∂_c( g(∂ₖf(x_{1:k}) ) ) = ∂_c∂ₖf(x_{1:k}) × g′(∂ₖf(x_{1:k}))
function grad_coeff_grad_xd(R::IntegratedFunction{m, Nψ, Nx}, ens::EnsembleState{Nx, Ne}) where {m, Nψ, Nx, Ne}
    dψ = grad_xd(R.f, ens)
    dcdψ = grad_coeff_grad_xd(R.f, ens)
    return grad_x(R.g, dψ) .* dcdψ
end

# Compute ∂²_c( g(∂ₖf(x_{1:k}) ) ) = ∂²_c∂ₖf(x_{1:k}) × g′(∂ₖf(x_{1:k})) + ∂_c∂ₖf(x_{1:k}) × ∂_c∂ₖf(x_{1:k}) g″(∂ₖf(x_{1:k}))
function hess_coeff_grad_xd(R::IntegratedFunction{m, Nψ, Nx}, ens::EnsembleState{Nx, Ne}) where {m, Nψ, Nx, Ne}
    dψ    = grad_xd(R.f, ens)
    dcdψ  = grad_coeff_grad_xd(R.f, ens)
    d2cdψ = hess_coeff_grad_xd(R.f.f, ens)

    dcdψouter = zeros(Ne, Nψ, Nψ)
    @inbounds for i=1:Nψ
        for j=1:Nψ
            dcdψouter[:,i,j] = dcdψ[:,i] .* dcdψ[:, j]
        end
    end
    return dψ .* d2cdψ + hess_x(R.g, dψ) .* dcdψouter
end


# Compute ∫_0^{x_k} g(∂ₖf(x_{1:k-1},t)) dt
# function integrate_xd(R::IntegratedFunction{m, Nψ, Nx}, x::Array{T,1}) where {m, Nψ, Nx, Ne, T <: Real}
#      return quadgk(t->R.g(ForwardDiff.gradient(y->R.f.f(y), vcat(x[1:end-1],t))[end]), 0, x[end])[1]
# end
#
function integrate_xd(R::IntegratedFunction{m, Nψ, Nx}, ens::EnsembleState{Nx, Ne}) where {m, Nψ, Nx, Ne}
    ψ = evaluate_offdiagbasis(Rfp, ens)
end

evaluate(R::IntegratedFunction{m, Nψ, Nx}, ens::EnsembleState{Nx, Ne}) where {m, Nψ, Nx ,Ne} =  map!(i -> R(member(ens,i)), zeros(Ne), 1:Ne)


function grad_coeff_integrate_xd(R::IntegratedFunction{m, Nψ, Nx}, x::EnsembleState{Nx, Ne}) where {m, Nψ, Nx, Ne}

end
