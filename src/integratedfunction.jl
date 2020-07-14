

export IntegratedFunction, grad_xd,
       grad_coeff_grad_xd,
       hess_coeff_grad_xd,
       repeatgrad_xk_basis,
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


## integrate_xd
# Compute ∫_0^{x_k} g(∂ₖf(x_{1:k-1},t)) dt
# function integrate_xd(R::IntegratedFunction{m, Nψ, Nx}, x::Array{T,1}) where {m, Nψ, Nx, Ne, T <: Real}
#      return quadgk(t->R.g(ForwardDiff.gradient(y->R.f.f(y), vcat(x[1:end-1],t))[end]), 0, x[end])[1]
# end
#

function repeatgrad_xk_basis(f::ExpandedFunction{m, Nψ, Nx}, x::Array{Float64,1}) where {m, Nψ, Nx}
    # Compute the k=th order deriviative of an expanded function along the direction grad_dim
    N = size(x,1)
    # ∂ᵏf/∂x_{grad_dim} = ψ
    k = 1
    grad_dim = Nx
    dims = Nx

    dkψ = ones(N, Nψ)
    midxj = f.idx[:,Nx]
    maxj = maximum(midxj)
    #   Compute the kth derivative along grad_dim
    dkψj = vander(f.B.B, maxj, k, x)
    return dkψj[:, midxj .+ 1]

end
function integrate_xd(R::IntegratedFunction{m, Nψ, Nx}, ens::EnsembleState{Nx, Ne}) where {m, Nψ, Nx, Ne}
    ψoff = evaluate_offdiagbasis(R.f, ens)
    xk = deepcopy(ens.S[Nx, :])
    cache = zeros(Ne)
    # ψoffdxdψ = zeros(Ne, Nψ)

    function integrand!(v::Vector{Float64}, t::Float64)
        # ψoffdxdψ .= repeatgrad_xk_basis(R.f.f,  0.5*(t+1)*xk)
        # ψoffdxdψ .*= ψoff
        # map!(R.g, v, ψoffdxdψ*R.f.f.coeff)
        v .= R.g((repeatgrad_xk_basis(R.f.f,  0.5*(t+1)*xk) .* ψoff)*R.f.f.coeff)
    end

    return 0.5*xk .* quadgk!(integrand!, cache, -1, 1)[1]
end

evaluate(R::IntegratedFunction{m, Nψ, Nx}, ens::EnsembleState{Nx, Ne}) where {m, Nψ, Nx ,Ne} =  map!(i -> R(member(ens,i)), zeros(Ne), 1:Ne)


function grad_coeff_integrate_xd(R::IntegratedFunction{m, Nψ, Nx}, x::EnsembleState{Nx, Ne}) where {m, Nψ, Nx, Ne}

end
