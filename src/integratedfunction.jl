

export  IntegratedFunction,
        grad_xd,
        grad_coeff_grad_xd,
        hess_coeff_grad_xd,
        repeated_grad_xk_basis,
        integrate_xd,
        evaluate!,
        evaluate,
        grad_coeff_integrate_xd,
        hess_coeff_integrate_xd,
        grad_coeff,
        hess_coeff,
        evalgrad_coeff!


struct IntegratedFunction{m, Nψ, Nx}
    g::Rectifier
    f::ParametricFunction{m, Nψ, Nx}
end

function IntegratedFunction(f::ParametricFunction{m, Nψ, Nx}) where {m, Nψ, Nx}
    return IntegratedFunction{m, Nψ, Nx}(Rectifier("softplus"), f)
end

function IntegratedFunction(f::ExpandedFunction{m, Nψ, Nx}) where {m, Nψ, Nx}
    return IntegratedFunction{m, Nψ, Nx}(Rectifier("softplus"), ParametricFunction(f))
end

function integrate_xd(R::IntegratedFunction{m, Nψ, Nx}, X::Array{Float64,2}) where {m, Nψ, Nx}
    NxX, Ne = size(X)
    ψoff = evaluate_offdiagbasis(R.f, X)
    xk = deepcopy(X[Nx, :])
    cache = zeros(Ne)
    # ψoffdxdψ = zeros(Ne, Nψ)

    function integrand!(v::Vector{Float64}, t::Float64)
        # ψoffdxdψ .= repeated_grad_xk_basis(R.f.f,  0.5*(t+1)*xk)
        # ψoffdxdψ .*= ψoff
        # map!(R.g, v, ψoffdxdψ*R.f.f.coeff)
        v .= R.g((repeated_grad_xk_basis(R.f.f,  0.5*(t+1)*xk) .* ψoff)*R.f.f.coeff)
    end

    return 0.5*xk .* quadgk!(integrand!, cache, -1, 1)[1]
end

# Compute g(∂ₖf(x_{1:k}))
function grad_xd(R::IntegratedFunction{m, Nψ, Nx}, X::Array{Float64,2}) where {m, Nψ, Nx}
    dψ = grad_xd(R.f, X)
    gdψ = R.g.(dψ)
    return gdψ
end

# Compute ∂_c( g(∂ₖf(x_{1:k}) ) ) = ∂_c∂ₖf(x_{1:k}) × g′(∂ₖf(x_{1:k}))
function grad_coeff_grad_xd(R::IntegratedFunction{m, Nψ, Nx}, X::Array{Float64,2}) where {m, Nψ, Nx}
    dψ = grad_xd(R.f, X)
    dcdψ = grad_coeff_grad_xd(R.f, X)
    return grad_x(R.g, dψ) .* dcdψ
end

# Compute ∂²_c( g(∂ₖf(x_{1:k}) ) ) = ∂²_c∂ₖf(x_{1:k}) × g′(∂ₖf(x_{1:k})) + ∂_c∂ₖf(x_{1:k}) × ∂_c∂ₖf(x_{1:k}) g″(∂ₖf(x_{1:k}))
function hess_coeff_grad_xd(R::IntegratedFunction{m, Nψ, Nx}, X::Array{Float64,2}) where {m, Nψ, Nx}
    # The second term can be dropped for improve performance
    NxX, Ne = size(X)
    dψ    = grad_xd(R.f, X)
    dcdψ  = grad_coeff_grad_xd(R.f, X)
    # d2cdψ = hess_coeff_grad_xd(R.f.f, X)

    dcdψouter = zeros(Ne, Nψ, Nψ)
    @inbounds for i=1:Nψ
        for j=1:Nψ
            dcdψouter[:,i,j] = dcdψ[:,i] .* dcdψ[:, j]
        end
    end
    return hess_x(R.g, dψ) .* dcdψouter # + dψ .* d2cdψ
end


## integrate_xd
# Compute ∫_0^{x_k} g(∂ₖf(x_{1:k-1},t)) dt
# function integrate_xd(R::IntegratedFunction{m, Nψ, Nx}, x::Array{T,1}) where {m, Nψ, Nx, Ne, T <: Real}
#      return quadgk(t->R.g(ForwardDiff.gradient(y->R.f.f(y), vcat(x[1:end-1],t))[end]), 0, x[end])[1]
# end
#

function repeated_grad_xk_basis(f::ExpandedFunction{m, Nψ, Nx}, x::Array{Float64,1}, idx::Array{Int64,2}) where {m, Nψ, Nx}
    # Compute the k=th order deriviative of an expanded function along the direction grad_dim
    N = size(x,1)
    # ∂ᵏf/∂x_{grad_dim} = ψ
    k = 1
    grad_dim = Nx
    dims = Nx

    midxj = idx[:,Nx]
    maxj = maximum(midxj)
    #   Compute the kth derivative along grad_dim
    dkψj = vander(f.B.B, maxj, k, x)
    return dkψj[:, midxj .+ 1]
end

repeated_grad_xk_basis(f::ExpandedFunction{m, Nψ, Nx}, x::Array{Float64,1}) where {m, Nψ, Nx} =
        repeated_grad_xk_basis(f, x, f.idx)

function evaluate!(out::Array{Float64,1}, R::IntegratedFunction{m, Nψ, Nx}, X::Array{Float64,2}) where {m, Nψ, Nx}
    NxX, Ne = size(X)
    ψoff = evaluate_offdiagbasis(R.f, X)
    ψdiag = repeated_evaluate_basis(R.f.f, zeros(Ne))
    xk = deepcopy(X[Nx, :])
    cache = zeros(Ne)

    @assert size(out,1) == Ne

    function integrand!(v::Vector{Float64}, t::Float64)
        v .= R.g((repeated_grad_xk_basis(R.f.f,  0.5*(t+1)*xk) .* ψoff)*R.f.f.coeff)
    end

     out .= (ψoff .* ψdiag)*R.f.f.coeff + 0.5*xk .* quadgk!(integrand!, cache, -1, 1)[1]

     return out
 end



evaluate(R::IntegratedFunction{m, Nψ, Nx}, X::Array{Float64,2}) where {m, Nψ, Nx} =
        evaluate!(zeros(size(X,2)), R, X)


## Compute ∂_c int_0^{x_k} g(∂ₖf(x_{1:k-1}, t))dt
function grad_coeff_integrate_xd(R::IntegratedFunction{m, Nψ, Nx}, X::Array{Float64,2}) where {m, Nψ, Nx}
    ψoff = evaluate_offdiagbasis(R.f, X)
    NxX, Ne = size(X)
    xk = deepcopy(X[Nx, :])
    dcdψ = zeros(Ne, Nψ)

    cache = zeros(Ne, Nψ)

    function integrand!(v::Matrix{Float64}, t::Float64)
        dcdψ .= repeated_grad_xk_basis(R.f.f,  0.5*(t+1)*xk) .* ψoff
        v .= grad_x(R.g, dcdψ*R.f.f.coeff) .* dcdψ
    end

    return 0.5*xk .* quadgk!(integrand!, cache, -1, 1)[1]
end

# Compute ∂²_c int_0^{x_k} g(c^T F(t))dt = int_0^{x_k} ∂_c F(t) g′(c^T F(t)) + F(t)F(t)*g″(c^T F(t)) dt
function hess_coeff_integrate_xd(R::IntegratedFunction{m, Nψ, Nx}, X::Array{Float64,2}) where {m, Nψ, Nx}
     # We drop the first term for speed improvement since it is always equal to 0
    NxX, Ne = size(X)
    ψoff = evaluate_offdiagbasis(R.f, X)
    xk = deepcopy(X[Nx, :])
    dcdψ = zeros(Ne, Nψ)

    dcdψouter = zeros(Ne, Nψ, Nψ)

    cache = zeros(Ne*Nψ*Nψ)

    function integrand!(v::Vector{Float64}, t::Float64)
        dcdψ .= repeated_grad_xk_basis(R.f.f,  0.5*(t+1)*xk) .* ψoff
        @inbounds for i=1:Nψ
            for j=1:Nψ
                dcdψouter[:,i,j] = dcdψ[:,i] .* dcdψ[:, j]
            end
        end
        v .= reshape(hess_x(R.g, (dcdψ ) * R.f.f.coeff) .* dcdψouter, (Ne*Nψ*Nψ))
    end

    return 0.5*xk .* reshape(quadgk!(integrand!, cache, -1, 1)[1], (Ne, Nψ, Nψ))
end




function grad_coeff(R::IntegratedFunction{m, Nψ, Nx}, X::Array{Float64,2}) where {m, Nψ, Nx}
    NxX, Ne = size(X)
    ψoff = evaluate_offdiagbasis(R.f, X)
    ψdiag = repeated_evaluate_basis(R.f.f, zeros(Ne))

    xk = deepcopy(X[Nx, :])
    dcdψ = zeros(Ne, Nψ)

    cache = zeros(Ne, Nψ)

    function integrand!(v::Matrix{Float64}, t::Float64)
        dcdψ .= repeated_grad_xk_basis(R.f.f,  0.5*(t+1)*xk) .* ψoff
        v .= grad_x(R.g, dcdψ*R.f.f.coeff) .* dcdψ
    end

    return ψoff .* ψdiag + 0.5*xk .* quadgk!(integrand!, cache, -1, 1)[1]
end

hess_coeff(R::IntegratedFunction{m, Nψ, Nx}, X::Array{Float64,2}) where {m, Nψ, Nx} = hess_coeff_integrate_xd(R, X)
