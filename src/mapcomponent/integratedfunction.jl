

export  IntegratedFunction,
        grad_xd,
        grad_coeff_grad_xd,
        hess_coeff_grad_xd,
        repeated_grad_xk_basis!,
        repeated_grad_xk_basis,
        integrate_xd,
        evaluate!,
        evaluate,
        grad_coeff_integrate_xd,
        hess_coeff_integrate_xd,
        grad_coeff,
        hess_coeff,
        evalgrad_coeff!


struct IntegratedFunction
    m::Int64
    Nψ::Int64
    Nx::Int64
    g::Rectifier
    f::ParametricFunction
end

function IntegratedFunction(f::ParametricFunction)
    return IntegratedFunction(f.f.m, f.f.Nψ, f.f.Nx, Rectifier("softplus"), f)
end

function IntegratedFunction(f::ExpandedFunction)
    return IntegratedFunction(f.m, f.Nψ, f.Nx, Rectifier("softplus"), ParametricFunction(f))
end

function integrate_xd(R::IntegratedFunction, X::Array{Float64,2})
    NxX, Ne = size(X)
    ψoff = evaluate_offdiagbasis(R.f, X)
    Nx = R.Nx
    xk = deepcopy(X[Nx, :])
    cache = zeros(Ne)
    # ψoffdxdψ = zeros(Ne, Nψ)

    function integrand!(v::Vector{Float64}, t::Float64)
        # ψoffdxdψ .= repeated_grad_xk_basis(R.f.f,  0.5*(t+1)*xk)
        # ψoffdxdψ .*= ψoff
        # map!(R.g, v, ψoffdxdψ*R.f.f.coeff)
        evaluate!(v, R.g, (repeated_grad_xk_basis(R.f.f,  t*xk) .* ψoff)*R.f.f.coeff)
    end

    return xk .* quadgk!(integrand!, cache, 0.0, 1.0)[1]
end

# Compute g(∂ₖf(x_{1:k}))
function grad_xd(R::IntegratedFunction, X::Array{Float64,2})
    dψ = grad_xd(R.f, X)
    evaluate!(dψ, R.g, dψ)
    # @show size(dψ)
    # gdψ = R.g.(dψ)
    return dψ
end

# Compute ∂_c( g(∂ₖf(x_{1:k}) ) ) = ∂_c∂ₖf(x_{1:k}) × g′(∂ₖf(x_{1:k}))
function grad_coeff_grad_xd(R::IntegratedFunction, X::Array{Float64,2})
    dψ = grad_xd(R.f, X)
    dcdψ = grad_coeff_grad_xd(R.f, X)
    return vgrad_x(R.g, dψ) .* dcdψ
end

# Compute ∂²_c( g(∂ₖf(x_{1:k}) ) ) = ∂²_c∂ₖf(x_{1:k}) × g′(∂ₖf(x_{1:k})) + ∂_c∂ₖf(x_{1:k}) × ∂_c∂ₖf(x_{1:k}) g″(∂ₖf(x_{1:k}))
function hess_coeff_grad_xd(R::IntegratedFunction, X::Array{Float64,2})
    # The second term can be dropped for improve performance
    Nψ = R.Nψ
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
    return vhess_x(R.g, dψ) .* dcdψouter # + dψ .* d2cdψ
end


## integrate_xd
# Compute ∫_0^{x_k} g(∂ₖf(x_{1:k-1},t)) dt


# function repeated_grad_xk_basis(f::ExpandedFunction, x, idx::Array{Int64,2})
#     # Compute the k=th order deriviative of an expanded function along the direction grad_dim
#     N = size(x,1)
#     Nx = f.Nx
#     # ∂ᵏf/∂x_{grad_dim} = ψ
#     k = 1
#     grad_dim = Nx
#     dims = Nx
#
#     midxj = idx[:,Nx]
#     maxj = maximum(midxj)
#     #   Compute the kth derivative along grad_dim
#     dkψj = vander(f.B.B, maxj, k, x)
#     return dkψj[:, midxj .+ 1]
# end


function repeated_grad_xk_basis!(out, cache, f::ExpandedFunction, x, idx::Array{Int64,2})
    # Compute the k=th order deriviative of an expanded function along the direction grad_dim
    Ne = size(x,1)
    Nx = f.Nx

    # @assert size(out,1) = (N, size(idx, 1)) "Wrong dimension of the output vector"
    # ∂ᵏf/∂x_{grad_dim} = ψ
    k = 1
    grad_dim = Nx
    dims = Nx

    midxj = idx[:, Nx]
    maxj = maximum(midxj)
    #   Compute the kth derivative along grad_dim
    # dkψj = zeros(Ne, maxj+1)
    vander!(cache, f.B.B, maxj, k, x)
    Nψreduced = size(idx, 1)
    @avx for l = 1:Nψreduced
        for k=1:Ne
            out[k,l] = cache[k, midxj[l] + 1]
        end
    end

    return out#dkψj[:, midxj .+ 1]
end

repeated_grad_xk_basis!(out, cache, f::ExpandedFunction, x) = repeated_grad_xk_basis!(out, cache, f, x, f.idx)

# repeated_grad_xk_basis(f::ExpandedFunction{m, Nψ, Nx}, x::Array{Float64,1}) where {m, Nψ, Nx} =
repeated_grad_xk_basis(f::ExpandedFunction, x, idx::Array{Int64,2}) =
    repeated_grad_xk_basis!(zeros(size(x,1),size(idx,1)), zeros(size(x,1), maximum(idx[:,f.Nx])+1), f, x, idx)

repeated_grad_xk_basis(f::ExpandedFunction, x) = repeated_grad_xk_basis(f, x, f.idx)

# function evaluate!(out::Array{Float64,1}, R::IntegratedFunction{m, Nψ, Nx}, X::Array{Float64,2}) where {m, Nψ, Nx}

function evaluate!(out, R::IntegratedFunction, X)
    NxX, Ne = size(X)
    Nx = R.Nx
    @assert NxX == Nx "Wrong dimension of the sample X"
    ψoff = evaluate_offdiagbasis(R.f, X)
    ψdiag = repeated_evaluate_basis(R.f.f, zeros(Ne))
    xk = deepcopy(X[Nx, :])
    cache = zeros(Ne)

    @assert size(out,1) == Ne

    function integrand!(v::Vector{Float64}, t::Float64)
        v .= (repeated_grad_xk_basis(R.f.f,  t*xk) .* ψoff) * R.f.f.coeff
        evaluate!(v, R.g, v)

        # v .= R.g((repeated_grad_xk_basis(R.f.f,  t*xk) .* ψoff)*R.f.f.coeff)
    end

     out .= (ψoff .* ψdiag)*R.f.f.coeff + xk .* quadgk!(integrand!, cache, 0.0, 1.0)[1]

     return out
 end



evaluate(R::IntegratedFunction, X::Array{Float64,2}) = evaluate!(zeros(size(X,2)), R, X)


## Compute ∂_c int_0^{x_k} g(∂ₖf(x_{1:k-1}, t))dt
function grad_coeff_integrate_xd(R::IntegratedFunction, X::Array{Float64,2})
    ψoff = evaluate_offdiagbasis(R.f, X)
    Nx = R.Nx
    Nψ = R.Nψ
    NxX, Ne = size(X)
    @assert NxX == Nx "Wrong dimension of the sample X"
    xk = deepcopy(X[Nx, :])
    dcdψ = zeros(Ne, Nψ)

    cache = zeros(Ne, Nψ)

    function integrand!(v::Matrix{Float64}, t::Float64)
        dcdψ .= repeated_grad_xk_basis(R.f.f,  t*xk) .* ψoff

        v .= vgrad_x(R.g, dcdψ*R.f.f.coeff) .* dcdψ
    end

    return xk .* quadgk!(integrand!, cache, 0.0, 1.0)[1]
end

# Compute ∂²_c int_0^{x_k} g(c^T F(t))dt = int_0^{x_k} ∂_c F(t) g′(c^T F(t)) + F(t)F(t)*g″(c^T F(t)) dt
function hess_coeff_integrate_xd(R::IntegratedFunction, X::Array{Float64,2})
     # We drop the first term for speed improvement since it is always equal to 0
    Nx = R.Nx
    Nψ = R.Nψ
    NxX, Ne = size(X)
    @assert NxX == Nx "Wrong dimension of the sample X"
    ψoff = evaluate_offdiagbasis(R.f, X)
    xk = deepcopy(X[Nx, :])
    dcdψ = zeros(Ne, Nψ)

    dcdψouter = zeros(Ne, Nψ, Nψ)

    cache = zeros(Ne*Nψ*Nψ)

    function integrand!(v::Vector{Float64}, t::Float64)
        dcdψ .= repeated_grad_xk_basis(R.f.f,  t*xk) .* ψoff
        @inbounds for i=1:Nψ
            for j=1:Nψ
                dcdψouter[:,i,j] = dcdψ[:,i] .* dcdψ[:, j]
            end
        end
        v .= reshape(vhess_x(R.g, (dcdψ ) * R.f.f.coeff) .* dcdψouter, (Ne*Nψ*Nψ))
    end

    return xk .* reshape(quadgk!(integrand!, cache, 0.0, 1.0)[1], (Ne, Nψ, Nψ))
end




function grad_coeff(R::IntegratedFunction, X::Array{Float64,2})
    Nx = R.Nx
    Nψ = R.Nψ
    NxX, Ne = size(X)
    @assert NxX == Nx "Wrong dimension of the sample X"

    ψoff = evaluate_offdiagbasis(R.f, X)
    ψdiag = repeated_evaluate_basis(R.f.f, zeros(Ne))

    xk = deepcopy(X[Nx, :])
    dcdψ = zeros(Ne, Nψ)

    cache = zeros(Ne, Nψ)

    function integrand!(v::Matrix{Float64}, t::Float64)
        dcdψ .= repeated_grad_xk_basis(R.f.f,  t*xk) .* ψoff
        v .= vgrad_x(R.g, dcdψ*R.f.f.coeff) .* dcdψ
    end

    return ψoff .* ψdiag + xk .* quadgk!(integrand!, cache, 0.0, 1.0)[1]
end

hess_coeff(R::IntegratedFunction, X::Array{Float64,2}) = hess_coeff_integrate_xd(R, X)
