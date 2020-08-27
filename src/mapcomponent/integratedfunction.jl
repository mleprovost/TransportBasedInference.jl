

export  IntegratedFunction,
        grad_xd,
        grad_coeff_grad_xd,
        hess_coeff_grad_xd,
        integrate_xd,
        evaluate!,
        evaluate,
        grad_x!,
        grad_x,
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
function grad_xd(R::IntegratedFunction, X)
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


# function evaluate!(out::Array{Float64,1}, R::IntegratedFunction{m, Nψ, Nx}, X::Array{Float64,2}) where {m, Nψ, Nx}

function evaluate!(out, R::IntegratedFunction, X)
    NxX, Ne = size(X)
    Nx = R.Nx
    @assert NxX == Nx "Wrong dimension of the sample X"
    ψoff = evaluate_offdiagbasis(R.f, X)
    ψdiag = repeated_evaluate_basis(R.f.f, zeros(Ne))
    xlast = view(X,NxX,:)
    cache = zeros(Ne)

    @assert size(out,1) == Ne

    function integrand!(v::Vector{Float64}, t::Float64)
        v .= (repeated_grad_xk_basis(R.f.f,  t*xlast) .* ψoff) * R.f.f.coeff
        evaluate!(v, R.g, v)

        # v .= R.g((repeated_grad_xk_basis(R.f.f,  t*xk) .* ψoff)*R.f.f.coeff)
    end

     out .= (ψoff .* ψdiag)*R.f.f.coeff + xlast .* quadgk!(integrand!, cache, 0.0, 1.0)[1]

     return out
 end


evaluate(R::IntegratedFunction, X::Array{Float64,2}) = evaluate!(zeros(size(X,2)), R, X)
evaluate(R::IntegratedFunction, x::Array{Float64,1}) = evaluate!(zeros(1), R, reshape(x,(size(x,1), 1)))[1]


## Gradient of an IntegratedFunction

function grad_x!(out, R::IntegratedFunction, X)
    NxX, Ne = size(X)
    Nx = R.Nx
    @assert NxX == Nx "Wrong dimension of the sample"
    @assert size(out) == (Ne, Nx) "Dimensions of the output and the samples don't match"

    x0 = zeros(Ne)
    xlast = view(X,Nx,:)
    ψk0  = repeated_evaluate_basis(R.f.f, x0)
    dxkψk0 = repeated_grad_xk_basis(R.f.f, x0)
    ψoff = evaluate_offdiagbasis(R.f, X)
    dxkψ = zero(ψk0)

    # Cache for the integration
    coeff = R.f.f.coeff
    cache = zeros((Nx-1)*Ne)
    cacheg = zeros(Ne)

    # Compute the basis for each component
    ψbasis = zeros(Ne, R.Nψ, Nx-1)
    @inbounds for i=1:Nx-1
        ψbasis_i = view(ψbasis, :, :, i)
        ψbasis_i .= evaluate_basis(R.f.f, X, [i], R.f.f.idx)
    end

    # Compute ψ1 ⊗ ψ2 ⊗ ψi′ ⊗ … ⊗ ψk-1
    dxψbasis = zero(ψbasis)
    fill!(dxψbasis, 1.0)
    @inbounds for i=1:Nx-1
        for j=1:Nx-1
            if i==j
            dxψbasis[:,:,i] .*= grad_xk_basis(R.f.f, X, 1, j, j, R.f.f.idx)
            else
            dxψbasis[:,:,i] .*= ψbasis[:,:,j]
            end
        end
    end


    # This integrand computes ψ1 ⊗ … ⊗ ψi′ ⊗ … ⊗ ψk′(t) × c g′(ψ1 ⊗ … ⊗ ψk′(t)×c)
    function integrand!(v::Vector{Float64}, t::Float64)
    dxkψ .= repeated_grad_xk_basis(R.f.f,  t*xlast)
    @avx @. cacheg = (dxkψ * ψoff) *ˡ coeff
    grad_x!(cacheg, R.g, cacheg)

        @inbounds for i=1:Nx-1
            vi = view(v, (i-1)*Ne+1:i*Ne)
            dxψbasisi = view(dxψbasis,:,:,i)
            vi .= ((dxψbasisi .* dxkψ) * coeff) .* cacheg
        end
    end
    quadgk!(integrand!, cache, 0.0, 1.0; rtol = 1e-3)

    # Multiply integral by xlast (change of variable in the integration)
    @inbounds for i=1:Nx-1
        @. cache[(i-1)*Ne+1:i*Ne] *= xlast
    end

    # Add ψ1 ⊗ … ⊗ ψi′ ⊗ … ⊗ ψk(0) × c
    @inbounds for i=1:Nx-1
        dxψbasisi = view(dxψbasis,:,:,i)
        view(out,:,i) .+= (dxψbasisi .* ψk0)*coeff + view(cache,(i-1)*Ne+1:i*Ne)
    end

    # Fill the last column
    lastcol = view(out, :, Nx)
    evaluate!(lastcol, R.g, (repeated_grad_xk_basis(R.f.f,  xlast) .* ψoff)*coeff)
    return out
end

grad_x(R::IntegratedFunction, X) = grad_x!(zeros(size(X')), R, X)


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
