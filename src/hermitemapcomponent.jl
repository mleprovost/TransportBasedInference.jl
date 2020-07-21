export  HermiteMapk,
        getcoeff,
        setcoeff!,
        getidx,
        # inverse!,
        # inverse,
        negative_log_likelihood!,
        negative_log_likelihood
        # optimize


struct HermiteMapk{m, Nψ, k}
    # IntegratedFunction
    I::IntegratedFunction{m, Nψ, k}
    # Regularization parameter
    α::Float64

    function HermiteMapk(I::IntegratedFunction{m, Nψ, k}; α::Float64 = 1e-6) where {m, Nψ, k}
        return new{m, Nψ, k}(I, α)
    end
end

function HermiteMapk(m::Int64, k::Int64, idx::Array{Int64,2}, coeff::Array{Float64,1}; α::Float64 = 1e-6)
    Nψ = size(coeff,1)
    @assert size(coeff,1) == size(idx,1) "Wrong dimension"
    B = MultiBasis(BasisProHermite(m--; scaled =true), k)

    return HermiteMapk(IntegratedFunction(ExpandedFunction(B, idx, coeff)); α = α)
end

function HermiteMapk(f::ExpandedFunction{m, Nψ, k}; α::Float64 = 1e-6) where {m, Nψ, k}
    return HermiteMapk(IntegratedFunction(f); α = α)
end



function HermiteMapk(m::Int64, k::Int64; α::Float64 = 1e-6)
    Nψ = 1

    # m is the dimension of the basis
    B = MultiBasis(BasisProHermite(m-1; scaled =true), k)
    idx = zeros(Int64, Nψ,k)
    coeff = zeros(Nψ)

    f = ExpandedFunction(B, idx, coeff)
    I = IntegratedFunction(f)
    return HermiteMapk(I; α = α)
end

ncoeff(Hk::HermiteMapk{m, Nψ, k}) where {m, Nψ, k} = Nψ
getcoeff(Hk::HermiteMapk{m, Nψ, k}) where {m, Nψ, k} = Hk.I.f.f.coeff

function setcoeff!(Hk::HermiteMapk{m, Nψ, k}, coeff::Array{Float64,1}) where {m, Nψ, k}
        @assert size(coeff,1) == Nψ "Wrong dimension of coeff"
        Hk.I.f.f.coeff .= coeff
end

getidx(Hk::HermiteMapk{m, Nψ, k}) where {m, Nψ, k} = Hk.I.f.f.idx


## Evaluate
function evaluate!(out::Array{Float64,1}, Hk::HermiteMapk{m, Nψ, k}, X::Array{Float64,2}) where {m, Nψ, k}
    @assert k==size(X,1) "Wrong dimension of the sample"
    @assert size(out,1) == size(X,2) "Dimensions of the output and the samples don't match"
    return evaluate!(out, Hk.I, X)
end

evaluate(out::Array{Float64,1}, Hk::HermiteMapk{m, Nψ, k}, X::Array{Float64,2}) where {m, Nψ, k} =
    evaluate!(zeros(size(X,2)), Hk, X)


## Invert transport map

# function inverse!(out::Array{Float64,1}, Hk::HermiteMapk{m, Nψ, k}, X::Array{Float64,2}) where {m, Nψ, k}
#     @assert k==size(X,1) "Wrong dimension of the sample"
#     @assert size(out,1) == size(X,2) "Dimensions of the output and the samples don't match"
#     return inverse!(out, Hk.I, X)
# end
#
# inverse(out::Array{Float64,1}, Hk::HermiteMapk{m, Nψ, k}, X::Array{Float64,2}) where {m, Nψ, k} =
#         inverse!(zeros(size(X,2)), Hk, X)



## Compute Negative log likelihood and its gradient
function negative_log_likelihood!(J, dJ, coeff, S::Storage{m, Nψ, k}, Hk::HermiteMapk{m, Nψ, k}, X::Array{Float64,2}) where {T <: Real, m, Nψ, k}
    NxX, Ne = size(X)
    @assert NxX == k "Wrong dimension of the sample X"
    @assert size(S.ψoff, 1) == Ne
    @assert size(S.ψoff, 2) == Nψ

    # Output objective, gradient
    xk = deepcopy(X[NxX,:])#)

    fill!(S.cache_integral, 0)

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

    quadgk!(integrand!, S.cache_integral, 0, 1; rtol = 1e-3)#; order = 9, rtol = 1e-10)

    # Multiply integral by xk (change of variable in the integration)
    @inbounds for j=1:Nψ+1
        @. S.cache_integral[(j-1)*Ne+1:j*Ne] *= xk
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
        dJ .= zeros(Nψ)
        @inbounds for i=1:Ne
            # dJ .= zeros(Nψ)
            for j=1:Nψ
            dJ[j] += gradlogpdf(Normal(), S.cache_integral[i])*(reshape_cacheintegral[i,j] + S.ψoff[i,j]*S.ψd0[i,j])
            dJ[j] += grad_x(Hk.I.g, S.cache_g[i])*S.ψoff[i,j]*S.dψxd[i,j]/Hk.I.g(S.cache_g[i])
            end
            # @show i, dJ
        end
        rmul!(dJ, -1/Ne)
        # Add derivative of the L2 penalty term ∂_c α ||c||^2 = 2 *α c
        dJ .+= 2*Hk.α*coeff
    end

    if J != nothing
        J = 0.0
        @inbounds for i=1:Ne
            J += logpdf(Normal(), S.cache_integral[i]) + log(Hk.I.g(S.cache_g[i]))
        end
        J *=(-1/Ne)
        J += Hk.α*norm(coeff)^2
        return J
    end
end

negative_log_likelihood!(S::Storage{m, Nψ, k}, Hk::HermiteMapk{m, Nψ, k}, X::Array{Float64,2}) where {T <: Real, m, Nψ, k} =
    (J, dJ, coeff) -> negative_log_likelihood!(J, dJ, coeff, S, Hk, X)




# function optimize()




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
end
