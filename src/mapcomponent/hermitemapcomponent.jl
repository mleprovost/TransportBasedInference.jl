export  MapComponent,
        EmptyMapComponent,
        ncoeff,
        getcoeff,
        setcoeff!,
        getidx,
        active_dim,
        evaluate!,
        evaluate,
        log_pdf!,
        log_pdf,
        grad_x_log_pdf!,
        grad_x_log_pdf,
        hess_x_log_pdf!,
        hess_x_log_pdf,
        reduced_hess_x_log_pdf!,
        reduced_hess_x_log_pdf,
        mean_hess_x_log_pdf!,
        mean_hess_x_log_pdf,
        negative_log_likelihood!,
        negative_log_likelihood,
        hess_negative_log_likelihood!


struct MapComponent
    m::Int64
    Nψ::Int64
    Nx::Int64
    # IntegratedFunction
    I::IntegratedFunction
    # Regularization parameter
    α::Float64
end

function MapComponent(I::IntegratedFunction; α::Float64=1e-6)
    return MapComponent(I.m, I.Nψ, I.Nx, I, α)
end

function MapComponent(m::Int64, Nx::Int64, idx::Array{Int64,2}, coeff::Array{Float64,1}; α::Float64 = 1e-6)
    Nψ = size(coeff,1)
    @assert size(coeff,1) == size(idx,1) "Wrong dimension"
    B = MultiBasis(CstProHermite(m-2), Nx)

    return MapComponent(m, Nψ, Nx, IntegratedFunction(ExpandedFunction(B, idx, coeff)), α)
end

function MapComponent(f::ExpandedFunction; α::Float64 = 1e-6)
    return MapComponent(f.m, f.Nψ, f.Nx, IntegratedFunction(f), α)
end

function MapComponent(m::Int64, Nx::Int64; α::Float64 = 1e-6)
    Nψ = 1

    # m is the dimension of the basis
    B = MultiBasis(CstProHermite(m-2), Nx)
    idx = zeros(Int64, Nψ, Nx)
    coeff = zeros(Nψ)

    f = ExpandedFunction(B, idx, coeff)
    I = IntegratedFunction(f)
    return MapComponent(I; α = α)
end

ncoeff(C::MapComponent) = C.Nψ
getcoeff(C::MapComponent)= C.I.f.f.coeff

function setcoeff!(C::MapComponent, coeff::Array{Float64,1})
        @assert size(coeff,1) == C.Nψ "Wrong dimension of coeff"
        C.I.f.f.coeff .= coeff
end

getidx(C::MapComponent) = C.I.f.f.idx

active_dim(C::MapComponent) = C.I.f.f.dim

## Evaluate
function evaluate!(out, C::MapComponent, X)
    @assert C.Nx==size(X,1) "Wrong dimension of the sample"
    @assert size(out,1) == size(X,2) "Dimensions of the output and the samples don't match"
    return evaluate!(out, C.I, X)
end

evaluate(C::MapComponent, X::Array{Float64,2}) =
    evaluate!(zeros(size(X,2)), C, X)

## Compute log_pdf

function log_pdf!(result, cache, C::MapComponent, X)
    NxX, Ne = size(X)
    @assert C.Nx == NxX "Wrong dimension of the sample"
    @assert size(result, 1) == Ne "Wrong dimension of the sample"
    @assert size(cache, 1) == Ne "Wrong dimension of the sample"

    evaluate!(result, C, X)
    cache .= grad_xd(C.I, X)

    @avx for i=1:size(X,2)
        result[i] = log_pdf(result[i]) + log(cache[i])
    end

    return result
end

log_pdf(C::MapComponent, X) = log_pdf!(zeros(size(X,2)), zeros(size(X,2)), C, X)

## Compute grad_x_log_pdf

function grad_x_log_pdf!(result, cache, C::MapComponent, X)
    NxX, Ne = size(X)
    @assert C.Nx == NxX "Wrong dimension of the sample"
    @assert size(result) == (Ne, NxX) "Wrong dimension of the result"

    # Compute gradient of log η∘C(x_{1:k})
    evaluate!(cache, C, X)
    grad_x!(result, C.I, X)
    @avx @. result *= cache
    rmul!(result, -1.0)

    # Compute gradient of log ∂k C(x_{1:k})
    cache .= grad_xd(C.I.f, X)
    grad_x_logeval!(cache, C.I.g, cache)
    result += cache .* grad_x_grad_xd(C.I.f.f, X)
    return result
end

grad_x_log_pdf(C::MapComponent, X) = grad_x_log_pdf!(zeros(size(X,2), size(X,1)), zeros(size(X,2)), C, X)

## Compute hess_x_log_pdf

function hess_x_log_pdf!(result, dcache, cache, C::MapComponent, X)
    NxX, Ne = size(X)
    Nx = C.Nx
    @assert Nx == NxX "Wrong dimension of the sample"
    @assert size(result) == (Ne, NxX, NxX) "Wrong dimension of the result"

    # Compute hessian of log η∘C(x_{1:k}) with η the log pdf of N(O, I_n)
    evaluate!(cache, C, X)
    grad_x!(dcache, C.I, X)
    hess_x!(result, C.I, X)

    dim = active_dim(C)

    @inbounds for i=1:length(dim)
                for j=i:length(dim)
                dcachei = view(dcache,:,dim[i])
                dcachej = view(dcache,:,dim[j])
                resultij = view(result,:,dim[i],dim[j])
                resultji = view(result,:,dim[j], dim[i])
                @avx @. resultij = resultij * cache + dcachei * dcachej
                resultji .= resultij
        end
    end

    rmul!(result, -1.0)

    # Compute hessian of log ∂k C(x_{1:k})
    cache .= grad_xd(C.I.f, X)
    cached2log = vhess_x_logeval(C.I.g, cache)
    dcache .= grad_x_grad_xd(C.I.f.f, X)

    @inbounds for i=1:length(dim)
                for j=i:length(dim)
                dcachei = view(dcache,:,dim[i])
                dcachej = view(dcache,:,dim[j])
                resultij = view(result,:,dim[i],dim[j])
                resultji = view(result,:,dim[j], dim[i])
                @avx @. resultij += dcachei * dcachej * cached2log

                resultji .= resultij
        end
    end
    #
    grad_x_logeval!(cache, C.I.g, cache)
    result .+= hess_x_grad_xd(C.I.f.f, X) .* cache

    return result
end

hess_x_log_pdf(C::MapComponent, X) = hess_x_log_pdf!(zeros(size(X,2), size(X,1), size(X,1)),
                                                     zeros(size(X,2), size(X,1)),
                                                     zeros(size(X,2)), C, X)



# This version outputs a result of size(Ne, active_dim(C), active_dim(C))


function reduced_hess_x_log_pdf!(result, dcache, cache, C::MapComponent, X)
    NxX, Ne = size(X)
    Nx = C.Nx

    dim = active_dim(C)
    dimoff = dim[dim .< Nx]


    @assert Nx == NxX "Wrong dimension of the sample"
    @assert size(result) == (Ne, length(dim), length(dim)) "Wrong dimension of the result"

    # Compute hessian of log η∘C(x_{1:k}) with η the log pdf of N(O, I_n)
    evaluate!(cache, C, X)
    reduced_grad_x!(dcache, C.I, X)
    reduced_hess_x!(result, C.I, X)

    dim = active_dim(C)

    @inbounds for i=1:length(dim)
        for j=i:length(dim)
             # dcachei = view(dcache,:,dim[i])
             # dcachej = view(dcache,:,dim[j])
             # resultij = view(result,:,dim[i],dim[j])
             # resultji = view(result,:,dim[j], dim[i])
             dcachei = view(dcache,:,i)
             dcachej = view(dcache,:,j)
             resultij = view(result,:,i,j)
             resultji = view(result,:,j,i)
             @. resultij = resultij * cache + dcachei * dcachej
             resultji .= resultij
         end
    end

    rmul!(result, -1.0)

    # Compute hessian of log ∂k C(x_{1:k})
    cache .= grad_xd(C.I.f, X)
    cached2log = vhess_x_logeval(C.I.g, cache)

    if Nx == dim[end] # check if the last component is an active dimension
        # dcache .= grad_x_grad_xd(C.I.f.f, X)
        # Clear dcache
        fill!(dcache, 0.0)
        reduced_grad_x_grad_xd!(dcache, C.I.f.f, X)
        @inbounds for i=1:length(dim)
             for j=i:length(dim)
                 # dcachei = view(dcache,:,dim[i])
                 # dcachej = view(dcache,:,dim[j])
                 # resultij = view(result,:,dim[i],dim[j])
                 # resultji = view(result,:,dim[j], dim[i])
                 dcachei = view(dcache,:,i)
                 dcachej = view(dcache,:,j)
                 resultij = view(result,:,i,j)
                 resultji = view(result,:,j, i)
                 @avx @. resultij += dcachei * dcachej * cached2log

                 resultji .= resultij
             end
         end
    end
    grad_x_logeval!(cache, C.I.g, cache)
    result .+= reduced_hess_x_grad_xd(C.I.f.f, X) .* cache

    return result
end


reduced_hess_x_log_pdf(C::MapComponent, X) = reduced_hess_x_log_pdf!(zeros(size(X,2), length(active_dim(C)), length(active_dim(C))),
                                                zeros(size(X,2), length(active_dim(C))), zeros(size(X,2)), C, X)



## negative_log_likelihood

# function negative_log_likelihood!(J, dJ, coeff, S::Storage{m, Nψ, k}, C::MapComponent{m, Nψ, k}, X::Array{Float64,2}) where {m, Nψ, k}

function negative_log_likelihood!(J, dJ, coeff, S::Storage, C::MapComponent, X)
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
        # The reshape version is faster than unrolling
        v[Ne+1:Ne+Ne*Nψ] .= reshape(S.cache_dψxd .* S.cache_dcψxdt , (Ne*Nψ))
        # v[Ne+1:Ne+Ne*Nψ] .= reshape(grad_x(C.I.g, S.cache_dψxd) .* S.cache_dcψxdt , (Ne*Nψ))
    end

    quadgk!(integrand!, S.cache_integral, 0.0, 1.0; rtol = 1e-3)#, order = 3)#; order = 9, rtol = 1e-10)

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
            f0i += S.ψoffψd0[i,j]*coeff[j]
        end
        S.cache_integral[i] += f0i
    end

    # Store g(∂_{xk}f(x_{1:k})) in S.cache_g

    @avx for i=1:Ne
        prelogJi = zero(Float64)
        for j=1:Nψ
            prelogJi += S.ψoffdψxd[i,j]*coeff[j]
        end
        S.cache_g[i] = prelogJi
    end


    # Formatting to use with Optim.jl
    if dJ != nothing
        reshape_cacheintegral = reshape(S.cache_integral[Ne+1:end], (Ne, Nψ))
        fill!(dJ, 0.0)
        @inbounds for i=1:Ne
            for j=1:Nψ
            dJ[j] += gradlog_pdf(S.cache_integral[i])*(reshape_cacheintegral[i,j] + S.ψoffψd0[i,j]) + # dsoftplus(S.cache_g[i])*S.ψoff[i,j]*S.dψxd[i,j]*(1/softplus(S.cache_g[i]))
                     grad_x(C.I.g, S.cache_g[i])*S.ψoffdψxd[i,j]/C.I.g(S.cache_g[i])
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

negative_log_likelihood(S::Storage, C::MapComponent, X) = (J, dJ, coeff) -> negative_log_likelihood!(J, dJ, coeff, S, C, X)


function hess_negative_log_likelihood!(J, dJ, d2J, coeff, S::Storage, C::MapComponent, X::Array{Float64,2})
    Nψ = C.Nψ
    Nx = C.Nx

    NxX, Ne = size(X)
    @assert NxX == Nx "Wrong dimension of the sample X"
    @assert size(S.ψoff, 1) == Ne
    @assert size(S.ψoff, 2) == Nψ

    # Output objective, gradient
    xlast = view(X, NxX, :)#)

    fill!(S.cache_integral, 0)

    dcdψouter = zeros(Ne, Nψ, Nψ)

    # Integrate at the same time for the objective, gradient
    function integrand!(v::Vector{Float64}, t::Float64)
        S.cache_dcψxdt .= repeated_grad_xk_basis(C.I.f.f, t*xlast)

        # @avx @. S.cache_dψxd = (S.cache_dcψxdt .* S.ψoff) *ˡ coeff
        S.cache_dcψxdt .*= S.ψoff
        mul!(S.cache_dψxd, S.cache_dcψxdt, coeff)

        @inbounds for i=1:Nψ
            for j=1:Nψ
                dcdψouter[:,i,j] = S.cache_dcψxdt[:,i] .* S.cache_dcψxdt[:, j]
            end
        end

        # Integration for J
        vJ = view(v,1:Ne)
        evaluate!(vJ, C.I.g, S.cache_dψxd)
        # v[1:Ne] .= C.I.g(S.cache_dψxd)

        # Integration for dcJ
        v[Ne+1:Ne+Ne*Nψ] .= reshape(grad_x(C.I.g, S.cache_dψxd) .* S.cache_dcψxdt , (Ne*Nψ))

        # Integration for d2cJ
        v[Ne + Ne*Nψ + 1: Ne + Ne*Nψ + Ne*Nψ*Nψ] .= reshape(hess_x(C.I.g, S.cache_dψxd) .* dcdψouter, (Ne*Nψ*Nψ))

    end

    quadgk!(integrand!, S.cache_integral, 0.0, 1.0; rtol = 1e-3)#; order = 9, rtol = 1e-10)

    # Multiply integral by xlast (change of variable in the integration)
    @inbounds for j=1:1 + Nψ# + Nψ*Nψ
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
        reshape_cacheintegral = reshape(S.cache_integral[Ne+1:Ne+Ne*Nψ], (Ne, Nψ))
        fil!(dJ, 0.0)#dJ .= zeros(Nψ)
        @inbounds for i=1:Ne
            # dJ .= zeros(Nψ)
            for j=1:Nψ
            dJ[j] += gradlog_pdf(S.cache_integral[i])*(reshape_cacheintegral[i,j] + S.ψoff[i,j]*S.ψd0[i,j])
            dJ[j] += grad_x(C.I.g, S.cache_g[i])*S.ψoff[i,j]*S.dψxd[i,j]/C.I.g(S.cache_g[i])
            end
            # @show i, dJ
        end
        rmul!(dJ, -1/Ne)
        # Add derivative of the L2 penalty term ∂_c α ||c||^2 = 2 *α c
        dJ .+= 2*C.α*coeff
    end

    if d2J != nothing
        reshape_cacheintegral = reshape(S.cache_integral[Ne+1:Ne+Ne*Nψ], (Ne, Nψ))
        reshape2_cacheintegral = reshape(S.cache_integral[Ne + Ne*Nψ + 1: Ne + Ne*Nψ + Ne*Nψ*Nψ], (Ne, Nψ, Nψ))
        # @show reshape2_cacheintegral
        fill!(d2J, 0.0)
        # d2J .= zeros(Nψ, Nψ)
        @inbounds for l=1:Ne
            # Exploit symmetry of the Hessian
            for j=1:Nψ
                for i=j:Nψ
                d2J[i,j] +=  reshape2_cacheintegral[l,i,j]*S.cache_integral[l]
                d2J[i,j] +=  (reshape_cacheintegral[l,i] + S.ψoff[l,i]*S.ψd0[l,i]) * (reshape_cacheintegral[l,j] + S.ψoff[l,j]*S.ψd0[l,j])
                d2J[i,j] -=  ( (S.ψoff[l,i]*S.dψxd[l,i]) * (S.ψoff[l,j]*S.dψxd[l,j])*(
                                hess_x(C.I.g, S.cache_g[l]) * C.I.g(S.cache_g[l]) -
                                grad_x(C.I.g, S.cache_g[l])^2))/C.I.g(S.cache_g[l])^2

                d2J[j,i] = d2J[i,j]
                end
            end
        end
        rmul!(d2J, 1/Ne)
        # Add derivative of the L2 penalty term ∂^2_c α ||c||^2 = 2 *α *I
        @inbounds for i=1:Nψ
            d2J[i,i] += 2*C.α*I
        end
        # d2J = Symmetric(d2J)
        # return d2J
    end

    if J != nothing
        J = 0.0
        @inbounds for i=1:Ne
            J += log_pdf(S.cache_integral[i]) + log(C.I.g(S.cache_g[i]))
        end
        J *=(-1/Ne)
        J += C.α*norm(coeff)^2
        return J
    end
    # return J, dJ, d2J
end


hess_negative_log_likelihood!(S::Storage, C::MapComponent, X::Array{Float64,2}) =
    (J, dJ, d2J, coeff) -> hess_negative_log_likelihood!(J, dJ, d2J, coeff, S, C, X)
