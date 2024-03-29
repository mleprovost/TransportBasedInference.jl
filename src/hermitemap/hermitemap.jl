export  HermiteMap,
        setcoeff!,
        clearcoeff!,
        evaluate!,
        evaluate,
        log_pdf,
        grad_x_log_pdf!,
        grad_x_log_pdf,
        hess_x_log_pdf!,
        hess_x_log_pdf,
        reduced_hess_x_log_pdf!,
        reduced_hess_x_log_pdf,
        optimize,
        inverse!,
        hybridinverse!

import Base: @propagate_inbounds


"""
$(TYPEDEF)


## Fields

$(TYPEDFIELDS)
"""
struct HermiteMap
        m::Int64
        Nx::Int64
        L::LinearTransform
        C::Array{HermiteMapComponent,1}
end

function HermiteMap(L::LinearTransform, C::Array{HermiteMapComponent,1})
        m = C[end].m
        Nx = C[end].Nx

        return HermiteMap(m, Nx, L, C)
end

"""
$(TYPEDSIGNATURES)

Access the i-th `HermiteMapComponent` of `M`.
"""
@propagate_inbounds Base.getindex(M::HermiteMap, i::Int) = getindex(M.C,i)

"""
$(TYPEDSIGNATURES)

Set the i-th `HermiteMapComponent` of `M`.
"""
@propagate_inbounds Base.setindex!(M::HermiteMap, C::HermiteMapComponent, i::Int) = setindex!(M.C,C,i)

function Base.show(io::IO, M::HermiteMap)
        println(io,"Hermite map of dimension "*string(M.Nx)*":")
        for i=1:M.Nx
                show(io, M.C[i])
        end
end

function HermiteMap(m::Int64, X::Array{Float64,2}; diag::Bool=true, factor::Float64=1.0, α::Float64 = αreg, b::String="CstProHermiteBasis")
        L = deepcopy(LinearTransform(X; diag = diag, factor = factor))

        if b ∈ ["ProHermiteBasis"; "PhyHermiteBasis";
                "CstProHermiteBasis"; "CstPhyHermiteBasis";
                "CstLinProHermiteBasis"; "CstLinPhyHermiteBasis"]
            B = eval(Symbol(b))(m)
        else
            error("The basis "*b*" is not defined.")
        end

        Nx = size(X,1)
        Nψ = 1
        coeff = zeros(Nψ)
        C = HermiteMapComponent[]
        idx = zeros(Int, Nψ, Nx)
        @inbounds for i=1:Nx
                MultiB = MultiBasis(B, i)
                vidx = idx[:,1:i]
                push!(C, HermiteMapComponent(IntegratedFunction(ExpandedFunction(MultiB, vidx, coeff)); α = α))
        end

        return HermiteMap(m, Nx, L, C)
end

"""
$(TYPEDSIGNATURES)

Set all the coefficients of the HermiteMap to zero
"""
function clearcoeff!(M::HermiteMap; start::Int64 = 1)
        for i=start:M.Nx
                clearcoeff!(M.C[i])
        end
end


"""
$(TYPEDSIGNATURES)

Evaluates in-place the HermiteMap `M` for the ensemble matrix `X`.
"""
function evaluate!(out, M::HermiteMap, X; apply_rescaling::Bool=true, start::Int64=1, P::Parallel = serial)

        Nx = M.Nx
        NxX, Ne = size(X)
        @assert NxX == Nx
        @assert size(out, 2) == Ne

        # We can apply the rescaling to all the components once
        if apply_rescaling == true
                transform!(M.L, X)
        end

        if typeof(P) <: Serial
        # We can skip the evaluation of the map on the observations componentd
        @inbounds for k=start:Nx
                Xk = view(X,1:k,:)
                col = view(out,k,:)
                evaluate!(col, M.C[k], Xk)
        end
        elseif typeof(P) <: Thread
        @inbounds Threads.@threads for k=start:Nx
                Xk = view(X,1:k,:)
                col = view(out,k,:)
                evaluate!(col, M.C[k], Xk)
        end
        end

        if apply_rescaling == true
                itransform!(M.L, X)
        end

        return out
end

"""
$(TYPEDSIGNATURES)

Evaluates the HermiteMap `M` for the ensemble matrix `X`.
"""
evaluate(M::HermiteMap, X; apply_rescaling::Bool=true, start::Int64=1, P::Parallel = serial) =
         evaluate!(zero(X), M, X; apply_rescaling = apply_rescaling, start = start, P = P)

"""
$(TYPEDSIGNATURES)

Evaluates the logarithm of the pullback of the reference density (standard normal) by the map `M` along the `component` directions  for the ensemble matrix `X`.
"""
function log_pdf(M::HermiteMap, X, component::Union{Int64, Array{Int64,1}, UnitRange{Int64}};
                 apply_rescaling::Bool = true)

        @assert size(X,1) == M.Nx "Wrong dimension of the input vector"

        # We can apply the rescaling to all the components once
        if apply_rescaling == true
                transform!(M.L, X)
        end

        logπ = zeros(size(X,2))

        if apply_rescaling == true
                # We add a - in front of the logdet
                # since we are interested inthe logdet of the inverse of M.L
                @inbounds for i in component
                        if typeof(M.L.L) <: Diagonal
                                logπ += log_pdf(M.C[i], view(X,1:i,:)) .+ log(1/(M.L.L.diag[i]))
                        elseif typeof(M.L.L) <: LowerTriangular
                                # For a triangular matrix, the inverse matrix has the inverse
                                # coefficients on its diagonal
                                logπ += log_pdf(M.C[i], view(X,1:i,:)) .+ log(1/(M.L.L[i,i]))
                        end
                end

        else
                @inbounds for i in component
                        logπ += log_pdf(M.C[i], view(X,1:i,:)) #.+ log(1/(M.L.L.diag[i]))
                end
        end

        # We can apply the rescaling to all the components once
        if apply_rescaling == true
                itransform!(M.L, X)
        end
        return logπ
end

"""
$(TYPEDSIGNATURES)

Evaluates the logarithm of the pullback of the reference density (standard normal) by the map `M` for the ensemble matrix `X`.
"""
log_pdf(M::HermiteMap, X; apply_rescaling::Bool = true) = log_pdf(M, X, 1:M.Nx; apply_rescaling = apply_rescaling)

## Compute grad_x_log_pdf

"""
$(TYPEDSIGNATURES)

Evaluates in-place the gradient of the logarithm of the pullback of the reference density (standard normal) by the map `M` for the ensemble matrix `X`.
"""
function grad_x_log_pdf!(result, cache_grad, cache, M::HermiteMap, X; apply_rescaling::Bool = true)
        Nx = M.Nx
        NxX, Ne = size(X)
        @assert size(X,1) == Nx "Wrong dimension of the input vector"
        @assert size(result) == (Ne, Nx) "Wrong dimension of the result"
        @assert size(cache_grad) == (Ne, Nx) "Wrong dimension of cache_grad"
        @assert size(cache) == (Ne, ) "Wrong dimension of cache"

        # We can apply the rescaling to all the components once
        if apply_rescaling == true
                transform!(M.L, X)
        end

        # The rescaling doesn't appears in the gradient (nor hessian), log(xy) = log(x) + log(y)
        @inbounds for i=1:Nx
                Xi = view(X,1:i,:)
                resulti = view(result,:,1:i)
                cache_gradi = view(cache_grad,:,1:i)
                grad_x_log_pdf!(cache_gradi, cache, M.C[i], Xi)
                @avx @. resulti += cache_gradi
        end

        if apply_rescaling == true
                itransform!(M.L, X)
        end

        return result
end

"""
$(TYPEDSIGNATURES)

Evaluates the gradient of the logarithm of the pullback of the reference density (standard normal) by the map `M` for the ensemble matrix `X`.
"""
grad_x_log_pdf(M::HermiteMap, X; apply_rescaling::Bool = true) = grad_x_log_pdf!(zeros(size(X,2), size(X,1)), zeros(size(X,2), size(X,1)), zeros(size(X,2)),
                                                                 M, X; apply_rescaling = apply_rescaling)

## Compute hess_x_log_pdf

"""
$(TYPEDSIGNATURES)

Evaluates in-place the hessian of the logarithm of the pullback of the reference density (standard normal) by the map `M` for the ensemble matrix `X`.
"""
function hess_x_log_pdf!(result, M::HermiteMap, X; apply_rescaling::Bool = true)

        Nx = M.Nx
        NxX, Ne = size(X)
        @assert size(X,1) == Nx "Wrong dimension of the input vector"
        @assert size(result) == (Ne, Nx, Nx) "Wrong dimension of the result"
        # @assert size(cache_hess) == (Ne, Nx, Nx) "Wrong dimension of cache_hess"
        # @assert size(cache_grad) == (Ne, Nx) "Wrong dimension of cache_grad"
        # @assert size(cache) == (Ne, ) "Wrong dimension of cache"

        # We can apply the rescaling to all the components once
        if apply_rescaling == true
                transform!(M.L, X)
        end

        # The rescaling doesn't appears in the hessian, log(xy) = log(x) + log(y)
        @inbounds for i=1:Nx
                Xi = view(X,1:i,:)
                resulti = view(result,:,1:i,1:i)
                # cache_gradi = view(cache_grad,:,1:i)
                # cache_hessi = view(cache_hess,:,1:i,1:i)
                # hess_x_log_pdf!(cache_hessi, cache_gradi, cache, M.C[i], Xi)
                # @avx @. resulti += cache_hessi
                resulti .+= hess_x_log_pdf(M.C[i], Xi)

        end

        if apply_rescaling == true
                itransform!(M.L, X)
        end

        return result
end

"""
$(TYPEDSIGNATURES)

Evaluates the hessian of the logarithm of the pullback of the reference density (standard normal) by the map `M` for the ensemble matrix `X`.
"""
hess_x_log_pdf(M::HermiteMap, X; apply_rescaling::Bool = true) = hess_x_log_pdf!(zeros(size(X,2), size(X,1), size(X,1)), M, X; apply_rescaling = apply_rescaling)


# This forms output an array of dimension (Ne, Nx, Nx) but the intermediate computation have been done
# by only looking at the active dimensions

"""
$(TYPEDSIGNATURES)

Evaluates in-place the hessian of the logarithm of the pullback of the reference density (standard normal) by the map `M` for the ensemble matrix `X`.
This routine exploits the sparsity pattern of the map `M`.
"""
function reduced_hess_x_log_pdf!(M::HermiteMap, X; apply_rescaling::Bool = true)

        Nx = M.Nx
        NxX, Ne = size(X)

        # Find the maximal length of active dimensions among the different map components
        Nmax = maximum(i -> length(active_dim(M.C[i])), 1:Nx)

        @assert size(X,1)       ==  Nx              "Wrong dimension of the input vector"
        # @assert length(result)  ==  Ne              "Wrong dimension of the result"
        # @assert size(d2cache)   == (1, Nmax, Nmax)  "Wrong dimension of the d2cache"
        # @assert size(dcache)    == (1, Nmax)        "Wrong dimension of the dcache"
        # @assert size(cache)     == (1, )            "Wrong dimension of the cache"

        Nmax = maximum(i -> length(active_dim(M.C[i])), 1:M.Nx)
        NxX, Ne = size(X)
        Nx = M.Nx

        sparsity_pattern = spzeros(Nx, Nx)
        @inbounds for i=1:Nx
                dimi = active_dim(M.C[i])
                sparsity_pattern[dimi, dimi] .= 0.0
        end

        result = ntuple(x->copy(sparsity_pattern), Ne)


        # We can apply the rescaling to all the components once
        if apply_rescaling == true
                transform!(M.L, X)
        end

        # The rescaling doesn't appears in the hessian, log(xy) = log(x) + log(y)
        @inbounds for j=1:Ne
                for i=1:Nx
                dimi = active_dim(M.C[i])
                Xi = X[1:i,j:j]

                result[j][dimi, dimi] .+= view(reduced_hess_x_log_pdf(M.C[i], Xi),1,:,:)
                end
        end

        if apply_rescaling == true
                itransform!(M.L, X)
        end

        return result
end

"""
$(TYPEDSIGNATURES)

Evaluates in-place the hessian of the logarithm of the pullback of the reference density (standard normal) by the map `M` for the ensemble matrix `X`.
This routine exploits the sparsity pattern of the map `M`.
"""
function reduced_hess_x_log_pdf!(result, d2cache, dcache, cache,M::HermiteMap, X; apply_rescaling::Bool = true)

        Nx = M.Nx
        NxX, Ne = size(X)

        # Find the maximal length of active dimensions among the different map components
        Nmax = maximum(i -> length(active_dim(M.C[i])), 1:Nx)

        @assert size(X,1)       ==  Nx              "Wrong dimension of the input vector"
        @assert length(result)  ==  Ne              "Wrong dimension of the result"
        @assert size(d2cache)   == (1, Nmax, Nmax)  "Wrong dimension of the d2cache"
        @assert size(dcache)    == (1, Nmax)        "Wrong dimension of the dcache"
        @assert size(cache)     == (1, )            "Wrong dimension of the cache"


        # We can apply the rescaling to all the components once
        if apply_rescaling == true
                transform!(M.L, X)
        end

        # The rescaling doesn't appears in the hessian, log(xy) = log(x) + log(y)
        @inbounds for j=1:Ne
                for i=1:Nx
                dimi = active_dim(M.C[i])
                Xi = X[1:i,j:j]
                d2cachei = view(d2cache,:,1:length(dimi), 1:length(dimi))
                dcachei  = view(dcache,:,1:length(dimi))
                fill!(d2cachei, 0.0)
                fill!(dcachei, 0.0)
                fill!(cache, 0.0)

                reduced_hess_x_log_pdf!(d2cachei, dcachei, cache, M.C[i], Xi)
                result[j][dimi, dimi] .+= view(d2cachei,1,:,:)
                end
        end

        if apply_rescaling == true
                itransform!(M.L, X)
        end

        return result
end

# In this version the samples are treated one at a time for scalability
"""
$(TYPEDSIGNATURES)

Evaluates the hessian of the logarithm of the pullback of the reference density (standard normal) by the map `M` for the ensemble matrix `X`.
This routine exploits the sparsity pattern of the map `M`.
"""
function reduced_hess_x_log_pdf(M::HermiteMap, X; apply_rescaling::Bool = true)

        Nmax = maximum(i -> length(active_dim(M.C[i])), 1:M.Nx)
        NxX, Ne = size(X)
        Nx = M.Nx

        sparsity_pattern = spzeros(Nx, Nx)
        @inbounds for i=1:Nx
                dimi = active_dim(M.C[i])
                sparsity_pattern[dimi, dimi] .= 0.0
        end

        result = ntuple(x->copy(sparsity_pattern), Ne)

        d2cache = zeros(1, Nmax, Nmax)
        dcache  = zeros(1, Nmax)
        cache   = zeros(1)
        reduced_hess_x_log_pdf!(result, d2cache, dcache, cache, M, X; apply_rescaling = apply_rescaling)
end

## Optimization function

"""
$(TYPEDSIGNATURES)

Optimizes the `HermiteMap` `M` based on the ensemble matrix `X`. Several kind of optimization are implemented
depending on the desired robustness of the map

Several options for `optimkind` are available:
* `kfold` uses a k-fold cross validation procedure (the more robust choice)
* `split` splits the set of samples into a training and a testing
*  An `Int64` to determine the maximum number of features for each map component.
* `nothing` to simply optimize the existing coefficients in the basis expansion.

The following optionial settings can be tuned:
* `maxterms::Int64 = 1000`: a maximum number of terms for each map component
* `withconstant::Bool = false`: the option to remove the constant feature in the greedy optimization routine, if the zero feature is the constant function for the basis of interest.
* `withqr::Bool = false`: improve the conditioning of the optimization problem with a QR factorization of the feature basis (recommended option)
* `maxpatience::Int64 = 10^5`: for `optimkind = split`, the maximum number of extra terms that can be added without decreasing the validation error before the greedy optimmization get stopped.
* `verbose::Bool = false`: prints details of the optimization procedure, current component optimize, training and validation errors, number of features added.
* `apply_rescaling::Bool=true`: standardize the ensemble matrix `X` according to the `LinearTransform` `M.L`.
* `hessprecond::Bool=true`: use a preconditioner based on the Gauss-Newton of the Hessian of the loss function to accelerate the convergence.
* `start::Int64=1`: the first component of the map to optimize
* `P::Parallel = serial`: option to use multi-threading to optimize in parallel the different map components
* `ATMcriterion::String="gradient"`: sensitivty criterion used to select the feature to add to the expansion. The default uses the derivative of the cost function with respect to the coefficient
   of the features in the reduced margin of the current set of features.
"""
function optimize(M::HermiteMap, X::Array{Float64,2}, optimkind::Union{Nothing, Int64, String};
                  maxterms::Int64 = 100, withconstant::Bool = false, withqr::Bool = false, maxpatience::Int64 = 10^5,
                  verbose::Bool = false, apply_rescaling::Bool=true, hessprecond::Bool=true,
                  start::Int64=1, P::Parallel = serial, ATMcriterion::String="gradient")
        Nx = M.Nx

        @assert size(X,1) == Nx "Error dimension of the sample"

        # We can apply the rescaling to all the components once
        if apply_rescaling == true
         transform!(M.L, X)
        end

        if typeof(P) <: Serial
                # We can skip the evaluation of the map on the observation components
                for i=start:Nx
                        Xi = view(X,1:i,:)
                        M.C[i], _ = optimize(M.C[i], Xi, optimkind;
                                             maxterms = maxterms, withconstant = withconstant,
                                             withqr = withqr, maxpatience = maxpatience, verbose = verbose,
                                             hessprecond = hessprecond, ATMcriterion = ATMcriterion)
                end

        elseif typeof(P) <: Thread
                # We can skip the evaluation of the map on the observation components,
                # ThreadPools.@qthreads perform better than Threads.@threads for non-uniform tasks
                @inbounds ThreadPools.@qthreads for i=Nx:-1:start
                         Xi = view(X,1:i,:)
                         M.C[i], _ = optimize(M.C[i], Xi, optimkind;
                                              maxterms = maxterms, withconstant = withconstant,
                                              withqr = withqr, maxpatience = maxpatience, verbose = verbose,
                                              hessprecond = hessprecond, ATMcriterion = ATMcriterion)
                end
        end

        # We can apply the rescaling to all the components once
        if apply_rescaling == true
                itransform!(M.L, X)
        end

        return M
end

"""
$(TYPEDSIGNATURES)

Optimizes the `HermiteMap` `M` based on the ensemble matrix `X`.
In this routine, a greedy feature selection is performed for each map component with the entire ensemble matrix until optimkind[i] terms
are in the i-th map component.

The following optionial settings can be tuned:
* `maxterms::Int64 = 1000`: a maximum number of terms for each map component
* `withconstant::Bool = false`: the option to remove the constant feature in the greedy optimization routine, if the zero feature is the constant function for the basis of interest.
* `withqr::Bool = false`: improve the conditioning of the optimization problem with a QR factorization of the feature basis (recommended option)
* `maxpatience::Int64 = 10^5`: for `optimkind = split`, the maximum number of extra terms that can be added without decreasing the validation error before the greedy optimmization get stopped.
* `verbose::Bool = false`: prints details of the optimization procedure, current component optimize, training and validation errors, number of features added.
* `apply_rescaling::Bool=true`: standardize the ensemble matrix `X` according to the `LinearTransform` `M.L`.
* `hessprecond::Bool=true`: use a preconditioner based on the Gauss-Newton of the Hessian of the loss function to accelerate the convergence.
* `start::Int64=1`: the first component of the map to optimize
* `P::Parallel = serial`: option to use multi-threading to optimize in parallel the different map components
* `ATMcriterion::String="gradient"`: sensitivty criterion used to select the feature to add to the expansion. The default uses the derivative of the cost function with respect to the coefficient
   of the features in the reduced margin of the current set of features.
"""
function optimize(M::HermiteMap, X::Array{Float64,2}, optimkind::Array{Int64,1};
                  maxterms::Int64 = 100, withconstant::Bool = false, withqr::Bool = false,
                  verbose::Bool = false, apply_rescaling::Bool=true, hessprecond::Bool=true,
                  start::Int64=1, P::Parallel = serial, ATMcriterion = ATMcriterion)
        Nx = M.Nx

        @assert size(X,1) == Nx "Error dimension of the sample"
        @assert size(optimkind, 1) == Nx-start+1 "Error dimension of the components"

        # We can apply the rescaling to all the components once
        if apply_rescaling == true
                transform!(M.L, X)
        end

        if typeof(P) <: Serial
        # We can skip the evaluation of the map on the observation components
	        for i=start:Nx
		        Xi = view(X,1:i,:)
		        M.C[i], _ = optimize(M.C[i], Xi, optimkind[i-start+1];
                                             maxterms = maxterms, withconstant = withconstant, withqr = withqr,
                                             verbose = verbose, hessprecond = hessprecond, ATMcriterion = ATMcriterion)
	        end

        elseif typeof(P) <: Thread
	        # We can skip the evaluation of the map on the observation components,
	        # ThreadPools.@qthreads perform better than Threads.@threads for non-uniform tasks
	        @inbounds ThreadPools.@qthreads for i=Nx:-1:start
		         Xi = view(X,1:i,:)
		         M.C[i], _ = optimize(M.C[i], Xi, optimkind[i-start+1];
                                              maxterms = maxterms, withconstant = withconstant, withqr = withqr,
                                              verbose = verbose, hessprecond = hessprecond, ATMcriterion = ATMcriterion)
	        end
        end

        # We can apply the rescaling to all the components once
        if apply_rescaling == true
        	itransform!(M.L, X)
        end

        return M
end


"""
$(TYPEDSIGNATURES)

Partially-inverts the ensemble matrix `X` such that M_i([Ystar[1:Ny,j]; X[Ny+1:i,j]]) = F[Ny+i,j] for i = start:Nx
"""
function inverse!(X, F, M::HermiteMap, Ystar::AbstractMatrix{Float64}; apply_rescaling::Bool=true, start::Int64=1, P::Parallel = serial)

        Nx = M.Nx
        NxX, Ne = size(X)

        Ny, NeY = size(Ystar)
        @assert NxX == Nx
        @assert Ne == NeY
        @assert 1 <= Ny < Nx
        @assert size(F) == (Nx, Ne)

        @view(X[1:Ny,:]) .= Ystar

        # We can apply the rescaling to all the components once
        if apply_rescaling == true
            transform!(M.L, X)
        end
        # if P == serial
        # We can skip the evaluation of the map on the observations components
        @inbounds for k = start:Nx
            Fk = view(F,k,:)
            Xk = view(X,1:k,:)
            Sk = Storage(M[k].I.f, Xk)
            inverse!(Xk, Fk, M[k], Sk)

        end


        if apply_rescaling == true
            itransform!(M.L, X)
        end
        # else P == thread
        # There is a run-race problem, and the serial version is fast enough.
        #         nthread = Threads.nthreads()
        #         @time if nthread == 1
        #                 idx_folds = 1:Ne
        #         else
        #                 q = div(Ne, nthread)
        #                 r = rem(Ne, nthread)
        #                 @assert Ne == q*nthread + r
        #                 idx_folds = UnitRange{Int64}[i < nthread ? ((i-1)*q+1:i*q) : ((i-1)*q+1:i*q+r) for i in 1:nthread]
        #         end
        #
        #         @inbounds Threads.@threads for idx in idx_folds
        #                 for k = start:Nx
        #                 Fk = view(F,k,idx)
        #                 Xk = view(X,1:k,idx)
        #                 Sk = Storage(M[k].I.f, Xk)
        #                 inverse!(Xk, Fk, M[k], Sk)
        #                 end
        #         end
        # end
end


"""
$(TYPEDSIGNATURES)

Partially-inverts the ensemble matrix `X` such that M_i([ystar; X[Ny+1:i,j]]) = F[Ny+i,j] for i = start:Nx
"""
function inverse!(X, F, M::HermiteMap, ystar::AbstractVector{Float64}; apply_rescaling::Bool=true, start::Int64=1, P::Parallel = serial)

        Nx = M.Nx
        NxX, Ne = size(X)

        Ny = size(ystar,1)
        @assert NxX == Nx
        @assert 1 <= Ny < Nx
        @assert size(F) == (Nx, Ne)


        @view(X[1:Ny,:]) .= ystar

        # We can apply the rescaling to all the components once
        if apply_rescaling == true
            transform!(M.L, X)
        end

        # We can skip the evaluation of the map on the observations components
        @inbounds for k = start:Nx
            Fk = view(F,k,:)
            Xk = view(X,1:k,:)
            Sk = Storage(M[k].I.f, Xk)
            inverse!(Xk, Fk, M[k], Sk)
        end


        if apply_rescaling == true
            itransform!(M.L, X)
        end
end

"""
$(TYPEDSIGNATURES)

Inverts the ensemble matrix `X` such that `M(X) = F`
"""
function inverse!(X, F, M::HermiteMap; apply_rescaling::Bool=true, P::Parallel = serial)

        Nx = M.Nx
        NxX, Ne = size(X)

        @assert NxX == Nx
        @assert size(F) == (Nx, Ne)

        # We can apply the rescaling to all the components once
        if apply_rescaling == true
            transform!(M.L, X)
        end
        # We can skip the evaluation of the map on the observations components
        @inbounds for k = 1:Nx
            Fk = view(F,k,:)
            Xk = view(X,1:k,:)
            Sk = Storage(M[k].I.f, Xk)
            inverse!(Xk, Fk, M[k], Sk)
        end

        if apply_rescaling == true
            itransform!(M.L, X)
        end
end


"""
$(TYPEDSIGNATURES)

Partially-inverts the ensemble matrix `X` such that M_i([Ystar[1:Ny,j]; X[Ny+1:i,j]]) = F[Ny+i,j] for i = start:Nx
"""
function hybridinverse!(X, F, M::HermiteMap, Ystar::AbstractMatrix{Float64}; apply_rescaling::Bool=true, start::Int64=1, P::Parallel = serial)

        Nx = M.Nx
        NxX, Ne = size(X)

        Ny, NeY = size(Ystar)
        @assert NxX == Nx
        @assert Ne == NeY
        @assert 1 <= Ny < Nx
        @assert size(F) == (Nx, Ne)


        @view(X[1:Ny,:]) .= Ystar

        # We can apply the rescaling to all the components once
        if apply_rescaling == true
            transform!(M.L, X)
        end
        # if P == serial
        # We can skip the evaluation of the map on the observations components
        @inbounds for k = start:Nx
            Fk = view(F,k,:)
            Xk = view(X,1:k,:)
            Sk = Storage(M[k].I.f, Xk)
            hybridinverse!(Xk, Fk, M[k], Sk; P = P)
        end


        if apply_rescaling == true
            itransform!(M.L, X)
        end
end


"""
$(TYPEDSIGNATURES)

Partially-inverts the ensemble matrix `X` such that M_i([ystar; X[Ny+1:i,j]]) = F[Ny+i,j] for i = start:Nx
"""
function hybridinverse!(X, F, M::HermiteMap, ystar::AbstractVector{Float64}; apply_rescaling::Bool=true, start::Int64=1, P::Parallel = serial)

        Nx = M.Nx
        NxX, Ne = size(X)

        Ny = size(ystar,1)
        @assert NxX == Nx
        @assert 1 <= Ny < Nx
        @assert size(F) == (Nx, Ne)


        @view(X[1:Ny,:]) .= ystar

        # We can apply the rescaling to all the components once
        if apply_rescaling == true
            transform!(M.L, X)
        end

        # We can skip the evaluation of the map on the observations components
        @inbounds for k = start:Nx
            Fk = view(F,k,:)
            Xk = view(X,1:k,:)
            Sk = Storage(M[k].I.f, Xk)
            hybridinverse!(Xk, Fk, M[k], Sk; P = P)
        end


        if apply_rescaling == true
            itransform!(M.L, X)
        end
end


"""
$(TYPEDSIGNATURES)

Inverts the ensemble matrix `X` such that `M(X) = F`
"""
function hybridinverse!(X, F, M::HermiteMap; apply_rescaling::Bool=true, P::Parallel = serial)

        Nx = M.Nx
        NxX, Ne = size(X)

        @assert NxX == Nx
        @assert size(F) == (Nx, Ne)

        # We can apply the rescaling to all the components once
        if apply_rescaling == true
            transform!(M.L, X)
        end
        # We can skip the evaluation of the map on the observations components
        @inbounds for k = 1:Nx
            Fk = view(F,k,:)
            Xk = view(X,1:k,:)
            Sk = Storage(M[k].I.f, Xk)
            hybridinverse!(Xk, Fk, M[k], Sk; P = P)
        end

        if apply_rescaling == true
            itransform!(M.L, X)
        end
end
