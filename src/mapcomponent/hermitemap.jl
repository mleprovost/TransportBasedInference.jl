export  HermiteMap,
        evaluate!,
        evaluate,
        log_pdf,
        grad_x_log_pdf!,
        grad_x_log_pdf,
        hess_x_log_pdf!,
        hess_x_log_pdf,
        reduced_hess_x_log_pdf!,
        reduced_hess_x_log_pdf,
        optimize

import Base: @propagate_inbounds

struct HermiteMap
        m::Int64
        Nx::Int64
        L::LinearTransform
        C::Array{MapComponent,1}
end

function HermiteMap(L::LinearTransform, C::Array{MapComponent,1})
        m = C[end].m
        Nx = C[end].Nx

        return HermiteMap(m, Nx, L, C)
end

@propagate_inbounds Base.getindex(M::HermiteMap, i::Int) = getindex(M.C,i)
@propagate_inbounds Base.setindex!(M::HermiteMap, C::MapComponent, i::Int) = setindex!(M.C,C,i)


function HermiteMap(m::Int64, X::Array{Float64,2}; diag::Bool=true, α::Float64 = 1e-6)
        L = LinearTransform(X; diag = diag)

        B = CstProHermite(m-2)
        Nx = size(X,1)
        Nψ = 1
        coeff = zeros(Nψ)
        C = MapComponent[]
        idx = zeros(Int, Nψ, Nx)
        @inbounds for i=1:Nx
                MultiB = MultiBasis(B, i)
                vidx = idx[:,1:i]
                push!(C, MapComponent(IntegratedFunction(ExpandedFunction(MultiB, vidx, coeff))))
        end

        return HermiteMap(m, Nx, L, C)
end


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

evaluate(M::HermiteMap, X; apply_rescaling::Bool=true, start::Int64=1, P::Parallel = serial) =
         evaluate!(zero(X), M, X; apply_rescaling = apply_rescaling, start = start, P = P)


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

log_pdf(M::HermiteMap, X; apply_rescaling::Bool = true) = log_pdf(M, X, 1:M.Nx; apply_rescaling = apply_rescaling)

## Compute grad_x_log_pdf

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

grad_x_log_pdf(M::HermiteMap, X; apply_rescaling::Bool = true) = grad_x_log_pdf!(zeros(size(X,2), size(X,1)), zeros(size(X,2), size(X,1)), zeros(size(X,2)),
                                                                 M, X; apply_rescaling = apply_rescaling)

## Compute hess_x_log_pdf

# function hess_x_log_pdf!(result, cache_hess, cache_grad, cache, M::HermiteMap, X; apply_rescaling::Bool = true)
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

hess_x_log_pdf(M::HermiteMap, X; apply_rescaling::Bool = true) = hess_x_log_pdf!(zeros(size(X,2), size(X,1), size(X,1)), M, X; apply_rescaling = apply_rescaling)


# This forms output an array of dimension (Ne, Nx, Nx) but the intermediate computation have been done
# by only looking at the active dimensions

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

function optimize(M::HermiteMap, X::Array{Float64,2}, maxterms::Union{Nothing, Int64, String};
                  withconstant::Bool = false, withqr::Bool = false, verbose::Bool = false, apply_rescaling::Bool=true, conditioner::Bool=true,
                  start::Int64=1, P::Parallel = serial)
        Nx = M.Nx

        @assert size(X,1) == Nx "Error dimension of the sample"

        # We can apply the rescaling to all the components once
        if apply_rescaling == true
         transform!(M.L, X)
        end

        if typeof(P) <: Serial
        # We can skip the evaluation of the map on the observations components
        @showprogress for i=start:Nx
         Xi = view(X,1:i,:)
         M.C[i], _ = optimize(M.C[i], Xi, maxterms; withconstant = withconstant,
                              withqr = withqr, verbose = verbose, conditioner = conditioner)
        end

        elseif typeof(P) <: Thread
        # We can skip the evaluation of the map on the observations components,
        # ThreadPools.@qthreads perform better than Threads.@threads for non-uniform tasks
        @inbounds ThreadPools.@qthreads for i=Nx:-1:start
         Xi = view(X,1:i,:)
         M.C[i], _ = optimize(M.C[i], Xi, maxterms; withconstant = withconstant,
                              withqr = withqr, verbose = verbose, conditioner = conditioner)
        end
        end

        # We can apply the rescaling to all the components once
        if apply_rescaling == true
         itransform!(M.L, X)
        end

        return M
end

function inverse!(F, M::HermiteMap, X, Ystar; apply_rescaling::Bool=true, start::Int64=1, P::Parallel = serial)

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

# function inverse!(F, M::HermiteMap, X, ystar; apply_rescaling::Bool=true, start::Int64=1, P::Parallel = serial)
#
#         Nx = M.Nx
#         NxX, Ne = size(X)
#
#         Ny, NeY = size(Ystar)
#         @assert NxX == Nx
#         @assert Ne == NeY
#         @assert 1 <= Ny < Nx
#         @assert size(F) == (Nx, Ne)
#
#         @view(X[1:Ny,:]) .= Ystar
#
#         # We can apply the rescaling to all the components once
#         if apply_rescaling == true
#             transform!(M.L, X)
#         end
#         # if P == serial
#         # We can skip the evaluation of the map on the observations components
#         @inbounds for k = start:Nx
#             Fk = view(F,k,:)
#             Xk = view(X,1:k,:)
#             Sk = Storage(M[k].I.f, Xk)
#             inverse!(Xk, Fk, M[k], Sk)
#         end
#
#
#         if apply_rescaling == true
#             itransform!(M.L, X)
#         end
#         # else P == thread
#         # There is a run-race problem, and the serial version is fast enough.
#         #         nthread = Threads.nthreads()
#         #         @time if nthread == 1
#         #                 idx_folds = 1:Ne
#         #         else
#         #                 q = div(Ne, nthread)
#         #                 r = rem(Ne, nthread)
#         #                 @assert Ne == q*nthread + r
#         #                 idx_folds = UnitRange{Int64}[i < nthread ? ((i-1)*q+1:i*q) : ((i-1)*q+1:i*q+r) for i in 1:nthread]
#         #         end
#         #
#         #         @inbounds Threads.@threads for idx in idx_folds
#         #                 for k = start:Nx
#         #                 Fk = view(F,k,idx)
#         #                 Xk = view(X,1:k,idx)
#         #                 Sk = Storage(M[k].I.f, Xk)
#         #                 inverse!(Xk, Fk, M[k], Sk)
#         #                 end
#         #         end
#         # end
# end
