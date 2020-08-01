export  HermiteMap,
        evaluate!,
        evaluate,
        log_pdf,
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

        B = CstProHermite(m-2; scaled =true)
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

        # We can apply the rescaling to all the components once
        if apply_rescaling == true
                itransform!(M.L, X)
        end
        return logπ
end

log_pdf(M::HermiteMap, X; apply_rescaling::Bool = true) = log_pdf(M, X, 1:M.Nx; apply_rescaling = apply_rescaling)


## Optimization function

function optimize(M::HermiteMap, X::Array{Float64,2}, maxterms::Union{Nothing, Int64, String};
                  withconstant::Bool = false, verbose::Bool = false, apply_rescaling::Bool=true, start::Int64=1, P::Parallel = serial)
         Nx = M.Nx

         @assert size(X,1) == Nx "Error dimension of the sample"

         # We can apply the rescaling to all the components once
         if apply_rescaling == true
                 transform!(M.L, X)
         end

         if typeof(P) <: Serial
         # We can skip the evaluation of the map on the observations components
         @inbounds for i=start:Nx
                 Xi = view(X,1:i,:)
                 M.C[i], _ = optimize(M.C[i], Xi, maxterms; withconstant = withconstant,
                                      verbose = verbose)
         end

         elseif typeof(P) <: Thread
         # We can skip the evaluation of the map on the observations components
         @inbounds Threads.@threads for i=start:Nx
                 Xi = view(X,1:i,:)
                 M.C[i], _ = optimize(M.C[i], Xi, maxterms; withconstant = withconstant,
                                      verbose = verbose)
         end
         end

         # We can apply the rescaling to all the components once
         if apply_rescaling == true
                 itransform!(M.L, X)
         end

         return M
end

# F is an array than contians the evaluation of the Hermite Map for each prior star
function inverse!(F, M::HermiteMap, X, Ystar; apply_rescaling::Bool=true, P::Parallel = serial)
         Nx = M.Nx
         Ny, NeY = size(Ystar)
         NxX, Ne = size(X)
         @assert NxX == Nx
         @assert NeY == Ne
         @assert 1 <= Ny < Nx
         @assert size(F, 2) == Ne
         @assert size(F, 2) == Ne

         # We can apply the rescaling to all the components once
         if apply_rescaling == true
                 transform!(M.L, X)
                 transform!(M.L, Ystar)
         end

         # if typeof(P) <: Serial
         # We can skip the evaluation of the map on the observations componentd
         @inbounds for k=Ny+1:Nx
                 Xk = view(X,1:k,:)
                 col = view(out,k,:)
                 evaluate!(col, M.C[k], Xk)
         end
         # elseif typeof(P) <: Thread
         #
         #
         # end

         if apply_rescaling == true
                 itransform!(M.L, X)
                 itransform!(M.L, Ystar)
         end

         return out
 end
