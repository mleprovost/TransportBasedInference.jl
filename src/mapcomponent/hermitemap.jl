export  HermiteMap,
        evaluate!,
        evaluate,
        optimize

struct HermiteMap
    m::Int64
    Nx::Int64
    L::LinearTransform
    C::Array{MapComponent,1}
end

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

        # if typeof(P) <: Serial
        # We can skip the evaluation of the map on the observations componentd
        @inbounds for k=start:Nx
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
        end

        return out
end

evaluate(M::HermiteMap, X; apply_rescaling::Bool=true, start::Int64=1, P::Parallel = serial) =
         evaluate!(zero(X), M, X; apply_rescaling = apply_rescaling, start = start, P = P)

function optimize(M::HermiteMap, X::Array{Float64,2}, maxterms::Union{Nothing, Int64, String};
                  verbose::Bool = false, apply_rescaling::Bool=true, start::Int64=1, P::Parallel = serial)
         Nx = M.Nx
         @assert size(X,1) == Nx "Error dimension of the sample"

         # We can apply the rescaling to all the components once
         if apply_rescaling == true
                 transform!(M.L, X)
         end

         @inbounds for i = start:Nx
                Xi = view(X,1:i,:)
                M.C[i], _ = optimize(M.C[i], Xi, maxterms; verbose = verbose)
         end

         # We can apply the rescaling to all the components once
         if apply_rescaling == true
                 itransform!(M.L, X)
         end

         return M
end


 # function inverse!(out, M::HermiteMap{m, Nx}, X; apply_rescaling::Bool=true, start::Int64=1, P::Parallel = serial) where {m, Nx}
 #
 #         NxX, Ne = size(X)
 #         @assert NxX == Nx
 #         @assert size(out, 2) == Ne
 #
 #         # We can apply the rescaling to all the components once
 #         if apply_rescaling == true
 #                 transform!(M.L, X)
 #         end
 #
 #         # if typeof(P) <: Serial
 #         # We can skip the evaluation of the map on the observations componentd
 #         @inbounds for k=start:Nx
 #                 Xk = view(X,1:k,:)
 #                 col = view(out,k,:)
 #                 evaluate!(col, M.C[k], Xk)
 #         end
 #         # elseif typeof(P) <: Thread
 #         #
 #         #
 #         # end
 #
 #         if apply_rescaling == true
 #                 itransform!(M.L, X)
 #         end
 #
 #         return out
 # end
