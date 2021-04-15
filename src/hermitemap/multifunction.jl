export MultiFunction, first

# MultiFunction is a type to hold an elementary function F: R^{k} → R
# that can be decomposed as the product of univariate basis,
# where each basis is finite and contains constant and/or linear
# and Hermite functions
# F(x_1, x_2, ..., x_k) = f_1(x_1) × f_2(x_2) × ... × f_M(x_k)


struct MultiFunction
    m::Int64
    Nx::Int64
    B::MultiBasis
    α::Array{Int64,1}
    function MultiFunction(B::MultiBasis, α::Array{Int64,1})
        m = B.B.m
        Nx = B.Nx
        @assert Nx == size(α,1) "Dimension of the space doesn't match the size of α"
        for i=1:Nx
            @assert α[i]<=m "multi index α can't be greater than the size of the univariate basis "
        end
        return new(m, Nx, B, α)
    end
end


function MultiFunction(B::MultiBasis)
    return MultiFunction(B.B.m, B.Nx, B, ones(Int64, B.Nx))
end

function MultiFunction(B::Basis, Nx::Int64; scaled::Bool = true)
    return MultiFunction(B.B.m, Nx, MultiBasis(k, B), ones(Int64, Nx))
end

# This version is only working for the basis CstProHermite with rescaling
function (F::MultiFunction)(x::Array{T,1}) where {T <: Real}

    # @assert Nx == size(x,1) "Wrong dimension of input vector x"
    out = 1.0
    @inbounds for i=1:F.Nx
        out *= F.B.B[F.α[i]+1](x[i])
        # Skip the 1.0 constant evaluation if F.α[i] = 0
        # if F.α[i]>0
        #     out *= FamilyScaledProHermite[F.α[i]](x[i])
        #     # out *= F.B.B[F.α[i]+1](x[i])
        # end
    end
    return out
end

# # This version is only working for the basis CstProHermite with rescaling
# function (F::MultiFunction)(x::Array{T,1}) where {T <: Real}
#
#     # @assert Nx == size(x,1) "Wrong dimension of input vector x"
#     out = 1.0
#     @inbounds for i=1:F.Nx
#         # Skip the 1.0 constant evaluation if F.α[i] = 0
#         if F.α[i]>0
#             out *= FamilyScaledProHermite[F.α[i]](x[i])
#             # out *= F.B.B[F.α[i]+1](x[i])
#         end
#     end
#     return out
# end
