export MultiFunction, first


"""
$(TYPEDEF)

An immutable structure to hold an elementary function F: Rᵏ → R
that can be decomposed as the product of univariate basis,
where each basis is finite and contains constant and/or linear
and Hermite functions
F(x_1, x_2, ..., x_k) = f_1(x_1) × f_2(x_2) × ... × f_M(x_k)

## Fields

$(TYPEDFIELDS)

"""
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

"""
$(TYPEDSIGNATURES)
Evaluates the `MultiFunction` `F` at `x`
"""
function (F::MultiFunction)(x::Array{T,1}) where {T <: Real}
    out = 1.0
    @inbounds for i=1:F.Nx
        out *= F.B.B[F.α[i]+1](x[i])
    end
    return out
end
