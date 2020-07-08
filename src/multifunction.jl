export MultiFunction, first

# MultiFunction is a type to hold an elementary function F: R^{Nx} → R
# that can be decomposed as the product of univariate basis,
# where each basis is finite and contains constant and/or linear
# and Hermite functions
# F(x_1, x_2, ..., x_Nx) = f_1(x_1) × f_2(x_2) × ... × f_M(x_Nx)


struct MultiFunction{m, Nx}
    B::MultiBasis{m, Nx}
    α::Array{Int64,1}
    scaled::Bool
    function MultiFunction(B::MultiBasis{m, Nx}, α::Array{Int64,1}; scaled::Bool=true) where {m, Nx}
        @assert Nx == size(α,1) "Dimension of the space doesn't match the size of α"
        for i=1:Nx
            @assert α[i]<=m "multi index α can't be greater than the size of the univariate basis "
        end
        return new{m, Nx}(B, α, scaled)
    end
end


function MultiFunction(B::MultiBasis{m, Nx}; scaled::Bool = true) where {m, Nx}
    return MultiFunction{m, Nx}(B, ones(Int64, Nx), scaled)
end

function MultiFunction(B::Basis{m}, Nx::Int64; scaled::Bool = true) where {m}
    return MultiFunction{m, Nx}(MultiBasis(B, Nx), ones(Int64, Nx), scaled)
end


function (F::MultiFunction{m, Nx})(x::Array{T,1}) where {m, Nx, T <: Real}

    @assert Nx == size(x,1) "Wrong dimension of input vector x"
    out = 1.0
    for i=1:Nx
        if F.α[i]>1
            out *= F.B.B[F.α[i]](x[i])
        end
    end
    return out
end
