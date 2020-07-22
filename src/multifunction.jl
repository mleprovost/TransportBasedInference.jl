export MultiFunction, first

# MultiFunction is a type to hold an elementary function F: R^{Nx} → R
# that can be decomposed as the product of univariate basis,
# where each basis is finite and contains constant and/or linear
# and Hermite functions
# F(x_1, x_2, ..., x_Nx) = f_1(x_1) × f_2(x_2) × ... × f_M(x_Nx)


struct MultiFunction{m, Nx}
    B::MultiBasis{m, Nx}
    α::Array{Int64,1}
    function MultiFunction(B::MultiBasis{m, Nx}, α::Array{Int64,1}; check::Bool=true) where {m, Nx}
        if check==true
            @assert Nx == size(α,1) "Dimension of the space doesn't match the size of α"
            for i=1:Nx
                @assert α[i]<=m "multi index α can't be greater than the size of the univariate basis "
            end
        end
        return new{m, Nx}(B, α)
    end
end


function MultiFunction(B::MultiBasis{m, Nx}) where {m, Nx}
    return MultiFunction{m, Nx}(B, ones(Int64, Nx))
end

function MultiFunction(B::Basis{m}, Nx::Int64; scaled::Bool = true) where {m}
    return MultiFunction{m, Nx}(MultiBasis(B, Nx), ones(Int64, Nx))
end


function (F::MultiFunction{m, Nx})(x::Array{T,1}) where {m, Nx, T <: Real}

    # @assert Nx == size(x,1) "Wrong dimension of input vector x"
    out = 1.0
    @inbounds for i=1:Nx
        out *= F.B.B[F.α[i]+1](x[i])
    end
    return out
end
