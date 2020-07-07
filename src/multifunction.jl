export MultiFunction


struct MultiFunction{M}
    B::MultiBasis{M}
    α::Array{Int64,1}
    scaled::Bool
    function MultiFunction(B::MultiBasis{M}, α::Array{Int64,1}; scaled::Bool=true) where {M}
        for i=1:M
            @assert α[i]<=size(B[i],1) "multi index α can't be greater than the size of the univariate basis "
        end
        return new{M}(B, α, scaled)
    end
end


function MultiFunction(B::MultiBasis{M}; scaled::Bool = true) where {M}
    return MultiFunction{M}(B, ones(Int64, M), scaled)
end

function MultiFunction(B::Basis, M::Int64; scaled::Bool = true)
    return MultiFunction{M}(MultiBasis(B, M), ones(Int64, M), scaled)
end


function (f::MultiFunction{M})(x::Array{T,1}) where {M, T <: Real}
    N = size(x, 1)
    @assert N==M "Wrong dimension of imput vector x"
    out = 1.0
    for i=1:M
        if f.α[i]>1
            out *= f.B[i][f.α[i]](x[i])
        end
    end
    return out
end

# function
