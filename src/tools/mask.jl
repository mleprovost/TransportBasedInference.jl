

struct MaskedElementArray{T, N, A <: AbstractArray{T, N}} <: AbstractArray{T, N}
         parent::A
         masked_index::Int
end

function MaskedElementArray(parent::A, masked_index::Integer) where {T, N, A <: AbstractArray{T, N}}
         MaskedElementArray{T, N, A}(parent, masked_index)
end

Base.size(a::MaskedElementArray) = size(a.parent)
Base.eltype(a::MaskedElementArray) = eltype(a.parent)


function Base.getindex(a::MaskedElementArray, i::Integer)
         if i == a.masked_index
           zero(eltype(a))
         else
           a.parent[i]
         end
end
