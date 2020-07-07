import Base: size, show, @propagate_inbounds

export MultiBasis, Element


struct MultiBasis{M}
    B::Array{Basis, 1}
    function MultiBasis(B::Array{Basis, 1})
        M = size(B, 1)
        return new{M}(B)
    end
end


MultiBasis(B::Basis{m}, M::Int64) where {m} = MultiBasis{M}([B for i=1:M])

@propagate_inbounds Base.getindex(B::MultiBasis{M},i::Int) where {M} = getindex(B.B,i)
@propagate_inbounds Base.setindex!(B::MultiBasis{M}, v::Basis{m}, i::Int) where {M, m} = setindex!(B.B,v,i)

size(B::MultiBasis{M},d::Int) where {M} = size(B.B,d)
size(B::MultiBasis{M}) where {M} = size(B.B)
