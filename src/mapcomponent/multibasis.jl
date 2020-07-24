import Base: size

export MultiBasis, Element

# MultiBasis holds the Nx- cartesian product of the  1D basis of functions B
# MB = B × B× ... × B (Nx elements)
# The basis B contains m elements

struct MultiBasis{m, Nx}
    B::Basis{m}
end


MultiBasis(B::Basis{m}, Nx::Int64) where {m} = MultiBasis{m, Nx}(B)


size(B::MultiBasis{m, Nx}) where {m, Nx} = (m, Nx)

# MultiBasis()
