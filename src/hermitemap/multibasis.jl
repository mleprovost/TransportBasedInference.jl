import Base: size

export MultiBasis

# MultiBasis holds the Nx- cartesian product of the  1D basis of functions B
# MB = B × B× ... × B (Nx elements)
# The basis B contains m elements

struct MultiBasis
    B::Basis
    Nx::Int64
end

size(B::MultiBasis) = (B.B.m, B.Nx)
