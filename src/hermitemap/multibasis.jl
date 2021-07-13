import Base: size

export MultiBasis

"""
$(TYPEDEF)

An immutable structure to hold the Nx-cartesian product of the 1D basis of functions B
MultiB = B × B× ... × B (Nx elements). The basis B contains m elements.

## Fields

$(TYPEDFIELDS)

"""
struct MultiBasis
    B::Basis
    Nx::Int64
end

size(B::MultiBasis) = (B.B.m, B.Nx)
