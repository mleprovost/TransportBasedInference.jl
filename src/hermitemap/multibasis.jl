import Base: size

export MultiBasis, getbasis

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

"""
$(TYPEDSIGNATURES)

Returns the size of the `MultiBasis` `B`.
"""
size(B::MultiBasis) = (B.B.m, B.Nx)

"""
$(TYPEDSIGNATURES)

Returns the kind of the underlying basis of the `MultiBasis` `B`.
"""
getbasis(B::MultiBasis) = string(typeof(B.B))
