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

Returns the size of the `MultiBasis` `MB`.
"""
size(MB::MultiBasis) = (MB.B.m, MB.Nx)

"""
$(TYPEDSIGNATURES)

Returns the kind of the underlying basis of the `MultiBasis` `MB`.
"""
getbasis(MB::MultiBasis) = string(typeof(MB.B))
