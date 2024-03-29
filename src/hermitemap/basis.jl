import Base: size, show, @propagate_inbounds


# Define the concept of basis functions, where each function is indexed by an integer
# For instance, (1, x, ψ0, ψ1,..., ψn) defines a basis where the index:
# 0 corresponds to the constant function
# 1 corresponds to the linear function
# n+2 corresponds to the n-th order physicist Hermite function

export Basis,
       vander!,
       vander,
       ProHermiteBasis,
       PhyHermiteBasis,
       CstProHermiteBasis,
       CstPhyHermiteBasis,
       CstLinProHermiteBasis,
       CstLinPhyHermiteBasis,
       iszerofeatureactive

"""
$(TYPEDEF)

An abstract type to define basis of functions

## Constructors
- `Basis(m)`

## Construct new basis of functions

A specific feature basis `MyBasis` must be a subtype of `Basis`, and the following routines must be implemented:
* `Base.show(io::IO, B::MyBasis)`` (optional, but desired)
* `@propagate_inbounds Base.getindex(B::MyBasis, i::Int64)`
* `vander!(dV, B::MyBasis, maxi::Int64, k::Int64, x)`
* `iszerofeatureactive(B::MyBasis) = Bool`
"""
abstract type Basis end

Base.size(B::Basis) = B.m

"""
$(TYPEDEF)

The basis composed of the probabilistic Hermite functions

## Fields
$(TYPEDFIELDS)
"""
struct ProHermiteBasis <: Basis
    m::Int64
end

@propagate_inbounds Base.getindex(B::ProHermiteBasis, i::Int64) = FamilyScaledProHermite[i]

function Base.show(io::IO, B::ProHermiteBasis)
    println(io,"Basis of "*string(B.m)*" functions: 0th -> "*string(B.m-1)*"th degree Probabilistic Hermite function")
end

"""
$(TYPEDEF)

The basis composed of the physicist Hermite functions

## Fields
$(TYPEDFIELDS)
"""
struct PhyHermiteBasis <: Basis
    m::Int64
end

@propagate_inbounds Base.getindex(B::PhyHermiteBasis, i::Int64) = FamilyScaledPhyHermite[i]

function Base.show(io::IO, B::PhyHermiteBasis)
    println(io,"Basis of "*string(B.m)*" functions: 0th -> "*string(B.m-1)*"th degree Physicist Hermite function")
end

"""
$(TYPEDEF)

The basis composed of the constant function and probabilistic Hermite functions

## Fields
$(TYPEDFIELDS)
"""
struct CstProHermiteBasis <: Basis
    m::Int64
end

@propagate_inbounds Base.getindex(B::CstProHermiteBasis, i::Int64) = i==1 ? FamilyProPolyHermite[1] : FamilyScaledProHermite[i-1]

function Base.show(io::IO, B::CstProHermiteBasis)
    println(io,"Basis of "*string(B.m)*" functions: Constant, 0th -> "*string(B.m-2)*"th degree Probabilistic Hermite function")
end

"""
$(TYPEDEF)

The basis composed of the constant function and physicist Hermite functions

## Fields
$(TYPEDFIELDS)
"""
struct CstPhyHermiteBasis <: Basis
    m::Int64
end

@propagate_inbounds Base.getindex(B::CstPhyHermiteBasis, i::Int64) = i==1 ? FamilyProPolyHermite[1] : FamilyScaledPhyHermite[i-1]

function Base.show(io::IO, B::CstPhyHermiteBasis)
    println(io,"Basis of "*string(B.m)*" functions: Constant, 0th -> "*string(B.m-2)*"th degree Physicist Hermite function")
end

"""
$(TYPEDEF)

The basis composed of the constant function, the identity, and probabilistic Hermite functions

## Fields
$(TYPEDFIELDS)
"""
struct CstLinProHermiteBasis <: Basis
    m::Int64
end

@propagate_inbounds Base.getindex(B::CstLinProHermiteBasis, i::Int64) = i==1 ? FamilyProPolyHermite[1] : i==2 ? FamilyProPolyHermite[2] : FamilyScaledProHermite[i-2]

function Base.show(io::IO, B::CstLinProHermiteBasis)
    println(io,"Basis of "*string(B.m)*" functions: Constant, Linear, 0th -> "*string(B.m-3)*"th degree Probabilistic Hermite function")
end

"""
$(TYPEDEF)

The basis composed of the constant function, the identity, and physicist Hermite functions

## Fields
$(TYPEDFIELDS)
"""
struct CstLinPhyHermiteBasis <: Basis
    m::Int64
end

@propagate_inbounds Base.getindex(B::CstLinPhyHermiteBasis, i::Int64) = i==1 ? FamilyProPolyHermite[1] : i==2 ? FamilyProPolyHermite[2] : FamilyScaledPhyHermite[i-2]

function Base.show(io::IO, B::CstLinPhyHermiteBasis)
    println(io,"Basis of "*string(B.m)*" functions: Constant, Linear 0th -> "*string(B.m-3)*"th degree Physicist Hermite function")
end



# Implementation of vander! for the different bases
"""
    vander!(dV, B::T, m::Int64, k::Int64, x) where T <: Basis

Compute the Vandermonde matrix of the basis `B` for the vector `x`up to the m-th feature of the basis
"""

"""
    vander!(dV, B::ProHermiteBasis, maxi::Int64, k::Int64, x)

Compute the Vandermonde matrix for the vector `x`
"""
function vander!(dV, B::ProHermiteBasis, m::Int64, k::Int64, x)
    N = size(x,1)
    @assert size(dV) == (N, m+1) "Wrong dimension of the Vander matrix"
    dVshift = view(dV,:,1:m+1)
    vander!(dVshift, FamilyScaledProHermite[m+1], k, x)
    return dV
end

"""
    vander!(dV, B::PhyHermiteBasis, maxi::Int64, k::Int64, x)

Compute the Vandermonde matrix for the vector `x`
"""
function vander!(dV, B::PhyHermiteBasis, m::Int64, k::Int64, x)
    N = size(x,1)
    @assert size(dV) == (N, m+1) "Wrong dimension of the Vander matrix"
    dVshift = view(dV,:,1:m+1)
    vander!(dVshift, FamilyScaledPhyHermite[m+1], k, x)
    return dV
end

"""
    vander!(dV, B::T, m::Int64, k::Int64, x) where T <: Basis

Compute the Vandermonde matrix of the basis `B` for the vector `x`up to the m-th feature of the basis
"""
function vander!(dV, B::CstProHermiteBasis, m::Int64, k::Int64, x)
    N = size(x,1)
    @assert size(dV) == (N, m+1) "Wrong dimension of the Vander matrix"

    col0 = view(dV,:,1)
    if k==0
         fill!(col0, 1.0)
    else
         fill!(col0, 0.0)
    end
    if m == 0
        return dV
    end
    dVshift = view(dV,:,2:m+1)
    vander!(dVshift, FamilyScaledProHermite[m], k, x)
     # .= vander(B.f[maxi+1], k, x)
    return dV
end

"""
    vander!(dV, B::CstPhyHermiteBasis, maxi::Int64, k::Int64, x)

Compute the Vandermonde matrix for the vector `x`
"""
function vander!(dV, B::CstPhyHermiteBasis, m::Int64, k::Int64, x)
    N = size(x,1)
    @assert size(dV) == (N, m+1) "Wrong dimension of the Vander matrix"

    col0 = view(dV,:,1)
    if k==0
         fill!(col0, 1.0)
    else
         fill!(col0, 0.0)
    end
    if m == 0
        return dV
    end
    dVshift = view(dV,:,2:m+1)
    vander!(dVshift, FamilyScaledPhyHermite[m], k, x)
     # .= vander(B.f[maxi+1], k, x)
    return dV
end

"""
    vander!(dV, B::CstLinProHermiteBasis, maxi::Int64, k::Int64, x)

Compute the Vandermonde matrix for the vector `x`
"""
function vander!(dV, B::CstLinProHermiteBasis, m::Int64, k::Int64, x)
    N = size(x,1)
    @assert size(dV) == (N, m+1) "Wrong dimension of the Vander matrix"

    if m >= 0
        # Constant feature
        col0 = view(dV,:,1)
        if k==0
             fill!(col0, 1.0)
        else
             fill!(col0, 0.0)
        end
    end

    if m >= 1
        # Linear eature
        col1 = view(dV,:,2)
        if k==0
            copy!(col1, x)
        elseif k==1
            fill!(col1, 1.0)
        else
            fill!(col1, 0.0)
        end
    end

    if m < 2
        return dV
    end
    dVshift = view(dV,:,3:m+1)
    vander!(dVshift, FamilyScaledProHermite[m-1], k, x)
     # .= vander(B.f[maxi+1], k, x)
    return dV
end

"""
    vander!(dV, B::CstLinPhyHermiteBasis, maxi::Int64, k::Int64, x)

Compute the Vandermonde matrix for the vector `x`
"""
function vander!(dV, B::CstLinPhyHermiteBasis, m::Int64, k::Int64, x)
    N = size(x,1)
    @assert size(dV) == (N, m+1) "Wrong dimension of the Vander matrix"

    if m >= 0
        # Constant feature
        col0 = view(dV,:,1)
        if k==0
             fill!(col0, 1.0)
        else
             fill!(col0, 0.0)
        end
    end

    if m >= 1
        # Linear eature
        col1 = view(dV,:,2)
        if k==0
            copy!(col1, x)
        elseif k==1
            fill!(col1, 1.0)
        else
            fill!(col1, 0.0)
        end
    end

    if m < 2
        return dV
    end
    dVshift = view(dV,:,3:m+1)
    vander!(dVshift, FamilyScaledPhyHermite[m-1], k, x)
     # .= vander(B.f[maxi+1], k, x)
    return dV
end

vander!(dV, B::Union{PhyHermiteBasis, ProHermiteBasis}, k::Int64, x) = vander!(dV, B, B.m-1, k, x)
vander!(dV, B::Union{CstPhyHermiteBasis, CstProHermiteBasis}, k::Int64, x) = vander!(dV, B, B.m-1, k, x)
vander!(dV, B::Union{CstLinPhyHermiteBasis, CstLinProHermiteBasis}, k::Int64, x) = vander!(dV, B, B.m-1, k, x)

vander(B::Union{PhyHermiteBasis, ProHermiteBasis}, m::Int64, k::Int64, x) = vander!(zeros(size(x,1),m+1), B, m, k, x)
vander(B::Union{CstPhyHermiteBasis, CstProHermiteBasis}, m::Int64, k::Int64, x) = vander!(zeros(size(x,1),m+1), B, m, k, x)
vander(B::Union{CstLinPhyHermiteBasis, CstLinProHermiteBasis}, m::Int64, k::Int64, x) = vander!(zeros(size(x,1),m+1), B, m, k, x)

# Return the k-th derivative of the Vandermonde matrix of the basis B evaluated at the samples x
vander(B::Basis, k::Int64, x) = vander!(zeros(size(x,1),B.m), B, k, x)

"""
iszerofeatureactive(B::MyBasis) = Bool

Determine if the zeroth order feature of the basis `B` is a constant (return `true`).
For instance, the first dimension is active in a feature of index [0 1] if `B = ProHermiteBasis`, but not active if `B = CstProHermiteBasis`.
"""
iszerofeatureactive(B::Union{PhyHermiteBasis, ProHermiteBasis}) = true
iszerofeatureactive(B::Union{CstPhyHermiteBasis, CstProHermiteBasis, CstLinPhyHermiteBasis, CstLinProHermiteBasis}) = false


# """
#     vander!(dV, B, maxi, k, x)
#
# Compute the Vandermonde matrix for the vector `x`
# """
# function vander!(dV, B::Basis, maxi::Int64, k::Int64, x)
#     N = size(x,1)
#     @assert size(dV) == (N, maxi+1) "Wrong dimension of the Vander matrix"
#
#     col0 = view(dV,:,1)
#     if k==0
#          fill!(col0, 1.0)
#     else
#          fill!(col0, 0.0)
#     end
#     if maxi == 0
#         return dV
#     end
#     dVshift = view(dV,:,2:maxi+1)
#     vander!(dVshift, FamilyScaledProHermite[maxi], k, x)
#      # .= vander(B.f[maxi+1], k, x)
#     return dV
# end
# @propagate_inbounds Base.getindex(B::Basis, i::Int64) = B.f[i]

#
# function CstPhyHermiteBasis(m::Int64; scaled::Bool = true)
#     if scaled == true
#         return Basis(ntuple(i -> i==1 ? FamilyProPolyHermite[1] : FamilyScaledPhyHermite[i-1], m+2))
#     else
#         return Basis(ntuple(i -> i==1 ? FamilyProPolyHermite[1] : FamilyPhyHermite[i-1], m+2))
#     end
# end
#
# function CstLinPhyHermiteBasis(m::Int64; scaled::Bool = true)
#     if scaled == true
#         return Basis(ntuple(i -> i<=2 ? FamilyProPolyHermite[i] : FamilyScaledPhyHermite[i-2], m+3))
#     else
#         return Basis(ntuple(i -> i==1 ? FamilyProPolyHermite[i] : FamilyPhyHermite[i-2], m+3))
#     end
# end
#
# function CstLinProHermiteBasis(m::Int64; scaled::Bool = true)
#     if scaled == true
#         return Basis(ntuple(i -> i<=2 ? FamilyProPolyHermite[i] : FamilyScaledProHermite[i-2], m+3))
#     else
#         return Basis(ntuple(i -> i==1 ? FamilyProPolyHermite[i] : FamilyProHermite[i-2], m+3))
#     end
# end



# function CstProHermiteBasis(m::Int64; scaled::Bool = false)
#     f = zeros(ParamFcn, m+2)
#     # f[1] = 1.0
#     f[1] = FamilyProPolyHermite[1]
#     for i=0:m
#         f[2+i] = ProHermite(i; scaled = scaled)
#     end
#     return Basis(f)
# end
#
# function CstLinPhyHermiteBasis(m::Int64; scaled::Bool = false)
#     f = zeros(ParamFcn, m+3)
#     # f[1] = 1.0
#     f[1] = FamilyProPolyHermite[1]
#     # f[1] = x
#     f[2] = FamilyProPolyHermite[2]
#     for i=0:m
#         f[3+i] = PhyHermite(i; scaled = scaled)
#     end
#     return Basis(f)
# end
#
# function CstLinProHermiteBasis(m::Int64; scaled::Bool = false)
#     f = zeros(ParamFcn, m+3)
#     # f[1] = 1.0
#     f[1] = FamilyProPolyHermite[1]
#     # f[1] = x
#     f[2] = FamilyProPolyHermite[2]
#     for i=0:m
#         f[3+i] = ProHermite(i; scaled = scaled)
#     end
#     return Basis(f)
# end

# (F::Array{ParamFcn,1})(x::T) where {T <: Real} = map!(fi->fi(x), zeros(T, size(F,1)), F)
# (B::Basis)(x::T) where {T<:Real} = B.f(x)

# (B::Basis{m})(x::T) where {m, T<:Real} = map!(fi->fi(x), zeros(T, m), B.f)

# @propagate_inbounds Base.getindex(F::Array{T,1}, i::Int) where {T<:ParamFcn} = getindex(F,i)
# @propagate_inbounds Base.setindex!(F::Array{T,1}, v::ParamFcn, i::Int) where {T<:ParamFcn} = setindex!(F,v,i)

# @propagate_inbounds Base.getindex(B::Basis, i::Int) = getindex(B.f,i)
# @propagate_inbounds Base.setindex!(B::Basis, v::ParamFcn, i::Int) = setindex!(B.f,v,i)
#


# function vander!(dV, B::Basis{m}, maxi::Int64, k::Int64, x) where {m}
#     N = size(x,1)
#     @assert size(dV) == (N, maxi+1) "Wrong dimension of the Vander matrix"
#
#     col0 = view(dV,:,1)
#      if k==0
#          fill!(col0, 0.0)
#      else
#          fill!(col0, )
#     @inbounds for i=1:maxi+1
#         col = view(dV,:,i)
#
#         if i==1
#             if k==0
#                 fill!(col, 1.0)
#             else
#                 fill!(col , 0.0)
#             end
#         else
#             if typeof(B.f[i]) <: Union{PhyHermite, ProHermite}
#                 # Store the k-th derivative of the i-th order Hermite polynomial
#                 derivative!(col, B.f[i], k, x)
#             elseif typeof(B[i]) <: Union{PhyPolyHermite, ProPolyHermite} && degree(B[i])>0
#                     Pik = derivative(B.f[i], k)
#                 #     # In practice the first component of the basis will be constant,
#                 #     # so this is very cheap to construct the derivative vector
#                     @. col = Pik(x)
#             end
#         end
#     end
#     return dV
# end
