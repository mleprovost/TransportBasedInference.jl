import Base: size, show, @propagate_inbounds


# Define the concept of basis functions, where each function is indexed by an integer
# For instance, (1, x, ψ0, ψ1,..., ψn) defines a basis where the index:
# 0 corresponds to the constant function
# 1 corresponds to the linear function
# n+2 corresponds to the n-th order physicist Hermite function

export Basis,
       vander!,
       vander,
       CstProHermite,
       CstPhyHermite,
       CstLinPhyHermite,
       CstLinProHermite

"""
   $(TYPEDEF)

A structure to hold a basis of functions
For instance, (1, x, ψ0, ψ1,..., ψn) defines a basis where the index:
0 corresponds to the constant function
1 corresponds to the linear function
n+2 corresponds to the n-th order physicist Hermite function

## Fields
$(TYPEDFIELDS)

## Constructors
- `Basis(m)`

"""

abstract type Basis end

Base.size(B::Basis) = B.m

# A particular feature basis is a subtype of Basis.
# To define a new basis, you need to provide the following routines:
# Base.show(io::IO, B::MyBasis) (optional, but desired)
# @propagate_inbounds Base.getindex(B::MyBasis, i::Int64)
# vander!(dV, B::MyBasis, maxi::Int64, k::Int64, x)

struct CstProHermite <: Basis
    m::Int64
end

@propagate_inbounds Base.getindex(B::CstProHermite, i::Int64) = i==1 ? FamilyProPolyHermite[1] : FamilyScaledProHermite[i-1]

function Base.show(io::IO, B::CstProHermite)
    println(io,"Basis of "*string(B.m)*" functions: Constant, 0th -> "*string(B.m-2)*"th degree Probabilistic Hermite function")
end


struct CstPhyHermite <: Basis
    m::Int64
end

@propagate_inbounds Base.getindex(B::CstPhyHermite, i::Int64) = i==1 ? FamilyProPolyHermite[1] : FamilyScaledPhyHermite[i-1]

function Base.show(io::IO, B::CstPhyHermite)
    println(io,"Basis of "*string(B.m)*" functions: Constant, 0th -> "*string(B.m-2)*"th degree Physicist Hermite function")
end

struct CstLinProHermite <: Basis
    m::Int64
end

@propagate_inbounds Base.getindex(B::CstLinProHermite, i::Int64) = i==1 ? FamilyProPolyHermite[1] : i==2 ? FamilyProPolyHermite[2] : FamilyScaledProHermite[i-2]

function Base.show(io::IO, B::CstLinProHermite)
    println(io,"Basis of "*string(B.m)*" functions: Constant, Linear, 0th -> "*string(B.m-3)*"th degree Probabilistic Hermite function")
end

struct CstLinPhyHermite <: Basis
    m::Int64
end

@propagate_inbounds Base.getindex(B::CstLinPhyHermite, i::Int64) = i==1 ? FamilyProPolyHermite[1] : i==2 ? FamilyProPolyHermite[2] : FamilyScaledPhyHermite[i-2]

function Base.show(io::IO, B::CstLinPhyHermite)
    println(io,"Basis of "*string(B.m)*" functions: Constant, Linear 0th -> "*string(B.m-3)*"th degree Physicist Hermite function")
end



# Implementation of vander! for the different bases
"""
    vander!(dV, B::T, m::Int64, k::Int64, x) where T <: Basis

Compute the Vandermonde matrix of the basis `B` for the vector `x`up to the m-th feature of the basis
"""

function vander!(dV, B::CstProHermite, m::Int64, k::Int64, x)
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
    vander!(dV, B::CstPhyHermite, maxi::Int64, k::Int64, x)

Compute the Vandermonde matrix for the vector `x`
"""
function vander!(dV, B::CstPhyHermite, m::Int64, k::Int64, x)
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
    vander!(dV, B::CstLinProHermite, maxi::Int64, k::Int64, x)

Compute the Vandermonde matrix for the vector `x`
"""
function vander!(dV, B::CstLinProHermite, m::Int64, k::Int64, x)
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
    vander!(dV, B::CstLinPhyHermite, maxi::Int64, k::Int64, x)

Compute the Vandermonde matrix for the vector `x`
"""
function vander!(dV, B::CstLinPhyHermite, m::Int64, k::Int64, x)
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

vander!(dV, B::Union{CstPhyHermite, CstProHermite}, k::Int64, x) = vander!(dV, B, B.m-1, k, x)
vander!(dV, B::Union{CstLinPhyHermite, CstLinProHermite}, k::Int64, x) = vander!(dV, B, B.m-1, k, x)

vander(B::Union{CstPhyHermite, CstProHermite}, m::Int64, k::Int64, x) = vander!(zeros(size(x,1),m+1), B, m, k, x)
vander(B::Union{CstLinPhyHermite, CstLinProHermite}, m::Int64, k::Int64, x) = vander!(zeros(size(x,1),m+1), B, m, k, x)
vander(B::Basis, k::Int64, x) = vander!(zeros(size(x,1),B.m), B, k, x)

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
# function CstPhyHermite(m::Int64; scaled::Bool = true)
#     if scaled == true
#         return Basis(ntuple(i -> i==1 ? FamilyProPolyHermite[1] : FamilyScaledPhyHermite[i-1], m+2))
#     else
#         return Basis(ntuple(i -> i==1 ? FamilyProPolyHermite[1] : FamilyPhyHermite[i-1], m+2))
#     end
# end
#
# function CstLinPhyHermite(m::Int64; scaled::Bool = true)
#     if scaled == true
#         return Basis(ntuple(i -> i<=2 ? FamilyProPolyHermite[i] : FamilyScaledPhyHermite[i-2], m+3))
#     else
#         return Basis(ntuple(i -> i==1 ? FamilyProPolyHermite[i] : FamilyPhyHermite[i-2], m+3))
#     end
# end
#
# function CstLinProHermite(m::Int64; scaled::Bool = true)
#     if scaled == true
#         return Basis(ntuple(i -> i<=2 ? FamilyProPolyHermite[i] : FamilyScaledProHermite[i-2], m+3))
#     else
#         return Basis(ntuple(i -> i==1 ? FamilyProPolyHermite[i] : FamilyProHermite[i-2], m+3))
#     end
# end



# function CstProHermite(m::Int64; scaled::Bool = false)
#     f = zeros(ParamFcn, m+2)
#     # f[1] = 1.0
#     f[1] = FamilyProPolyHermite[1]
#     for i=0:m
#         f[2+i] = ProHermite(i; scaled = scaled)
#     end
#     return Basis(f)
# end
#
# function CstLinPhyHermite(m::Int64; scaled::Bool = false)
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
# function CstLinProHermite(m::Int64; scaled::Bool = false)
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
