import Base: size, show, @propagate_inbounds


# Define the concept of basis functions, where each function is indexed by an integer
# For instance, (1, x, ψ0, ψ1,..., ψn) defines a basis where the index:
# 0 corresponds to the constant function
# 1 corresponds to the linear function
# n+2 corresponds to the n-th order physicist Hermite function

export Basis,
       vander!,
       vander,
       CstProHermite
       # CstPhyHermite, CstProHermite,
       # CstLinPhyHermite, CstLinProHermite,


struct Basis
    m::Int64
    # f::Tuple
    # function Basis(f::NTuple{N, ParamFcn}) where {N}
    #     return new(length(f), f)
    # end
end


function Base.show(io::IO, B::Basis)
    println(io,"Basis of "*string(B.m)*" functions: Constant -> "*string(B.m-2)*"th degree Probabilistic Hermite function")
    # for i=1:B.m
    #     println(io, B[i])
    # end
end


# Specialize method
function vander!(dV, B::Basis, maxi::Int64, k::Int64, x)
    N = size(x,1)
    @assert size(dV) == (N, maxi+1) "Wrong dimension of the Vander matrix"

    col0 = view(dV,:,1)
    if k==0
         fill!(col0, 1.0)
    else
         fill!(col0, 0.0)
    end
    if maxi == 0
        return dV
    end
    dVshift = view(dV,:,2:maxi+1)
    vander!(dVshift, FamilyScaledProHermite[maxi], k, x)
     # .= vander(B.f[maxi+1], k, x)
    return dV
end

vander!(dV, B::Basis, k::Int64, x) = vander!(dV, B, B.m-1, k, x)

vander(B::Basis, maxi::Int64, k::Int64, x) = vander!(zeros(size(x,1),maxi+1), B, maxi, k, x)
vander(B::Basis, k::Int64, x) = vander!(zeros(size(x,1),B.m), B, k, x)

@propagate_inbounds Base.getindex(B::Basis, i::Int64) = i==1 ? FamilyProPolyHermite[1] : FamilyScaledProHermite[i-1]


#
# function CstPhyHermite(m::Int64; scaled::Bool = false)
#     if scaled == true
#         return Basis(ntuple(i -> i==1 ? FamilyProPolyHermite[1] : FamilyScaledPhyHermite[i-1], m+2))
#     else
#         return Basis(ntuple(i -> i==1 ? FamilyProPolyHermite[1] : FamilyPhyHermite[i-1], m+2))
#     end
# end
#
# This is just a placeholder
CstProHermite(m::Int64) = Basis(m+2)

# (m::Int64) = ntuple(i -> i==1 ? FamilyProPolyHermite[1] : FamilyScaledProHermite[i-1], m+2)

#     if scaled == true
#         return Basis(ntuple(i -> i==1 ? FamilyProPolyHermite[1] : FamilyScaledProHermite[i-1], m+2))
#     else
#         return Basis(ntuple(i -> i==1 ? FamilyProPolyHermite[1] : FamilyProHermite[i-1], m+2))
#     end
# end
#
# function CstLinPhyHermite(m::Int64; scaled::Bool = false)
#     if scaled == true
#         return Basis(ntuple(i -> i<3 ? FamilyProPolyHermite[i] : FamilyScaledPhyHermite[i-2], m+3))
#     else
#         return Basis(ntuple(i -> i==1 ? FamilyProPolyHermite[i] : FamilyPhyHermite[i-2], m+3))
#     end
# end
#
# function CstLinProHermite(m::Int64; scaled::Bool = false)
#     if scaled == true
#         return Basis(ntuple(i -> i<3 ? FamilyProPolyHermite[i] : FamilyScaledProHermite[i-2], m+3))
#     else
#         return Basis(ntuple(i -> i==1 ? FamilyProPolyHermite[i] : FamilyProHermite[i-2], m+3))
#     end
# end
    #
    #
    # end
    # ntuple(i-> i = 1 ? FamilyProPolyHermite[1] : )
    # # f = zeros(ParamFcn, m+2)
    # # # f[1] = 1.0
    # f[1] = FamilyProPolyHermite[1]
    # for i=0:m
    #     f[2+i] = PhyHermite(i; scaled = scaled)
    # end
    # return Basis(f)

#
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
Base.size(B::Basis) = B.m


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
