export SeparableFcn, AddedSeparableFcn, ui,
 uk, σscale, D!, D, H!, H

abstract type SeparableFcn end

## Need tools to identify

struct AddedSeparableFcn{T <: Tuple} <: SeparableFcn
    list::T
end

function Base.show(io::IO, Σg::AddedSeparableFcn)
    println(io, "AddedSeparableFcn:")
    for g in Σg.list
        println(io, "  $g")
    end
end

# """
#     g₁::SeparableFcn + g₂::SeparableFcn
#
# Add the separable functions so that `(g₁ + g₂)(z) = g₁(z) + g₂(z)`.
# """


+(g::SeparableFcn, Σg::AddedSeparableFcn) = AddedSeparableFcn((g, Σg.list...))
+(Σg::AddedSeparableFcn, g::SeparableFcn) = AddedSeparableFcn((Σg.list..., g))
function +(Σg₁::AddedSeparableFcn, Σg₂::AddedSeparableFcn)
    AddedSeparableFcn((Σg₁..., Σg₂...))
end

+(g::SeparableFcn...) = AddedSeparableFcn(g)

(Σg::AddedSeparableFcn)(z) = mapreduce(gi->gi(z),+, Σg.list)


# (+)(f1::T, f2::U) where {T<:SeparableFcn, U<:SeparableFcn}  =

### Generate parameterization for the i-th component of a lower triangular map U
#
#
#  U: z =(z1, z2,...,zn) → [U1(z1);
#                          U2(z1, z2);
#                            ...
#                          Un(z1, z2, ..., zn)]
#
#

# We use separable maps RadialMapComponent(z1,...,zk) = Σi=1,k-1 ui(z1,..,zi) + uk(z1,..,zk)
# uk has a different structure from ui to ensure the monotonocity condition
# of the triangular map
struct ui <:SeparableFcn
        p::Int64
        ξi::Array{Float64,1}
        σi::Array{Float64,1}
        coeffi::Array{Float64,1}
        # f::Array{ParamFcn,1}

        # Inner constructor
        function ui(p::Int64, ξi::Array{Float64, 1}, σi::Array{Float64, 1}, coeffi::Array{Float64, 1})

                if p==-1
                        # p==-1 , ui(z) = 0.0
                        @assert size(ξi,1)==0 "Size of ξi does not match the number of RBFs p"
                        @assert size(σi,1)==0 "Size of σi does not match the number of RBFs p"
                        @assert size(coeffi,1)==0 "Size of coeffi does not match the number of RBFs p"
                else
                        # p==0 ui(z) = az, p>0 ui(z) = a1z + p rbfs
                        @assert size(ξi,1)==p "Size of ξi does not match the number of RBFs p"
                        @assert size(σi,1)==p "Size of σi does not match the number of RBFs p"
                        @assert size(coeffi,1)==p+1 "Size of coeffi does not match the number of RBFs p"
                end
                new(p, ξi, σi, coeffi)
        end
end

function ui(p::Int64)
        if p==-1
                return ui(p, Float64[], Float64[], Float64[])
        else
                return ui(p, zeros(p), ones(p), zeros(p+1))
        end
end

function (u::ui)(z::Real)
        p = u.p
if p==-1
        return 0.0
else
        out = u.coeffi[1]*z
        if p>0
                @inbounds for j=1:p
                out +=rbf(u.ξi[j], u.σi[j])(z)*u.coeffi[j+1]
                end
        end
        return out
end
end


# Compute derivative of a ui function
function D!(u::ui, z::Real)
        p = u.p
if p==-1
        return 0.0
else
        out = u.coeffi[1]
        if p>0
                for j=1:p
                @inbounds out +=rbf′(u.ξi[j], u.σi[j])(z)*u.coeffi[j+1]
                end
        end
        return out
end
end

D(u::ui) =  z-> D!(u, z)

#### Structure for the final component of RadialMapComponent

# uk(z) = c + Σj=0, p+1 ukj ψⱼ(z), p.36 Couplings for nonlinear ensemble filtering

struct uk <:SeparableFcn
        p::Int64
        ξk::Array{Float64,1}
        σk::Array{Float64,1}
        coeffk::Array{Float64,1}

        function uk(p::Int64, ξk::Array{Float64, 1}, σk::Array{Float64, 1}, coeffk::Array{Float64, 1})
                if p==-1
                @assert size(ξk,1)==0 "Size of ξk doesn't match the number of RBFs p"
                @assert size(σk,1)==0 "Size of σk doesn't match the number of RBFs p"
                @assert size(coeffk,1)==0 "Size of coeffk doesn't match the number of RBFs p"
                elseif p==0
                @assert size(ξk,1)==0 "Size of ξk doesn't match the number of RBFs p"
                @assert size(σk,1)==0 "Size of σk doesn't match the number of RBFs p"
                @assert size(coeffk,1)==2 "Size of coeffk doesn't match the number of RBFs p"
                else
                @assert size(ξk,1)==p+2 "Size of ξk doesn't match the number of RBFs p"
                @assert size(σk,1)==p+2 "Size of σk doesn't match the number of RBFs p"
                @assert size(coeffk,1)==p+3 "Size of coeffk doesn't match the number of RBFs p"
                end
                return new(p, ξk, σk, coeffk)
        end
end

# uk is an affine function if p=0

function uk(p::Int64)
        if p==-1
                # uk(z) = z
                return uk(p, Float64[], Float64[], Float64[])
        elseif p==0
                # uk(z) = a[1] + a[2]*z
                return uk(p, Float64[], Float64[], zeros(2))
        else
                return uk(p, zeros(p+2), ones(p+2), zeros(p+3))
        end
end


function (u::uk)(z::T) where {T<:Real}
        p = u.p
        if p==-1
                return z
        elseif p==0
                out = u.coeffk[1] + u.coeffk[2]*z
                return out
        else
                out = u.coeffk[1]
                out +=ψ₀(u.ξk[1], u.σk[1])(z)*u.coeffk[2]
                out +=ψpp1(u.ξk[p+2], u.σk[p+2])(z)*u.coeffk[p+3]

                @inbounds for j=2:p+1
                out +=ψj(u.ξk[j], u.σk[j])(z)*u.coeffk[j+1]
                end
                return out
        end
end


# Compute derivative of a uk function
function D!(u::uk, z::Real)
        p = u.p

        if p==-1
                return 1.0
        elseif p==0
        # Linear function
                return u.coeffk[2]
        else
                out = ψ₀′(u.ξk[1], u.σk[1])(z)*u.coeffk[2]
                out += ψpp1′(u.ξk[p+2], u.σk[p+2])(z)*u.coeffk[p+3]

                @inbounds for j=2:p+1
                out +=rbf(u.ξk[j], u.σk[j])(z)*u.coeffk[j+1]
                end
                return out
        end
end

D(u::uk) =  z-> D!(u, z)


# Compute hessian of a uk function
function H!(u::uk, z::Real)
        p = u.p
        if p<1
                return 0.0
        else
                # ψ₀'' = -rbf
                out =rbf(u.ξk[1], u.σk[1])(z)*(-u.coeffk[2])
                # ψpp1′′
                out +=rbf(u.ξk[p+2], u.σk[p+2])(z)*u.coeffk[p+3]

                @inbounds for j=2:p+1
                out +=rbf′(u.ξk[j], u.σk[j])(z)*u.coeffk[j+1]
                end
                return out
        end
end

H(u::uk) =  z-> H!(u, z)

function σscale(ξ::Array{Float64,1}, γ::Float64)
        p = size(ξ,1)
        σ = zero(ξ)
        # Special treatment for the edges
        # ξ[0]=ξ[1] & ξ[p+1]=ξ[p]
        σ[1] = ξ[2]-ξ[1]
        σ[end] = ξ[end]-ξ[end-1]
        for j=2:p-1
        @inbounds σ[j] = 0.5*(ξ[j+1]-ξ[j-1])
        end

        rmul!(σ, γ)
        return σ
end
