export ParamFcn, AddedParamFcn, reduction, null, Null, constant, Cst, linear, Lin, rbf, rbf′,
        ψ₀, ψj, ψpp1, ψ₀′, ψpp1′

import Base: +, *, show, zero, size, getindex, setindex!

# Type to hold 1D functions
abstract type ParamFcn end


zero(::Type{T}) where {T <: ParamFcn} = null()


## Need tools to identify

# struct AddedParamFcn{T <: Tuple} <: ParamFcn
#     list::T
# end

struct AddedParamFcn{T <: Tuple} <: ParamFcn
    list::T
end
function Base.show(io::IO, Σg::AddedParamFcn)
    println(io, "AddedParamFcn:")
    for g in Σg.list
        println(io, "  $g")
    end
end

# """
#     g₁::ParamFcn + g₂::ParamFcn
#
# Add the parameterized functions so that `(g₁ + g₂)(z) = g₁(z) + g₂(z)`.
# """


+(g::ParamFcn, Σg::AddedParamFcn) = AddedParamFcn((g, Σg.list...))
+(Σg::AddedParamFcn, g::ParamFcn) = AddedParamFcn((Σg.list..., g))
function +(Σg₁::AddedParamFcn, Σg₂::AddedParamFcn)
    AddedParamFcn((Σg₁..., Σg₂...))
end

+(g::ParamFcn...) = AddedParamFcn(g)

(Σg::AddedParamFcn)(z::T) where {T<:Real} = map(g->g(z), Σg.list)
 # mapreduce(g->g(z),+, Σg.list;init=0.0)#
reduction(Σg::AddedParamFcn, z::T) where {T<:Real} = mapreduce(g->g(z),+, Σg.list;init=0.0)



# Add tools to access elements of an AddedParamFcn
# Base.size(A::AddedParamFcn{T}) where {T<:Tuple} = size(A.list)
# Base.getindex(A::AddedParamFcn{T}, i::Vararg{Int,N}) where {T,N} = A.list[i...]
# setindex!(::AddedParamFcn{T}, ::S, ::Int64)

# (*)(α::T)(f::F)(z::U) where {T<:Real, F<:paramfunction, U<:Real} = α*f(z)
struct null <:ParamFcn
end

const Null = null()

(C::null)(z::T) where {T<:Real} = 0.0

function Base.show(io::IO, N::null)
println(io,"Zero")
end

struct constant <:ParamFcn
end

const Cst = constant()

(C::constant)(z::T) where {T<:Real} = 1.0

function Base.show(io::IO, C::constant)
println(io,"Constant")
end

struct linear <:ParamFcn
end

const Lin = linear()

(L::linear)(z::T) where {T<:Real} = z

function Base.show(io::IO, L::linear)
println(io,"Linear")
end

struct rbf <:ParamFcn
        μ::Float64
        σ::Float64
end


function Base.show(io::IO, N::rbf)
println(io,"Gaussian kernel with mean $(N.μ) and std $(N.σ)")
end

(N::rbf)(z::T)  where {T<:Real} = 1/(N.σ*√(2*π))*exp(-(0.5/N.σ^2)*(z-N.μ)^2)


struct rbf′ <:ParamFcn
        μ::Float64
        σ::Float64
end


function Base.show(io::IO, N::rbf′)
println(io,"Derivative of a Gaussian kernel with mean $(N.μ) and std $(N.σ)")
end

(N::rbf′)(z::T)  where {T<:Real} = (N.μ-z)/(N.σ^3*√(2*π))*exp(-(0.5/N.σ^2)*(z-N.μ)^2)


## Notation of Couplings for nonlinear ensemble filtering, Spantini, Baptista, Marzouk
struct ψ₀ <:ParamFcn
        ξ₀::Float64
        σ₀::Float64
        function ψ₀(ξ₀::Float64, σ₀::Float64)
            @assert σ₀ >= 0.0 "std must be >= 0"
            new(ξ₀, σ₀)
        end
        # function ψ₀(ξ₀::T, σ₀::S) where {T<:Real, S<:Real}
        #     @assert σ₀ >= 0.0 "std must be >= 0"
        #     new(ξ₀, σ₀)
        # end
end


function Base.show(io::IO, F::ψ₀)
println(io,"ψ₀ function with mean $(F.ξ₀) and std $(F.σ₀)")
end


function (F::ψ₀)(z::T) where {T<:Real}
    Δ₀ = (1/(F.σ₀*√2))*(z-F.ξ₀)
    return 0.5*((z-F.ξ₀)*(1-erf(Δ₀))-F.σ₀*√(2/π)*exp(-Δ₀^2))
end


## Notation of Couplings for nonlinear ensemble filtering, Spantini, Baptista, Marzouk
struct ψj <:ParamFcn
        ξⱼ::Float64
        σⱼ::Float64
        function ψj(ξⱼ::Float64, σⱼ::Float64)
            @assert σⱼ >= 0.0 "std must be >= 0"
            new(ξⱼ, σⱼ)
        end
end

function Base.show(io::IO, F::ψj)
println(io,"ψj function with mean $(F.ξⱼ) and std $(F.σⱼ)")
end

function (F::ψj)(z::T) where {T<:Real}
    Δⱼ = (1/(F.σⱼ*√2))*(z-F.ξⱼ)
    return 0.5*(1+erf(Δⱼ))
end


## Notation of Couplings for nonlinear ensemble filtering, Spantini, Baptista, Marzouk
#  ψₚ₊₁
struct ψpp1 <:ParamFcn
        ξpp1::Float64
        σpp1::Float64
        function ψpp1(ξpp1::Float64, σpp1::Float64)
            @assert σpp1 >= 0.0 "std must be >= 0"
            new(ξpp1, σpp1)
        end
end


function Base.show(io::IO, F::ψpp1)
println(io,"ψpp1 function with mean $(F.ξpp1) and std $(F.σpp1)")
end

function (F::ψpp1)(z::T) where {T<:Real}
    Δpp1 = (1/(F.σpp1*√2))*(z-F.ξpp1)
    return 0.5*((z-F.ξpp1)*(1+erf(Δpp1)) +F.σpp1*√(2/π)*exp(-Δpp1^2))
end

# " Derivatives ψ′ of ψ for j=0:p+1"
## Notation of Couplings for nonlinear ensemble filtering, Spantini, Baptista, Marzouk
#####  ψ₀′
struct ψ₀′ <:ParamFcn
        ξ₀::Float64
        σ₀::Float64
        function ψ₀′(ξ₀::Float64, σ₀::Float64)
            @assert σ₀ >= 0.0 "std must be >= 0"
            new(ξ₀, σ₀)
        end
end


function Base.show(io::IO, F::ψ₀′)
println(io,"ψ₀′ function with mean $(F.ξ₀) and std $(F.σ₀)")
end


function (F::ψ₀′)(z::T) where {T<:Real}
    Δ₀ = (1/(F.σ₀*√2))*(z-F.ξ₀)
    return 0.5*(1-erf(Δ₀))
end

### ψₚ₊₁′
struct ψpp1′ <:ParamFcn
        ξpp1::Float64
        σpp1::Float64
        function ψpp1′(ξpp1::Float64, σpp1::Float64)
            @assert σpp1 >= 0.0 "std must be >= 0"
            new(ξpp1, σpp1)
        end
end

function Base.show(io::IO, F::ψpp1′)
println(io,"ψpp1′ function with mean $(F.ξpp1) and std $(F.σpp1)")
end

function (F::ψpp1′)(z::T) where {T<:Real}
    Δpp1 = (1/(F.σpp1*√2))*(z-F.ξpp1)
    return 0.5*(1+erf(Δpp1))
end
