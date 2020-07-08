

export ExpandedFunction

# ExpandedFunction decomposes a multi-dimensional function f:Rᴹ → R onto
# a basis of MultiFunctions ψ_α where c_α are scalar coefficients
# for each MultiFunction:
# f(x1, x2, ..., xNx) = ∑_α c_α ψ_α(x1, x2, ..., xNx)
# Nψ is the number of MultiFunctions used,
# Nx is the dimension of the input vector x

struct ExpandedFunction{Nψ, m, Nx}
    ψ::Array{MultiFunction{m, Nx},1}
    c::Array{Float64,1}
    scaled::Bool
    function ExpandedFunction(ψ::Array{MultiFunction{m, Nx},1}, c::Array{Float64,1}; scaled::Bool=true) where {m, Nx}
            @assert size(ψ,1) == size(c,1) "The dimension of the basis functions don't
                                            match the number of coefficients"
            Nψ = size(ψ,1)
        return new{Nψ, m, Nx}(ψ, c, scaled)
    end
end

# Naive approach of taking the full product
function (f::ExpandedFunction{Nψ, m, Nx})(x::Array{T,1}) where {Nψ, m, Nx, T <: Real}
    @assert Nx==size(x,1) "Wrong dimension of the input vector"
    out = 0.0
    @inbounds for i=1:Nψ
        if f.c[i] != 0.0
            out += f.c[i]*(f.ψ[i])(x)
        end
    end
    return out
end

# function
