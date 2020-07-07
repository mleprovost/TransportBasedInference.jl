

export ExpandedFunction

# ExpandedFunction decomposes a multi-dimensional function f:Rᴹ → R onto
# a basis of MultiFunctions ψ_α where c_α are scalar coefficients
# for each MultiFunction:
# f(x1, x2, ..., xM) = ∑_α c_α ψ_α(x1, x2, ..., xM)
# Nψ is the number of MultiFunctions used,
# M is the dimension of the input vector

struct ExpandedFunction{Nψ, M}
    ψ::Array{MultiFunction{M},1}
    c::Array{Float64,1}
    scaled::Bool
    function ExpandedFunction(ψ::Array{MultiFunction{M},1}, c::Array{Float64,1}; scaled::Bool=true) where {M}
            @assert size(ψ,1) == size(c,1) "The dimension of the basis functions don't
                                            match the number of coefficients"
            Nψ = size(ψ,1)
        return new{Nψ, M}(ψ, c, scaled)
    end
end


function (f::ExpandedFunction{Nψ, M})(x::Array{T,1}) where {Nψ, M, T <: Real}
    out = 0.0
    @inbounds for i=1:Nψ
        if f.c[i]!= 0.0
            out += f.c[i]*f.ψ[i](x)
        end
    end
    return out
end
