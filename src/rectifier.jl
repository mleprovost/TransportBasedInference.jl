

export Rectifier, inverse, grad_x, hess_x, evaluate!

# Structure for continuous rectifier
# x->rec(x)
struct Rectifier
    T::String
end

# function Rectifier(T)
#     @assert T ∈ ["squared", "exponential", "softplus", "explinearunit"] "Type of rectifier is not defined"
#     return Rectifier(T)
# end


# Type of the rectifier should be in the following list:
# "squared", "exponential", "softplus", "explinearunit"


Rectifier() = Rectifier("softplus")

function (g::Rectifier)(x::T) where {T <: Real}
    if g.T=="squared"
        return x^2
    elseif g.T=="exponential"
        return exp(x)
    elseif g.T=="softplus"
        a = log(2)
        return (log(1 + exp(-abs(a*x))) + max(a*x, 0))/a
    elseif g.T=="explinearunit"
        if x<0
            return exp(x)
        else
            return x+1.0
        end
    end
end
# @. g(x)#
(g::Rectifier)(x::Array{T,1}) where {T <: Real} = map!(g, zeros(T, size(x,1)), x)

evaluate!(result::Vector{T}, g::Rectifier, x::Array{T,1}) where {T <: Real} = map!(g, result, x)


function inverse(g::Rectifier, x::T) where {T <: Real}
    @assert x>=0 "Input to rectifier is negative"
    if g.T=="squared"
        error("squared rectifier is not invertible")
    elseif g.T=="exponential"
        return exp(x)
    elseif g.T=="softplus"
        a = log(2)
        return min(log(exp(a*x) - 1)/a, x)
    elseif g.T=="explinearunit"
        if x<1
            return log(x)
        else
            return x - 1.0
        end
    end
end

inverse(g::Rectifier, x::Array{T,1}) where {T <: Real} = map!(xi -> inverse(g,xi), zeros(T, size(x,1)), x)


function grad_x(g::Rectifier, x::T) where {T <: Real}
    if g.T=="squared"
        return 2*x
    elseif g.T=="exponential"
        return exp(x)
    elseif g.T=="softplus"
        a = log(2)
        return 1/(1 + exp(-abs(a*x)))
    elseif g.T=="explinearunit"
        if x<0
            return exp(x)
        else
            return 1.0
        end
    end
end

grad_x(g::Rectifier, x::Array{T,1}) where {T <: Real} = map!(xi -> grad_x(g,xi), zeros(T, size(x,1)), x)


function hess_x(g::Rectifier, x::T) where {T <: Real}
    if g.T=="squared"
        return 2.0
    elseif g.T=="exponential"
        return exp(x)
    elseif g.T=="softplus"
        a = log(2)
        return a/(2 + exp(a*x) + exp(-a*x))
    elseif g.T=="explinearunit"
        if x<0
            return exp(x)
        else
            return 0.0
        end
    end
end

hess_x(g::Rectifier, x::Array{T,1}) where {T <: Real} = map!(xi -> hess_x(g,xi), zeros(T, size(x,1)), x)
