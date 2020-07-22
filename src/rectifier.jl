

export  Rectifier,
        square, dsquare, d2square,
        softplus, dsoftplus, d2softplus, invsoftplus,
        explinearunit, dexplinearunit, d2explinearunit, invexplinearunit,
        inverse!, inverse,
        grad_x!, grad_x,
        hess_x!, hess_x,
        evaluate!

# Structure for continuous rectifier
# x->rec(x)
struct Rectifier
    T::String
end

square(x) = x^2
dsquare(x) = 2.0*x
d2square(x) = 2.0

# Softplus tools
softplus(x) = (log(1.0 + exp(-abs(log(2.0)*x))) + max(log(2.0)*x, 0.0))/log(2.0)
dsoftplus(x) = 1/(1 + exp(-log(2.0)*x))
d2softplus(x) = log(2.0)/(2.0*(1.0 + cosh(log(2.0)*x)))
invsoftplus(x) = min(log(exp(log(2.0)*x) - 1.0)/log(2.0), x)



explinearunit(x) = x < 0.0 ? exp(x) : x + 1.0
dexplinearunit(x) = x < 0.0 ? exp(x) : 1.0
d2explinearunit(x) = x < 0.0 ? exp(x) : 0.0
invexplinearunit(x) = x < 1.0 ? log(x) : x - 1.0

# Type of the rectifier should be in the following list:
# "squared", "exponential", "softplus", "explinearunit"


Rectifier() = Rectifier("softplus")

function (g::Rectifier)(x::T) where {T <: Real}
    if g.T=="squared"
        return square(x)
    elseif g.T=="exponential"
        return exp(x)
    elseif g.T=="softplus"
        return softplus(x)
    elseif g.T=="explinearunit"
        return explinearunit(x)
    end
end

function evaluate!(result::Array{T,1}, g::Rectifier, x::Array{T,1}) where {T <: Real}
    if g.T=="squared"
        map!(square, result, x)
        return result
    elseif g.T=="exponential"
        map!(exp, result, x)
        return result
    elseif g.T=="softplus"
        map!(softplus, result, x)
        return result
    elseif g.T=="explinearunit"
        map!(explinearunit, result, x)
        return result
    end
end

(g::Rectifier)(x::Array{T,1}) where {T <: Real} = evaluate!(zero(x), g, x)


function inverse(g::Rectifier, x::T) where {T <: Real}
    @assert x>=0 "Input to rectifier is negative"
    if g.T=="squared"
        error("squared rectifier is not invertible")
    elseif g.T=="exponential"
        return log(x)
    elseif g.T=="softplus"
        return invsoftplus(x)
    elseif g.T=="explinearunit"
        return invexplinearunit(x)
    end
end

function inverse!(result::Array{T,1}, g::Rectifier, x::Array{T,1}) where {T <: Real}
    @assert all(x .> 0) "Input to rectifier is negative"
    if g.T=="squared"
        error("squared rectifier is not invertible")
    elseif g.T=="exponential"
        map!(log, result, x)
        return result
    elseif g.T=="softplus"
        map!(invsoftplus, result, x)
        return result
    elseif g.T=="explinearunit"
        map!(invexplinearunit, result, x)
        return result
    end
end

inverse(g::Rectifier, x::Array{T,1}) where {T <: Real} = inverse!(zero(x), g, x)


function grad_x(g::Rectifier, x::T) where {T <: Real}
    if g.T=="squared"
        return dsquare(x)
    elseif g.T=="exponential"
        return exp(x)
    elseif g.T=="softplus"
        return dsoftplus(x)
    elseif g.T=="explinearunit"
        return dexplinearunit(x)
    end
end

function grad_x!(result::Array{T,1}, g::Rectifier, x::Array{T,1}) where {T <: Real}
    if g.T=="squared"
        map!(dsquare, result, x)
        return result
    elseif g.T=="exponential"
        map!(exp, result, x)
        return result
    elseif g.T=="softplus"
        map!(dsoftplus, result, x)
        return result
    elseif g.T=="explinearunit"
        map!(dexplinearunit, result, x)
        return result
    end
end

grad_x(g::Rectifier, x::Array{T,1}) where {T <: Real} = grad_x!(zero(x), g, x)


function hess_x(g::Rectifier, x::T) where {T <: Real}
    if g.T=="squared"
        return d2square(x)
    elseif g.T=="exponential"
        return exp(x)
    elseif g.T=="softplus"
        return d2softplus(x)
    elseif g.T=="explinearunit"
        return d2explinearunit(x)
    end
end

function hess_x!(result::Array{T,1}, g::Rectifier, x::Array{T,1}) where {T <: Real}
    if g.T=="squared"
        map!(d2square, result, x)
        return result
    elseif g.T=="exponential"
        map!(exp, result, x)
        return result
    elseif g.T=="softplus"
        map!(d2softplus, result, x)
        return result
    elseif g.T=="explinearunit"
        map!(d2explinearunit, result, x)
        return result
    end
end

hess_x(g::Rectifier, x::Array{T,1}) where {T <: Real} = hess_x!(zero(x), g, x)
