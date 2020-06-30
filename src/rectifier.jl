

export Rectifier, inverse, gradient, hessian

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

function (r::Rectifier)(x)
    if r.T=="squared"
        return x^2
    elseif r.T=="exponential"
        return exp(x)
    elseif r.T=="softplus"
        a = log(2)
        return (log(1 + exp(-abs(a*x))) + max(a*x, 0))/a
    elseif r.T=="explinearunit"
        if x>0
            return exp(x)
        else
            return x+1.0
        end
    end
end

function inverse(r::Rectifier, x::Float64)
    @assert x>=0 "Input to rectifier is negative"
    if r.T=="squared"
        error("squared rectifier is not invertible")
    elseif r.T=="exponential"
        return exp(x)
    elseif r.T=="softplus"
        a = log(2)
        return min(log(exp(a*x) - 1)/a, x)
    elseif r.T=="explinearunit"
        if x<1
            return log(x)
        else
            return x - 1.0
        end
    end
end

function gradient(r::Rectifier, x::Float64)
    if r.T=="squared"
        return 2*x
    elseif r.T=="exponential"
        return exp(x)
    elseif r.T=="softplus"
        a = log(2)
        return 1/(1 + exp(-abs(a*x)))
    elseif r.T=="explinearunit"
        if x>0
            return exp(x)
        else
            return 1.0
        end
    end
end

function hessian(r::Rectifier, x::Float64)
    if r.T=="squared"
        return 2.0
    elseif r.T=="exponential"
        return exp(x)
    elseif r.T=="softplus"
        a = log(2)
        return a/(2 + exp(a*x) + exp(-a*x))
    elseif r.T=="explinearunit"
        if x<0
            return exp(x)
        else
            return 0.0
        end
    end
end
