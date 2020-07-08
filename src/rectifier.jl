

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

function (g::Rectifier)(x)
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

function inverse(g::Rectifier, x)
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

function gradient(g::Rectifier, x)
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

function hessian(g::Rectifier, x)
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
