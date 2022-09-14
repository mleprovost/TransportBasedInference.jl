export  Rectifier,
        square, dsquare, d2square,
        softplus, dsoftplus, d2softplus, invsoftplus,
        explinearunit, dexplinearunit, d2explinearunit, invexplinearunit,
        inverse!, inverse, vinverse,
        grad_x!, grad_x, vgrad_x,
        grad_x_logeval, grad_x_logeval!, vgrad_x_logeval,
        hess_x_logeval, hess_x_logeval!, vhess_x_logeval,
        hess_x!, hess_x, vhess_x,
        evaluate!, vevaluate


"""
$(TYPEDEF)

This structure defines a continuous rectifier g,
i.e. a positive and monotonically increasing function
(e.g. square function, exponential , softplus, explinearunit).

## Fields

$(TYPEDFIELDS)

"""
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

function (g::Rectifier)(x)
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

function evaluate!(result, g::Rectifier, x)
    @assert size(result,1) == size(x,1) "Dimension of result and x don't match"
    if g.T=="squared"
        vmap!(square, result, x)
        return result
    elseif g.T=="exponential"
        vmap!(exp, result, x)
        return result
    elseif g.T=="softplus"
        vmap!(softplus, result, x)
        return result
    elseif g.T=="explinearunit"
        vmap!(explinearunit, result, x)
        return result
    end
end

vevaluate(g::Rectifier, x) = evaluate!(zero(x), g, x)

function inverse(g::Rectifier, x)
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

function inverse!(result, g::Rectifier, x)
    @assert all(x .> 0) "Input to rectifier is negative"
    @assert size(result,1) == size(x,1) "Dimension of result and x don't match"
    if g.T=="squared"
        error("squared rectifier is not invertible")
    elseif g.T=="exponential"
        vmap!(log, result, x)
        return result
    elseif g.T=="softplus"
        vmap!(invsoftplus, result, x)
        return result
    elseif g.T=="explinearunit"
        vmap!(invexplinearunit, result, x)
        return result
    end
end

vinverse(g::Rectifier, x)  = inverse!(zero(x), g, x)


function grad_x(g::Rectifier, x)
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


function grad_x!(result, g::Rectifier, x)
    @assert size(result,1) == size(x,1) "Dimension of result and x don't match"
    if g.T=="squared"
        vmap!(dsquare, result, x)
        return result
    elseif g.T=="exponential"
        vmap!(exp, result, x)
        return result
    elseif g.T=="softplus"
        vmap!(dsoftplus, result, x)
        return result
    elseif g.T=="explinearunit"
        vmap!(dexplinearunit, result, x)
        return result
    end
end

vgrad_x(g::Rectifier, x) = grad_x!(zero(x), g, x)

""" Compute g′(x)/g(x) i.e d/dx log(g(x))"""
function grad_x_logeval(g::Rectifier, x::T) where {T <: Real}
    if g.T=="squared"
        return dsquare(x)/square(x)
    elseif g.T=="exponential"
        return 1.0
    elseif g.T=="softplus"
        return dsoftplus(x)/softplus(x)
    elseif g.T=="explinearunit"
        return dexplinearunit(x)/explinearunit(x)
    end
end

function grad_x_logeval!(result, g::Rectifier, x)
    @assert size(result,1) == size(x,1) "Dimension of result and x don't match"
    if g.T=="squared"
        vmap!(xi->dsquare(xi)/square(xi), result, x)
        return result
    elseif g.T=="exponential"
        vmap!(1.0, result, x)
        return result
    elseif g.T=="softplus"
        vmap!(xi->dsoftplus(xi)/softplus(xi), result, x)
        return result
    elseif g.T=="explinearunit"
        vmap!(xi->dexplinearunit(xi)/explinearunit(xi), result, x)
        return result
    end
end

vgrad_x_logeval(g::Rectifier, x) = grad_x_logeval!(zero(x), g, x)


# Compute (g″(x)g(x)-g′(x)^2)/g(x) i.e d^2/dx^2 log(g(x))
function hess_x_logeval(g::Rectifier, x::T) where {T <: Real}
    if g.T=="squared"
        return (d2square(x)*square(x) - dsquare(x)^2)/square(x)^2
    elseif g.T=="exponential"
        return 0.0
    elseif g.T=="softplus"
        return (d2softplus(x)*softplus(x) - dsoftplus(x)^2)/softplus(x)^2
    elseif g.T=="explinearunit"
        return (d2explinearunit(x)*explinearunit(x) - dexplinearunit(x)^2)/explinearunit(x)^2
    end
end

function hess_x_logeval!(result, g::Rectifier, x)
    @assert size(result,1) == size(x,1) "Dimension of result and x don't match"
    if g.T=="squared"
        vmap!(xi->(d2square(xi)*square(xi) - dsquare(xi)^2)/square(xi)^2, result, x)
        return result
    elseif g.T=="exponential"
        vmap!(0.0, result, x)
        return result
    elseif g.T=="softplus"
        vmap!(xi->(d2softplus(xi)*softplus(xi) - dsoftplus(xi)^2)/softplus(xi)^2, result, x)
        return result
    elseif g.T=="explinearunit"
        vmap!(xi->(d2explinearunit(xi)*explinearunit(xi) - dexplinearunit(xi)^2)/explinearunit(xi)^2, result, x)
        return result
    end
end

vhess_x_logeval(g::Rectifier, x) = hess_x_logeval!(zero(x), g, x)

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

function hess_x!(result, g::Rectifier, x)
    @assert size(result,1) == size(x,1) "Dimension of result and x don't match"
    if g.T=="squared"
        vmap!(d2square, result, x)
        return result
    elseif g.T=="exponential"
        vmap!(exp, result, x)
        return result
    elseif g.T=="softplus"
        vmap!(d2softplus, result, x)
        return result
    elseif g.T=="explinearunit"
        vmap!(d2explinearunit, result, x)
        return result
    end
end

vhess_x(g::Rectifier, x) = hess_x!(zero(x), g, x)
