
export  PhyPolyHermite, Cphy, degree, PhyPolyH, phyhermite_coeffmatrix,
        derivative, vander


# Create a structure to hold physicist Hermite polynomials as well as their first and second derivative
struct PhyPolyHermite{m} <: PolyHermite
    P::ImmutablePolynomial
    scale::Bool
end
# Hn(x)  = (-1)ⁿ*exp(x²)dⁿ/dxⁿ exp(-x²)
# Hn′(x) = 2n*Hn-1(x)
# Hn″(x) = 2n*(n-1)*Hn-1(x)

Cphy(m::Int64) =sqrt(gamma(m+1) * 2^m * sqrt(π))
Cphy(P::PhyPolyHermite{m}) where {m} = Cphy(m)

degree(P::PhyPolyHermite{m}) where {m} = m

# Adapted https://people.sc.fsu.edu/~jburkardt/m_src/hermite_polynomial/h_polynomial_coefficients.m

#    Input, integer N, the highest order polynomial to compute.
#    Note that polynomials 0 through N will be computed.
#
#    Output, real C(1:N+1,1:N+1), the coefficients of the Hermite
#    polynomials.

# Recurrence relation:
# Hn+1 = 2*x*Hn - 2*n*Hn-1
function phyhermite_coeffmatrix(m::Int64)

    if m < 0
        return Float64[]
    end

    coeff = zeros(m+1, m+1)
    coeff[1, 1] = 1.0

    if m == 0
        return coeff
    end

    coeff[2, 2] = 2.0

    if m==1
        return coeff
    end

    for i  = 2:m
        coeff[i+1, 1]      = - 2.0 * (i - 1) * coeff[i-1, 1]
        coeff[i+1, 2:i-1]  =   2.0 *           coeff[i  , 1:i-2] -
                               2.0 * (i - 1) * coeff[i-1, 2:i-1]
        coeff[i+1, i]      =   2.0 *           coeff[i  , i-1]
        coeff[i+1, i+1]    =   2.0 *           coeff[i  , i]
    end
    return coeff
end


const PhyPolyH = phyhermite_coeffmatrix(20)

function PhyPolyHermite(m::Int64;scaled::Bool= false)
    @assert m>=0 "The order of the polynomial should be >=0"
    if scaled ==false
            return PhyPolyHermite{m}(ImmutablePolynomial(view(PhyPolyH,m+1,1:m+1)), scaled)
    else
        C = 1/Cphy(m)
            return PhyPolyHermite{m}(ImmutablePolynomial(C*view(PhyPolyH,m+1,1:m+1)), scaled)
    end
end

(P::PhyPolyHermite{m})(x) where {m} = P.P(x)

const FamilyPhyHermite = map(i->PhyPolyHermite(i),0:20)

# Compute the k-th derivative of a physicist Hermite polynomial
function derivative(P::PhyPolyHermite{m}, k::Int64; scaled::Bool = false) where {m}
    @assert k>=0 "This function doesn't compute anti-derivatives of Hermite polynomials"
    if m>=k
        factor = 2^k*exp(loggamma(m+1) - loggamma(m+1-k))
        if scaled == false
            return ImmutablePolynomial(factor*PhyPolyH[m-k+1,1:m-k+1])
        else
            return ImmutablePolynomial(sqrt(factor)*PhyPolyH[m-k+1,1:m-k+1])
        end
    else
        return ImmutablePolynomial((0.0))
    end
end


# For next week, the question is should we use the Family of polynomials  evaluate at the samples and just multiply by the constant
# seems to be faster !! than recomputing the derivative

function vander(m::Int64, k::Int64, x; scaled::Bool=false)
    N = size(x,1)
    dV = zeros(N, m+1)

    @inbounds for i=0:m

        col .= Famil

    end

end
