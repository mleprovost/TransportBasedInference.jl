
export  PhyPolyHermite, Cphy, degree, PhyPolyH, phyhermite_coeffmatrix,
        gradient, hessian


# Create a structure to hold physicist Hermite polynomials as well as their first and second derivative
struct PhyPolyHermite{m} <: Hermite
    P::ImmutablePolynomial
    Pprime::ImmutablePolynomial
    Ppprime::ImmutablePolynomial
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
        if m==0
            return PhyPolyHermite{m}(ImmutablePolynomial(1.0),
                                 ImmutablePolynomial(0.0),
                                 ImmutablePolynomial(0.0),
                                 scaled)
        elseif m==1
            return PhyPolyHermite{m}(ImmutablePolynomial(view(PhyPolyH,m+1,1:m+1)),
                                 ImmutablePolynomial(2*m*view(PhyPolyH,m,1:m)),
                                 ImmutablePolynomial(0.0),
                                 scaled)
        else
            return PhyPolyHermite{m}(ImmutablePolynomial(view(PhyPolyH,m+1,1:m+1)),
                                 ImmutablePolynomial(2*m*view(PhyPolyH,m,1:m)),
                                 ImmutablePolynomial(4*m*(m-1)*view(PhyPolyH,m-1,1:m-1)),
                                 scaled)
        end
    else
        C = 1/Cphy(m)
        if m==0
            return PhyPolyHermite{m}(ImmutablePolynomial(C),
                                 ImmutablePolynomial(0.0),
                                 ImmutablePolynomial(0.0),
                                 scaled)
        elseif m==1
            return PhyPolyHermite{m}(ImmutablePolynomial(C*view(PhyPolyH,m+1,1:m+1)),
                                 ImmutablePolynomial(C*2*m*view(PhyPolyH,m,1:m)),
                                 ImmutablePolynomial(0.0),
                                 scaled)
        else
            return PhyPolyHermite{m}(ImmutablePolynomial(C*view(PhyPolyH,m+1,1:m+1)),
                                 ImmutablePolynomial(C*2*m*view(PhyPolyH,m,1:m)),
                                 ImmutablePolynomial(C*4*m*(m-1)*view(PhyPolyH,m-1,1:m-1)),
                                 scaled)
        end

    end
end


(P::PhyPolyHermite{m})(x) where {m} = P.P(x)

gradient(P::PhyPolyHermite{m}, x) where {m} = P.Pprime(x)

hessian(P::PhyPolyHermite{m}, x) where {m} = P.Ppprime(x)
