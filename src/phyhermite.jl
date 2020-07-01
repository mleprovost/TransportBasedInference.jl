
export  PhyHermite, normalizephy, normalize, degree, PhyH, phyhermite_coeffmatrix,
        gradient, hessian


# Create a structure to hold physicist Hermite polynomials as well as their first and second derivative
struct PhyHermite{m} <: Hermite
    P::ImmutablePolynomial
    Pprime::ImmutablePolynomial
    Ppprime::ImmutablePolynomial
    scale::Bool
end

normalizephy(m::Int64) =sqrt(gamma(m+1) * 2^m * sqrt(π))
normalize(P::PhyHermite{m}) where {m} = normalizephy(m)

degree(P::PhyHermite{m}) where {m} = m

# Adapted https://people.sc.fsu.edu/~jburkardt/m_src/hermite_polynomial/h_polynomial_coefficients.m

#    Input, integer N, the highest order polynomial to compute.
#    Note that polynomials 0 through N will be computed.
#
#    Output, real C(1:N+1,1:N+1), the coefficients of the Hermite
#    polynomials.
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


const PhyH = phyhermite_coeffmatrix(20)

function PhyHermite(m::Int64;scaled::Bool= false)
    @assert m>=0 "The order of the polynomial should be >=0"
    if scaled ==false
        if m==0
            return PhyHermite{m}(ImmutablePolynomial(1.0),
                                 ImmutablePolynomial(0.0),
                                 ImmutablePolynomial(0.0),
                                 scaled)
        elseif m==1
            return PhyHermite{m}(ImmutablePolynomial(view(PhyH,m+1,1:m+1)),
                                 ImmutablePolynomial(2*m*view(PhyH,m,1:m)),
                                 ImmutablePolynomial(0.0),
                                 scaled)
        else
            return PhyHermite{m}(ImmutablePolynomial(view(PhyH,m+1,1:m+1)),
                                 ImmutablePolynomial(2*m*view(PhyH,m,1:m)),
                                 ImmutablePolynomial(2*m*(m-1)*view(PhyH,m-1,1:m-1)),
                                 scaled)
        end
    else
        C = 1/normalizephy(m)
        if m==0
            return PhyHermite{m}(ImmutablePolynomial(1.0),
                                 ImmutablePolynomial(0.0),
                                 ImmutablePolynomial(0.0),
                                 scaled)
        elseif m==1
            return PhyHermite{m}(ImmutablePolynomial(C*view(PhyH,m+1,1:m+1)),
                                 ImmutablePolynomial(C*2*m*view(PhyH,m,1:m)),
                                 ImmutablePolynomial(0.0),
                                 scaled)
        else
            return PhyHermite{m}(ImmutablePolynomial(C*view(PhyH,m+1,1:m+1)),
                                 ImmutablePolynomial(C*2*m*view(PhyH,m,1:m)),
                                 ImmutablePolynomial(C*2*m*(m-1)*view(PhyH,m-1,1:m-1)),
                                 scaled)
        end

    end
end


(P::PhyHermite{m})(x) where {m} = P.P(x)

gradient(P::PhyHermite{m}, x) where {m} = P.Pprime(x)

hessian(P::PhyHermite{m}, x) where {m} = P.Ppprime(x)
