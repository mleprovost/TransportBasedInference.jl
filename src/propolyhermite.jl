
export  ProPolyHermite, Cpro, degree, ProPolyH, prohermite_coeffmatrix,
        gradient, hessian


# Create a structure to hold physicist Hermite polynomials as well as their first and second derivative
struct ProPolyHermite{m} <: PolyHermite
    P::ImmutablePolynomial
    Pprime::ImmutablePolynomial
    Ppprime::ImmutablePolynomial
    scale::Bool
end
# Hen(x)  = (-1)ⁿ*exp(x²/2)dⁿ/dxⁿ exp(-x²/2)
# Hen′(x) = n*Hen-1(x)
# Hen″(x) = n*(n-1)*Hen-1(x)

Cpro(m::Int64) =sqrt(sqrt(2*π) * gamma(m+1))
Cpro(P::ProPolyHermite{m}) where {m} = Cpro(m)

degree(P::ProPolyHermite{m}) where {m} = m

# Adapted https://people.sc.fsu.edu/~jburkardt/m_src/hermite_polynomial/h_polynomial_coefficients.m

#    Input, integer N, the highest order polynomial to compute.
#    Note that polynomials 0 through N will be computed.
#
#    Output, real C(1:N+1,1:N+1), the coefficients of the Hermite
#    polynomials.

# Recurrence relation:
# Hen+1 = x*Hen - n*Hen-1
function prohermite_coeffmatrix(m::Int64)

    if m < 0
        return Float64[]
    end

    coeff = zeros(m+1, m+1)
    coeff[1, 1] = 1.0

    if m == 0
        return coeff
    end

    coeff[2, 2] = 1.0

    if m==1
        return coeff
    end

    for i  = 2:m
        coeff[i+1, 1]      =  -      (i - 1) * coeff[i-1, 1]
        coeff[i+1, 2:i-1]  =                   coeff[i  , 1:i-2] -
                                     (i - 1) * coeff[i-1, 2:i-1]
        coeff[i+1, i]      =                   coeff[i  , i-1]
        coeff[i+1, i+1]    =                   coeff[i  , i]
    end
    return coeff
end


const ProPolyH = prohermite_coeffmatrix(20)

function ProPolyHermite(m::Int64;scaled::Bool= false)
    @assert m>=0 "The order of the polynomial should be >=0"
    if scaled ==false
        if m==0
            return ProPolyHermite{m}(ImmutablePolynomial(1.0),
                                 ImmutablePolynomial(0.0),
                                 ImmutablePolynomial(0.0),
                                 scaled)
        elseif m==1
            return ProPolyHermite{m}(ImmutablePolynomial(view(ProPolyH,m+1,1:m+1)),
                                 ImmutablePolynomial(m*view(ProPolyH,m,1:m)),
                                 ImmutablePolynomial(0.0),
                                 scaled)
        else
            return ProPolyHermite{m}(ImmutablePolynomial(view(ProPolyH,m+1,1:m+1)),
                                 ImmutablePolynomial(m*view(ProPolyH,m,1:m)),
                                 ImmutablePolynomial(m*(m-1)*view(ProPolyH,m-1,1:m-1)),
                                 scaled)
        end
    else
        C = 1/Cpro(m)
        if m==0
            return ProPolyHermite{m}(ImmutablePolynomial(C),
                                 ImmutablePolynomial(0.0),
                                 ImmutablePolynomial(0.0),
                                 scaled)
        elseif m==1
            return ProPolyHermite{m}(ImmutablePolynomial(C*view(ProPolyH,m+1,1:m+1)),
                                 ImmutablePolynomial(C*m*view(ProPolyH,m,1:m)),
                                 ImmutablePolynomial(0.0),
                                 scaled)
        else
            return ProPolyHermite{m}(ImmutablePolynomial(C*view(ProPolyH,m+1,1:m+1)),
                                 ImmutablePolynomial(C*m*view(ProPolyH,m,1:m)),
                                 ImmutablePolynomial(C*m*(m-1)*view(ProPolyH,m-1,1:m-1)),
                                 scaled)
        end

    end
end


(P::ProPolyHermite{m})(x) where {m} = P.P(x)

gradient(P::ProPolyHermite{m}, x) where {m} = P.Pprime(x)

hessian(P::ProPolyHermite{m}, x) where {m} = P.Ppprime(x)
