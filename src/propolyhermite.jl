
export  ProPolyHermite, Cpro, degree, ProPolyH, prohermite_coeffmatrix,
        FamilyProPolyHermite, FamilyScaledProPolyHermite,
        derivative, vander!, vander


# Create a structure to hold physicist Hermite polynomials as well as their first and second derivative
struct ProPolyHermite{m} <: ParamFcn
    P::ImmutablePolynomial
    scaled::Bool
end

# function Base.show(io::IO, P::PhyPolyHermite{m}) where {m}
# println(io,string(m)*"-th order probabilistic Hermite polynomial"*string(P.P)*", scaled = "*string(P.scaled))
# end

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
            return ProPolyHermite{m}(ImmutablePolynomial(view(ProPolyH,m+1,1:m+1)), scaled)
    else
        C = 1/Cpro(m)
            return ProPolyHermite{m}(ImmutablePolynomial(C*view(ProPolyH,m+1,1:m+1)), scaled)
    end
end

(P::ProPolyHermite{m})(x) where {m} = P.P(x)

const FamilyProPolyHermite = map(i->ProPolyHermite(i),0:20)
const FamilyScaledProPolyHermite = map(i->ProPolyHermite(i; scaled = true),0:20)


# Compute the k-th derivative of a physicist Hermite polynomial according to
# H_{n}^(k)(x) = n!/(n-k)! H_{n-k}(x)
function derivative(P::ProPolyHermite{m}, k::Int64) where {m}
    @assert k>=0 "This function doesn't compute anti-derivatives of Hermite polynomials"
    if m>=k
        factor = exp(loggamma(m+1) - loggamma(m+1-k))
        if P.scaled == false
            return ImmutablePolynomial(factor*ProPolyH[m-k+1,1:m-k+1])
        else
            C = 1/Cpro(m)
            return ImmutablePolynomial(factor*C*ProPolyH[m-k+1,1:m-k+1])
        end
    else
        return ImmutablePolynomial((0.0))
    end
end


# For next week, the question is should we use the Family of polynomials  evaluate at the samples and just multiply by the constant
# seems to be faster !! than recomputing the derivative

# H_{n}^(k)(x) = n!/(n-k)! H_{n-k}(x)

function vander!(dV::Array{Float64,2}, P::ProPolyHermite{m}, k::Int64, x) where {m}
    N = size(x,1)
    @assert size(dV) == (N, m+1) "Wrong dimension of the Vander matrix"

    @inbounds for i=0:m
        col = view(dV,:,i+1)

        # Store the k-th derivative of the i-th order Hermite polynomial
        if P.scaled == false
            Pik = derivative(FamilyProPolyHermite[i+1], k)
            col .= Pik.(x)
        else
            Pik = derivative(FamilyScaledProPolyHermite[i+1], k)
            if i>=k
                factor = exp(loggamma(i+1) - loggamma(i+1-k))
            else
                factor = 1.0
            end
            col .= Pik.(x)*(1/sqrt(factor))
        end
    end
    return dV
end


vander(P::ProPolyHermite{m}, k::Int64, x::Array{Float64,1}) where {m} = vander!(zeros(size(x,1), m+1), P, k, x)
