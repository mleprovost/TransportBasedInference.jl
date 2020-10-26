
export  PhyPolyHermite, Cphy,
        degree, PhyPolyH,
        phyhermite_coeffmatrix,
        FamilyPhyPolyHermite, FamilyScaledPhyPolyHermite,
        derivative,
        evaluate!, evaluate,
        vander, vander!


# Create a structure to hold physicist Hermite polynomials as well as their first and second derivative
struct PhyPolyHermite <: Hermite
    m::Int64
    P::ImmutablePolynomial{Float64}
    scaled::Bool
end

# function Base.show(io::IO, P::PhyPolyHermite{m}) where {m}
# println(io,string(m)*"-th order physicist Hermite polynomial"*string(P.P.coeffs)*", scaled = "*string(P.scaled))
# end
# Hn(x)  = (-1)ⁿ*exp(x²)dⁿ/dxⁿ exp(-x²)
# Hn′(x) = 2n*Hn-1(x)
# Hn″(x) = 2n*(n-1)*Hn-1(x)

Cphy(m::Int64) =sqrt(gamma(m+1) * 2^m * sqrt(π))
Cphy(P::PhyPolyHermite) = Cphy(P.m)

degree(P::PhyPolyHermite) = P.m

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


const PhyPolyH = phyhermite_coeffmatrix(CstMaxDegree)

function PhyPolyHermite(m::Int64;scaled::Bool= false)
    @assert m>=0 "The order of the polynomial should be >=0"
    if scaled ==false
            return PhyPolyHermite(m, ImmutablePolynomial(view(PhyPolyH,m+1,1:m+1)), scaled)
    else
        C = 1/Cphy(m)
            return PhyPolyHermite(m, ImmutablePolynomial(C*view(PhyPolyH,m+1,1:m+1)), scaled)
    end
end

(P::PhyPolyHermite)(x) = P.P(x)

const FamilyPhyPolyHermite = ntuple(i->PhyPolyHermite(i-1), CstMaxDegree+1)
const FamilyScaledPhyPolyHermite = ntuple(i->PhyPolyHermite(i-1; scaled = true), CstMaxDegree+1)


# Compute the k-th derivative of a physicist Hermite polynomial according to
# H_{n}^(k)(x) = 2^{k} n!/(n-k)! H_{n-k}(x)
function derivative(P::PhyPolyHermite, k::Int64)
    m = P.m
    @assert k>=0 "This function doesn't compute anti-derivatives of Hermite polynomials"
    if m>=k
        factor = 2^k*exp(loggamma(m+1) - loggamma(m+1-k))
        if P.scaled == false
            return ImmutablePolynomial(factor*PhyPolyH[m-k+1,1:m-k+1])
        else
            C = 1/Cphy(m)
            return ImmutablePolynomial(factor*C*PhyPolyH[m-k+1,1:m-k+1])
        end
    else
        return ImmutablePolynomial((0.0))
    end
end

function evaluate!(dV, P::PhyPolyHermite, x)
    m = P.m
    N = size(x,1)
    @assert size(dV) == (N, m+1) "Wrong dimension of the Vander matrix"

    # H_0(x) = 1, H_1(x) = 2*x
    col = view(dV,:,1)
    @avx @. col = 1.0

    if P.scaled
        rmul!(col, 1/Cphy(0))
    end
    if m == 0
        return dV
    end

    col = view(dV,:,2)
    @avx @. col = 2.0*x

    if P.scaled
        rmul!(col, 1/Cphy(1))
    end

    if m == 1
        return dV
    end

    if P.scaled
        @inbounds for i=2:m
            colp1 = view(dV,:,i+1)
            col   = view(dV,:,i)
            colm1 = view(dV,:,i-1)
            @avx @. colp1 = 2.0*(Cphy(i-1)* x * col - Cphy(i-2)*(i-1)*colm1)
            # dV[:,i+1] = 2.0*Cphy(i-1)*x .* dV[:,i] - 2.0*Cphy(i-2)*i*dV[:,i-1]
            rmul!(colp1, 1/Cphy(i))
        end
    else
        @inbounds for i=2:m
            colp1 = view(dV,:,i+1)
            col   = view(dV,:,i)
            colm1 = view(dV,:,i-1)
            @avx @. colp1 = 2.0*(x * col - (i-1)*colm1)
        end
    end
    return dV
end

evaluate(P::PhyPolyHermite, x::Array{Float64,1}) = evaluate!(zeros(size(x,1), P.m+1), P, x)



# For next week, the question is should we use the Family of polynomials  evaluate at the samples and just multiply by the constant
# seems to be faster !! than recomputing the derivative

# H_{n}^(k)(x) = 2^{k} n!/(n-k)! H_{n-k}(x)

# vander!(dV::Array{Float64,2}, P::PhyPolyHermite{m}, k=0, x) where {m} = evaluate!(dV::Array{Float64,2}, P::PhyPolyHermite{m}, x)

function vander!(dV, P::PhyPolyHermite, k::Int64, x)
    m = P.m

    if k==0
        evaluate!(dV, P, x)
    # Derivative
    elseif k==1
        # Use recurrence relation H′_{n+1} = (1 + 1/n)*(2*x*H′_{n} - 2*n*H′_{n-1})
        N = size(x,1)
        @assert size(dV) == (N, m+1) "Wrong dimension of the Vander matrix"

        col = view(dV,:,1)
        @avx @. col = 0.0
        if m==0
            return dV
        end

        col = view(dV,:,2)
        @avx @. col = 2.0

        if m==1
            return dV
        end

        col = view(dV,:,3)
        @avx @. col = 8.0*x

        if m==2
            return dV
        end

        @inbounds for i=3:m
            colp1 = view(dV,:,i+1)
            col   = view(dV,:,i)
            colm1 = view(dV,:,i-1)

            @avx @. colp1 = 2.0*(1.0 + 1.0/(i-1.0))*(x * col - (i-1.0) * colm1)
        end

        if P.scaled
            @inbounds for i=1:m
                colp1 = view(dV,:,i+1)
                factor = 2^k*exp(loggamma(i+1) - loggamma(i+1-k))
                rmul!(colp1, 1/(Cphy(i)*sqrt(factor)))
            end

        end
        return dV
    # Second Derivative
    elseif k==2
        # Use recurrence relation H′_{n+1} = (1 + 1/n)*(2*x*H′_{n} - 2*n*H′_{n-1})
        N = size(x,1)
        @assert size(dV) == (N, m+1) "Wrong dimension of the Vander matrix"

        col = view(dV,:,1)
        @avx @. col = 0.0
        if m==0
            return dV
        end

        col = view(dV,:,2)
        @avx @. col = 0.0

        if m==1
            return dV
        end

        col = view(dV,:,3)
        @avx @. col = 8.0

        if m==2
            return dV
        end

        col = view(dV,:,4)
        @avx @. col = 48.0 * x

        if m==3
            return dV
        end

        @inbounds for i=4:m
            colp1 = view(dV,:,i+1)
            col   = view(dV,:,i)
            colm1 = view(dV,:,i-1)

            @avx @. colp1 = 2.0*((i)/(i-2.0))*(x * col - (i-1.0) * colm1)
        end

        if P.scaled
            @inbounds for i=1:m
                colp1 = view(dV,:,i+1)
                factor = 2^k*exp(loggamma(i+1.0) - loggamma(i+1.0-k))
                @avx rmul!(colp1, 1/(Cphy(i)*sqrt(factor)))
            end

        end
        return dV
    end
end


vander(P::PhyPolyHermite, k::Int64, x::Array{Float64,1}) = vander!(zeros(size(x,1), P.m+1), P, k, x)
