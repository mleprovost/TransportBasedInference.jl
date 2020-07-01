# Define probabilistic hermite polynomials

export ProHermite, normalize, prohermite_coeffmatrix, ProH

struct ProHermite{m} <: Hermite
end

# Need to modify this line
normalize(P::ProHermite{m}) where {m} = sqrt(sqrt(2*π)*gamma(m+1))

# Adapted https://people.sc.fsu.edu/~jburkardt/m_src/hermite_polynomial/h_polynomial_coefficients.m

#    Input, integer N, the highest order polynomial to compute.
#    Note that polynomials 0 through N will be computed.
#
#    Output, real C(1:N+1,1:N+1), the coefficients of the Hermite
#    polynomials.
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
        coeff[i+1, 1]      =      -(i - 1) * coeff[i-1, 1]
        coeff[i+1, 2:i-1]  =                 coeff[i  , 1:i-2] -
                                   (i - 1) * coeff[i-1, 2:i-1]
        coeff[i+1, i]      =                 coeff[i  , i-1]
        coeff[i+1, i+1]    =                 coeff[i  , i]
    end
    return coeff
end


const ProH = prohermite_coeffmatrix(20)

prohermite(m::Int64) = Polynomial(Tuple(view(ProH,m,1:m+1)))
prohermite(m::Int64, coeff::Array{Float64,2}) = Polynomial(Tuple(coeff[m,1:m+1]))
