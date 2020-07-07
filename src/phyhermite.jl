

export PhyHermite, degree, FamilyPhyHermite, FamilyScaledPhyHermite,
       derivative, vander

# Create a structure to hold physicist Hermite functions defined as
# ψn(x) = Hn(x)*exp(-x^2/2)

struct PhyHermite{m} <: Hermite
    Poly::PhyPolyHermite{m}
    scaled::Bool
end

PhyHermite(m::Int64; scaled::Bool = false) = PhyHermite{m}(PhyPolyHermite(m; scaled = scaled), scaled)

degree(P::PhyHermite{m}) where {m} = m

(P::PhyHermite{m})(x) where {m} = P.Poly.P(x)*exp(-x^2/2)

const FamilyPhyHermite = map(i->PhyHermite(i),0:20)
const FamilyScaledPhyHermite = map(i->PhyHermite(i; scaled = true),0:20)



function derivative(F::PhyHermite{m}, k::Int64, x::Array{Float64,1}) where {m}
    @assert k>-2 "anti-derivative is not implemented for k<-1"
    N = size(x,1)
    dF = zeros(N)
    if k==0
        map!(F, dF, x)
        return dF
    elseif k==1
        map!(y->ForwardDiff.derivative(F, y), dF, x)
        return dF

    elseif k==2
        map!(xi->ForwardDiff.derivative(y->ForwardDiff.derivative(z->F(z), y), xi), dF, x)
        return dF
    elseif k==-1 # Compute the anti-derivatives
        # extract monomial coefficients
        c = F.Poly.P.coeffs
        # Compute integral using monomial formula
        Cst = 0.0
        pi_erf_c = 0.0
        exp_c = zeros(m)

        for nn = 0:m
            if mod(nn,2)==0

                pi_erf_c += (c[nn+1]*fact2(nn-1))
                pi_erf_c
                for k = 1:ceil(Int64, nn/2)
                    exp_c[nn-2*k+1+1] += (c[nn+1]*fact2(nn-1) / fact2(nn-2*k-1))
                end
                exp_c
            else
                Cst += c[nn+1] * fact2(nn-1)
                Cst
                for k = 0:ceil(Int64, (nn-1)/2)
                    exp_c[nn-2*k-1+1] += (c[nn+1]*fact2(nn-1) / fact2(nn-2*k-1))
                end
                exp_c
            end
        end
        # Evaluate exponential terms
        ev_pi_erf = zeros(N)
        @. ev_pi_erf = sqrt(π/2)*erf(1/sqrt(2)*x)
        ev_exp = zeros(N)
        @. ev_exp = exp(-x^2/2)

        # Evaluate x^N matrix
        ntot = size(exp_c, 1)
        X = ones(N, ntot)
        @inbounds for i = 2:ntot
            X[:,i] = X[:,i-1] .*x
        end
        # Compute integral
        dF .= Cst .+ pi_erf_c .* ev_pi_erf - (X*exp_c) .* ev_exp
        return dF
    end

end

function vander(P::PhyHermite{m}, k::Int64, x::Array{Float64,1}; scaled::Bool=false) where {m}
    N = size(x,1)
    dV = zeros(N, m+1)

    @inbounds for i=0:m
        col = view(dV,:,i+1)

        # Store the k-th derivative of the i-th order Hermite polynomial
        if scaled == false
            col .= derivative(FamilyPhyHermite[i+1], k, x)
        else
            col .= derivative(FamilyScaledPhyHermite[i+1], k, x)
        end
    end
    return dV
end



# # use the recurrence relation for the k derivative
# # \psi_m^k(x) = \sum_{i=0}^{k} (k choose i) (-1)^i *
# #  2^((k-i)/2) * \sqrt{m!/(m-k+i)!} \psi_{m-k+i}(x) He_i(x)
# dp = zeros(N)
# ψ  = zeros(N)
#     for i=max(0,k-m):k
#         Fi  = binomial(k,i) * (-1)^i * 2^(k-i)
#         Fi *= exp(loggamma(m+1) - loggamma(m-k+i+1))
#
#         ψ  .= FamilyPhyHermite[m-k+i+1].(x)
#         dp += deepcopy(ψ .* FamilyProPolyHermite[i+1].(x))
#     end
# if F.scaled ==true
#     rmul!(dp, Cphy(m))
# end
# return dp
