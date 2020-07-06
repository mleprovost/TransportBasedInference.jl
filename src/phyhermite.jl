

export PhyHermite, degree, derivative

# Create a structure to hold physicist Hermite functions defined as
# ψn(x) = Hn(x)*exp(-x^2/2)

struct PhyHermite{m} <: Hermite
    Poly::PhyPolyHermite{m}
    scaled::Bool
end

PhyHermite(m::Int64; scaled::Bool = false) = PhyHermite{m}(PhyPolyHermite(m; scaled = scaled), scaled)

degree(P::PhyHermite{m}) where {m} = m

(P::PhyHermite{m})(x) where {m} = P.Poly.P(x)*exp(-x^2/2)

# 
# function derivative(F::PhyHermite{m}, k::Int64, x::Array{Float64,1}) where {m}
#     @assert k>-2 "anti-derivative is not implemented for k<-1"
#     N = size(x,1)
#     if k==0
#         return F.(x)
#     elseif k>0
#         # use the recurrence relation for the k derivative
#         # \psi_m^k(x) = \sum_{i=0}^{k} (k choose i) (-1)^i *
#         #  2^((k-i)/2) * \sqrt{m!/(m-k+i)!} \psi_{m-k+i}(x) He_i(x)
#         dp = zeros(N)
#         ψ  = zeros(N)
#         for i=max(0,k-m):k
#             Fi  = binomial(k,i) * (-1)^i * 2^(k-i)
#             Fi *= exp(loggamma(m+1) - loggamma(m-k+i+1))
#             ψ   =
#
#
#         end
#
#
#
#     return 0.0
#
#
# end
