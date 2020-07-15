

export  ProHermite, degree,
        FamilyProHermite, FamilyScaledProHermite,
        DProPolyHermite,
        FamilyDProPolyHermite, FamilyDScaledProPolyHermite,
        FamilyD2ProPolyHermite, FamilyD2ScaledProPolyHermite
        derivative!, derivative, vander!, vander

# Create a structure to hold physicist Hermite functions defined as
# ψen(x) = Hen(x)*exp(-x^2/4)

struct ProHermite{m} <: ParamFcn
    Poly::ProPolyHermite{m}
    scaled::Bool
end
#
# function Base.show(io::IO, P::ProHermite{m}) where {m}
# println(io,string(m)*"-th order probabilistic Hermite function, scaled = "*string(P.scaled))
# end

ProHermite(m::Int64; scaled::Bool = false) = ProHermite{m}(ProPolyHermite(m; scaled = scaled), scaled)

degree(P::ProHermite{m}) where {m} = m

(P::ProHermite{m})(x::T) where {m, T <: Real} = P.Poly.P(x)*exp(-x^2/4)

const FamilyProHermite = map(i->ProHermite(i),0:20)
const FamilyScaledProHermite = map(i->ProHermite(i; scaled = true),0:20)

# Store Pe′_n - Pe_n * X/2 with Pe_n the n-th Probabilistic Hermite Polynomial

function DProPolyHermite(m::Int64; scaled::Bool)
    k = 1
    factor = exp(loggamma(m+1) - loggamma(m+1-k))
    P = ProPolyHermite(m; scaled = false)
    Pprime = derivative(P, 1)
    if scaled == false
        return Pprime - ImmutablePolynomial((0.0, 0.5))*P.P
    else
        C = 1/Cpro(m)
        return C*(Pprime - ImmutablePolynomial((0.0, 0.5))*P.P)
    end
end


const FamilyDProPolyHermite = map(i->AbstractProHermite(DProPolyHermite(i; scaled = false), false), 0:20)
const FamilyDScaledProPolyHermite = map(i->AbstractProHermite(DProPolyHermite(i; scaled = true), true), 0:20)


# Store Pe″n -  Pe′n * X  + Pn * (X^2/2 - 1/2) with Pe_n the n-th Probabilistic Hermite Polynomial

function D2ProPolyHermite(m::Int64; scaled::Bool)
    k = 2
    factor = exp(loggamma(m+1) - loggamma(m+1-k))
    P = ProPolyHermite(m; scaled = false)
    Pprime  = derivative(P, 1)
    Ppprime = derivative(P, 2)
    if scaled == false
        return Ppprime - ImmutablePolynomial((0.0, 1.0))*Pprime +
               P.P*ImmutablePolynomial((-0.5, 0.0, 0.25))
    else
        C = 1/Cpro(m)
        return C*(Ppprime - ImmutablePolynomial((0.0, 1.0))*Pprime +
                  P.P*ImmutablePolynomial((-0.5, 0.0, 0.25)))
    end
end

const FamilyD2ProPolyHermite = map(i->AbstractProHermite(D2ProPolyHermite(i; scaled = false), false), 0:20)
const FamilyD2ScaledProPolyHermite = map(i->AbstractProHermite(D2ProPolyHermite(i; scaled = true), true), 0:20)


function derivative!(dF, F::ProHermite{m}, k::Int64, x::Array{Float64,1}) where {m}
    @assert k>-2 "anti-derivative is not implemented for k<-1"
    @assert size(dF,1) == size(x,1) "Size of dF and x don't match"
    N = size(x,1)
    if k==0
        # map!(F, dF, x)
        @avx @. dF = F.Poly.P(x)*exp(-x^2/4)
        return  dF
    elseif k==1
        if F.scaled ==false
            Pprime = FamilyDProPolyHermite[m+1]
            @avx @. dF = Pprime.P(x)*exp(-x^2/4)
            return dF
        else
            Pprime = FamilyDScaledProPolyHermite[m+1]
            @avx @. dF = Pprime.P(x)*exp(-x^2/4)
            return dF
        end
    elseif k==2
        if F.scaled ==false
            Ppprime = FamilyD2ProPolyHermite[m+1]
            @avx @. dF = Ppprime.P(x)*exp(-x^2/4)
            return dF
        else
            Ppprime = FamilyD2ScaledProPolyHermite[m+1]
            @avx @. dF = Ppprime.P(x)*exp(-x^2/4)
            return dF
        end
        # map!(xi->ForwardDiff.derivative(y->ForwardDiff.derivative(z->F(z), y), xi), dF, x)
    elseif k==-1
        # Call derivative function for PhyHermite{m}
        dF .= derivative(FamilyPhyHermite[m+1], k, (1/sqrt(2))*x)
        rmul!(dF, 1/sqrt(2^m))
        rmul!(dF, 1/sqrt(2)^k)
        if F.scaled == true
            rmul!(dF, 1/Cpro(m))
        end
        return dF
    end
end

derivative(F::ProHermite{m}, k::Int64, x::Array{Float64,1}) where {m} = derivative!(zero(x), F, k, x)


function vander!(dV::Array{Float64,2}, P::ProHermite{m}, k::Int64, x::Array{Float64,1}) where {m}
    N = size(x,1)

    @assert size(dV) == (N, m+1) "Wrong dimension of the Vander matrix"

    @inbounds for i=0:m
        col = view(dV,:,i+1)

        # Store the k-th derivative of the i-th order Hermite polynomial
        if P.scaled == false
            derivative!(col, FamilyProHermite[i+1], k, x)
        else
            derivative!(col, FamilyScaledProHermite[i+1], k, x)
        end
    end
    return dV
end


vander(P::ProHermite{m}, k::Int64, x::Array{Float64,1}) where {m} = vander!(zeros(size(x,1), m+1), P, k, x)
