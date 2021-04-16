

export  ProHermite, degree,
        FamilyProHermite, FamilyScaledProHermite,
        DProPolyHermite,
        FamilyDProPolyHermite, FamilyDScaledProPolyHermite,
        FamilyD2ProPolyHermite, FamilyD2ScaledProPolyHermite
        derivative!, derivative,
        evaluate!, evaluate,
        vander!, vander


"""
    ProHermite <: Hermite

An immutable structure for probabilistic Hermite functions defined as ψem(x) = Hem(x)*exp(-x^2/4).

## Fields
-  `m` : order of the function
-  `Poly` : probabilistic Hermite polynomial of order m
- `scaled` : with rescaling to have unitary norm

## Constructors
ProHermite(m, Poly, scaled)
ProHermite(m; scaled = false)
"""

struct ProHermite <: Hermite
    m::Int64
    Poly::ProPolyHermite
    scaled::Bool
end
#
function Base.show(io::IO, P::ProHermite)
    println(io, string(P.m)*"-th order probabilistic Hermite function, scaled = "*string(P.scaled))
end

ProHermite(m::Int64; scaled::Bool = false) = ProHermite(m, ProPolyHermite(m; scaled = scaled), scaled)

degree(P::ProHermite) = P.m

(P::ProHermite)(x) = P.Poly.P(x)*exp(-x^2/4)

const FamilyProHermite = ntuple(i->ProHermite(i-1),CstMaxDegree+1)
const FamilyScaledProHermite = ntuple(i->ProHermite(i-1; scaled = true),CstMaxDegree+1)

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


const FamilyDProPolyHermite = ntuple(i->AbstractProHermite(DProPolyHermite(i-1; scaled = false), false), CstMaxDegree+1)
const FamilyDScaledProPolyHermite = ntuple(i->AbstractProHermite(DProPolyHermite(i-1; scaled = true), true), CstMaxDegree+1)


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

const FamilyD2ProPolyHermite = ntuple(i->AbstractProHermite(D2ProPolyHermite(i-1; scaled = false), false), CstMaxDegree+1)
const FamilyD2ScaledProPolyHermite = ntuple(i->AbstractProHermite(D2ProPolyHermite(i-1; scaled = true), true), CstMaxDegree+1)


function derivative!(dF, F::ProHermite, k::Int64, x)
    m = F.m
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

derivative(F::ProHermite, k::Int64, x::Array{Float64,1}) = derivative!(zero(x), F, k, x)


function evaluate!(dV, P::ProHermite, x)
    m = P.m
    N = size(x,1)
    @assert size(dV) == (N, m+1) "Wrong dimension of the Vander matrix"

    # H_0(x) = 1, H_1(x) = x
    col0 = view(dV,:,1)
    @avx @. col0 = exp(-0.25*x^2)

    if P.scaled
        rmul!(col0, 1/Cpro(0))
    end
    if m == 0
        return dV
    end

    if P.scaled
        rmul!(col0, Cpro(0))
    end

    col = view(dV,:,2)
    @avx @. col = x*col0

    if P.scaled
        rmul!(col0, 1/Cpro(0))
        rmul!(col, 1/Cpro(1))
    end

    if m == 1
        return dV
    end

    if P.scaled
        @inbounds for i=2:m
            colp1 = view(dV,:,i+1)
            col   = view(dV,:,i)
            colm1 = view(dV,:,i-1)
            @avx @. colp1 = (Cpro(i-1)* x * col - Cpro(i-2)*(i-1)*colm1)
            # dV[:,i+1] = Cpro(i-1)*x .* dV[:,i] - Cpro(i-2)*i*dV[:,i-1]
            rmul!(colp1, 1/Cpro(i))
        end
    else
        @inbounds for i=2:m
            colp1 = view(dV,:,i+1)
            col   = view(dV,:,i)
            colm1 = view(dV,:,i-1)
            @avx @. colp1 = (x * col - (i-1)*colm1)
        end
    end
    return dV
end

evaluate(P::ProHermite, x::Array{Float64,1}) = evaluate!(zeros(size(x,1), P.m+1), P, x)



# Use ψe^{(k)}_n(x) = 1/√(2^n)*ψ^{k}_n(x/√2)
function vander!(dV, P::ProHermite, k::Int64, x)
    m = P.m
    if k==0
        evaluate!(dV, P, x)
        return dV
    else
        # x is an Array{Float64,1} or a view of it
        C = 1/√(2.0)
        vander!(dV, FamilyPhyHermite[m+1], k, C*x)

        if P.scaled==true
            @inbounds for i=0:m
                col = view(dV,:,i+1)
                @avx @. col *= 1.0/(Cpro(i)*√(2.0^i)*√(2.0)^k)
            end
        else
            @inbounds for i=0:m
                col = view(dV,:,i+1)
                @avx @. col *= 1.0/(√(2.0^i)*√(2.0)^k)
            end
        end
        return dV
    end
end

vander(P::ProHermite, k::Int64, x::Array{Float64,1}) = vander!(zeros(size(x,1), P.m+1), P, k, x)
