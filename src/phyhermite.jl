

export  PhyHermite, degree,
        FamilyPhyHermite, FamilyScaledPhyHermite,
        DPhyPolyHermite,
        FamilyDPhyPolyHermite, FamilyDScaledPhyPolyHermite,
        FamilyD2PhyPolyHermite, FamilyD2ScaledPhyPolyHermite
        derivative, vander

# Create a structure to hold physicist Hermite functions defined as
# ψn(x) = Hn(x)*exp(-x^2/2)

struct PhyHermite{m} <: ParamFcn
    Poly::PhyPolyHermite{m}
    scaled::Bool
end

# function Base.show(io::IO, P::PhyHermite{m}) where {m}
# println(io,string(m)*"-th order physicist Hermite function, scaled = "*string(P.scaled))
# end

PhyHermite(m::Int64; scaled::Bool = false) = PhyHermite{m}(PhyPolyHermite(m; scaled = scaled), scaled)

degree(P::PhyHermite{m}) where {m} = m

(P::PhyHermite{m})(x) where {m} = P.Poly.P(x)*exp(-x^2/2)

const FamilyPhyHermite = map(i->PhyHermite(i),0:20)
const FamilyScaledPhyHermite = map(i->PhyHermite(i; scaled = true),0:20)


# Store P′n - Pn * X with Pn the n-th Physicist Hermite Polynomial

function DPhyPolyHermite(m::Int64; scaled::Bool)
    k = 1
    factor = 2^k*exp(loggamma(m+1) - loggamma(m+1-k))
    P = PhyPolyHermite(m; scaled = false)
    Pprime = derivative(P, 1)
    if scaled == false
        return Pprime - ImmutablePolynomial((0.0, 1.0))*P.P
    else
        C = 1/Cphy(m)
        return C*(Pprime - ImmutablePolynomial((0.0, 1.0))*P.P)
    end
end


const FamilyDPhyPolyHermite = map(i->AbstractPhyHermite(DPhyPolyHermite(i; scaled = false), false), 0:20)
const FamilyDScaledPhyPolyHermite = map(i->AbstractPhyHermite(DPhyPolyHermite(i; scaled = true), true), 0:20)


# Store P″n - 2 P′n * X  + Pn * (X^2 - 1) with Pn the n-th Physicist Hermite Polynomial

function D2PhyPolyHermite(m::Int64; scaled::Bool)
    k = 1
    factor = 2^k*exp(loggamma(m+1) - loggamma(m+1-k))
    P = PhyPolyHermite(m; scaled = false)
    Pprime  = derivative(P, 1)
    Ppprime = derivative(P, 2)
    if scaled == false
        return Ppprime - ImmutablePolynomial((0.0, 2.0))*Pprime +
               P.P*ImmutablePolynomial((-1.0, 0.0, 1.0))
    else
        C = 1/Cphy(m)
        return C*(Ppprime - ImmutablePolynomial((0.0, 2.0))*Pprime +
                  P.P*ImmutablePolynomial((-1.0, 0.0, 1.0)))
    end
end

const FamilyD2PhyPolyHermite = map(i->AbstractPhyHermite(D2PhyPolyHermite(i; scaled = false), false), 0:20)
const FamilyD2ScaledPhyPolyHermite = map(i->AbstractPhyHermite(D2PhyPolyHermite(i; scaled = true), true), 0:20)



function derivative(F::PhyHermite{m}, k::Int64, x::Array{Float64,1}) where {m}
    @assert k>-2 "anti-derivative is not implemented for k<-1"
    N = size(x,1)
    dF = zeros(N)
    if k==0
        map!(F, dF, x)
        return dF
    elseif k==1
        if F.scaled ==false
            Pprime = FamilyDPhyPolyHermite[m+1]
            map!(Pprime, dF, x)
        else
            Pprime = FamilyDScaledPhyPolyHermite[m+1]
            map!(Pprime, dF, x)
        end
        # map!(y->ForwardDiff.derivative(F, y), dF, x)
        return dF

    elseif k==2
        if F.scaled ==false
            Ppprime = FamilyD2PhyPolyHermite[m+1]
            map!(Ppprime, dF, x)
        else
            Ppprime = FamilyD2ScaledPhyPolyHermite[m+1]
            map!(Ppprime, dF, x)
        end
        # map!(xi->ForwardDiff.derivative(y->ForwardDiff.derivative(z->F(z), y), xi), dF, x)
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
                for k = 1:ceil(Int64, nn/2)
                    exp_c[nn-2*k+1+1] += (c[nn+1]*fact2(nn-1) / fact2(nn-2*k+1))
                end
            else
                Cst += c[nn+1] * fact2(nn-1)
                for k = 0:ceil(Int64, (nn-1)/2)
                    exp_c[nn-2*k-1+1] += (c[nn+1]*fact2(nn-1) / fact2(nn-2*k-1))
                end

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

function vander(P::PhyHermite{m}, k::Int64, x::Array{Float64,1}) where {m}
    N = size(x,1)
    dV = zeros(N, m+1)

    @inbounds for i=0:m
        col = view(dV,:,i+1)

        # Store the k-th derivative of the i-th order Hermite polynomial
        if P.scaled == false
            col .= derivative(FamilyPhyHermite[i+1], k, x)
        else
            col .= derivative(FamilyScaledPhyHermite[i+1], k, x)
        end
    end
    return dV
end
