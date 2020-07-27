

export  PhyHermite, degree,
        FamilyPhyHermite, FamilyScaledPhyHermite,
        DPhyPolyHermite,
        FamilyDPhyPolyHermite, FamilyDScaledPhyPolyHermite,
        FamilyD2PhyPolyHermite, FamilyD2ScaledPhyPolyHermite,
        derivative!, derivative,
        evaluate!, evaluate,
        vander!, vander


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



function derivative!(dF, F::PhyHermite{m}, k::Int64, x) where {m}
    @assert k>-2 "anti-derivative is not implemented for k<-1"
    @assert size(dF,1) == size(x,1) "Size of dF and x don't match"
    N = size(x,1)
    if k==0
         # map!(xi -> F.Poly.P(xi)*exp(-xi^2/2), dF, x)
         @avx @. dF = F.Poly.P(x)*exp(-x^2/2)
         return  dF
    elseif k==1
        if F.scaled ==false
            Pprime = FamilyDPhyPolyHermite[m+1]
            @avx @. dF = Pprime.P(x)*exp(-x^2/2)
            return dF
        else
            Pprime = FamilyDScaledPhyPolyHermite[m+1]
            @avx @. dF = Pprime.P(x)*exp(-x^2/2)
            return dF
        end
    elseif k==2
        if F.scaled ==false
            Ppprime = FamilyD2PhyPolyHermite[m+1]
            @avx @. dF = Ppprime.P(x)*exp(-x^2/2)
            return dF
        else
            Ppprime = FamilyD2ScaledPhyPolyHermite[m+1]

            @avx @. dF = Ppprime.P(x)*exp(-x^2/2)
            return dF
        end
        # map!(xi->ForwardDiff.derivative(y->ForwardDiff.derivative(z->F(z), y), xi), dF, x)
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

derivative(F::PhyHermite{m}, k::Int64, x::Array{Float64,1}) where {m} = derivative!(zero(x), F, k, x)


function evaluate!(dV, P::PhyHermite{m}, x) where {m}
    N = size(x,1)
    @assert size(dV) == (N, m+1) "Wrong dimension of the Vander matrix"

    # H_0(x) = 1, H_1(x) = 2*x
    col0 = view(dV,:,1)
    @avx @. col0 = exp(-0.5*x^2)

    if P.scaled
        rmul!(col0, 1/Cphy(0))
    end
    if m == 0
        return dV
    end

    if P.scaled
        rmul!(col0, Cphy(0))
    end

    col = view(dV,:,2)
    @avx @. col = 2.0*x*col0

    if P.scaled
        rmul!(col0, 1/Cphy(0))
        rmul!(col , 1/Cphy(1))
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

evaluate(P::PhyHermite{m}, x::Array{Float64,1}) where {m} = evaluate!(zeros(size(x,1), m+1), P, x)


function vander!(dV, P::PhyHermite{m}, k::Int64, x::Array{Float64,1}) where {m}

    if k==0
        evaluate!(dV, P, x)
        return dV
    elseif k==1
        # ψ′_n(x) = Q_n(x) exp(-x^2/2)
        # Q_0(x) = -x
        # Q_1(x) = -2x^2 - 2
        evaluate!(dV, P, x)
        if m==0
            col0 = view(dV,:,1)
            @avx @. col0 *= -x
            return dV
        end

        col0 = view(dV,:,1)
        col1 = view(dV,:,2)

        N = size(x,1)
        ψn = zeros(N)
        ψnp1 = zeros(N)

        copy!(ψn, col0)
        copy!(ψnp1, col1)

        @avx @. col0 *= -x

        if P.scaled == true
            @avx @. col1 = 2.0*Cphy(0)/Cphy(1)*(1.0 - x^2) * ψn
        else
            @avx @. col1 = 2.0*(1.0 - x^2) * ψn
        end

        @inbounds for i=2:m
            copy!(ψn, ψnp1)
            colp1 = view(dV,:,i+1)
            copy!(ψnp1, colp1)
            if P.scaled == true
                @avx @. colp1 = (1/Cphy(i))*(2*(i)*Cphy(i-1)*ψn - x * Cphy(i)*ψnp1)
            else
                @avx @. colp1 = (2*(i)*ψn - x * ψnp1)
            end
        end

        return dV
    elseif k==2
        # Use the relation ψn″(x)  + (2*n + 1 -x^2) ψn(x) = 0
        evaluate!(dV, P, x)
        @inbounds for i=0:m
            col = view(dV,:,i+1)
            @avx @. col *= (-2.0*i - 1.0 + x^2)
        end
        return dV
    end
end

vander(P::PhyHermite{m}, k::Int64, x::Array{Float64,1}) where {m} = vander!(zeros(size(x,1), m+1), P, k, x)
