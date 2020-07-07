

export ProHermite, degree, FamilyProHermite, FamilyScaledProHermite,
       derivative, vander

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

(P::ProHermite{m})(x) where {m} = P.Poly.P(x)*exp(-x^2/4)

const FamilyProHermite = map(i->ProHermite(i),0:20)
const FamilyScaledProHermite = map(i->ProHermite(i; scaled = true),0:20)



function derivative(F::ProHermite{m}, k::Int64, x::Array{Float64,1}) where {m}
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
    elseif k==-1
        # Call derivative function for PhyHermite{m}
        dF .= derivative(FamilyPhyHermite[m+1], k, 1/sqrt(2)*x)
        rmul!(dF, 1/sqrt(2^m))
        rmul!(dF, 1/sqrt(2)^2)
        return dF
        if F.scaled == true
            rmul!(dF, 1/Cpro(m))
            return dF
        end
    end
end



function vander(P::ProHermite{m}, k::Int64, x::Array{Float64,1}; scaled::Bool=false) where {m}
    N = size(x,1)
    dV = zeros(N, m+1)

    @inbounds for i=0:m
        col = view(dV,:,i+1)

        # Store the k-th derivative of the i-th order Hermite polynomial
        if scaled == false
            col .= derivative(FamilyProHermite[i+1], k, x)
        else
            col .= derivative(FamilyScaledProHermite[i+1], k, x)
        end
    end
    return dV
end
