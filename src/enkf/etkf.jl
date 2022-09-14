export ETKF, rdnortho

"""
$(TYPEDSIGNATURES)

A routine to generate mean-preserving random rotations.

References:

Nerger, L., Janjić, T., Schröter, J., & Hiller, W. (2012). A unification of ensemble square root Kalman filters. Monthly Weather Review, 140(7), 2335-2345.
Tödter, J., & Ahrens, B. (2015). A second-order exact ensemble square root filter for nonlinear data assimilation. Monthly Weather Review, 143(4), 1347-1367.
"""
function rdnortho(N::Int64)
    Ω = (svd(randn(N-1,N-1)).V)';
    b1 = 1/√(N)
    b1sgn = b1*ones(N)
    b1sgn[end] += sign(b1)
    B = zeros(N,N)
    B[:,1] = b1*ones(N);
    B[:,2:end] = (-(I - 1/(abs(b1)+1.0)*b1sgn*b1sgn'))[:,1:end-1]
    Λb = zeros(N,N)
    Λb[1,1] = 1.0
    Λb[2:end,2:end] = Ω
    return B*Λb*B'
end



"""
$(TYPEDEF)

A structure for the ensemble transform Kalman filter (ETKF)

References:

Bishop, C. H., Etherton, B. J., & Majumdar, S. J. (2001). Adaptive sampling with the ensemble transform Kalman filter. Part I: Theoretical aspects. Monthly weather review, 129(3), 420-436.

## Fields
$(TYPEDFIELDS)
"""

struct ETKF<:SeqFilter
    "Filter function"
    G::Function

    "Standard deviations of the measurement noise distribution"
    ϵy::AdditiveInflation

    "Time step dynamic"
    Δtdyn::Float64

    "Time step observation"
    Δtobs::Float64

	"Random shuffle"
	Δtshuff::Float64

	"Boolean: is the covariance matrix localized"
	islocal::Bool

    "Boolean: is state vector filtered"
    isfiltered::Bool
end

function ETKF(G::Function, ϵy::AdditiveInflation,
    Δtdyn, Δtobs, Δtshuff; islocal = false, isfiltered = false)
    @assert norm(mod(Δtobs, Δtdyn))<1e-12 "Δtobs should be an integer multiple of Δtdyn"

    return ETKF(G, ϵy, Δtdyn, Δtobs, Δtshuff, islocal, isfiltered)
end

# If no filtering function is provided, use the identity in the constructor.
function ETKF(ϵy::AdditiveInflation,
    Δtdyn, Δtobs, Δtshuff; islocal = false)
    @assert norm(mod(Δtobs, Δtdyn))<1e-12 "Δtobs should be an integer multiple of Δtdyn"

    return ETKF(x-> x, ϵy, Δtdyn, Δtobs, Δtshuff, islocal, false)
end


function Base.show(io::IO, enkf::ETKF)
	println(io,"ETKF  with filtered = $(enkf.isfiltered)")
end


# """
#     Define action of ETKF on EnsembleStateMeas
# """
# Bocquet Data assimilation p.160 Chapter 6 ETKF

function (enkf::ETKF)(X, ystar, t)
	Ny = size(ystar,1)
	Nx = size(X,1)-Ny
	Ne = size(X, 2)

	meas  = viewmeas(X,Ny,Nx)
	state = viewstate(X,Ny,Nx)

	x̄ = copy(mean(state,dims=2)[:,1])

	Xf = copy(state)
	Xf .-= x̄
	rmul!(Xf, 1/sqrt(Ne-1))

	ȳ = copy(mean(meas,dims=2)[:,1])

	S = enkf.ϵy.σ^(-1)*(meas .- ȳ)

	rmul!(S, 1/√(Ne-1))

	δH = enkf.ϵy.σ^(-1)*(ystar - ȳ)


	# Th = Hermitian(inv(S'*S+I))
	λ, ϕ = eigen(Hermitian(S'*S + I))

	# w = Th*(S'*δH)
	w = ϕ * Diagonal(1 ./ λ) * ϕ'*(S'*δH)


	if mod(t, enkf.Δtshuff) ==0
		U = rdnortho(Ne)
	else
		U = I
	end

	# ens.state.S .= x̄ .+ Xf*(w .+ √(Ne-1)*sqrt(Th)*U)
	state .= x̄ .+ Xf*(w .+ √(Ne-1)*ϕ * Diagonal( 1 ./ sqrt.(λ)) * ϕ'*U)
	return X
end


function (enkf::ETKF)(X, ystar, ȳf, t)

	Ny = size(ystar,1)
	Nx = size(X,1)-Ny
	Ne = size(X, 2)

    meas  = viewmeas(X,Ny,Nx)
	state = viewstate(X,Ny,Nx)

	x̄ = copy(mean(state,dims=2)[:,1])

	Xf = copy(state)
	Xf .-= x̄
	rmul!(Xf, 1/sqrt(Ne-1))

	S = enkf.ϵy.σ^(-1)*(meas .- ȳf)

	δH = enkf.ϵy.σ^(-1)*(ystar - ȳf)

	# Th = inv(Hermitian(I + S'*S))
	λ, ϕ = eigen(Hermitian(S'*S + I))

	# w = Th*(S'*δH)
	w = ϕ * Diagonal(1 ./ λ) * ϕ'*(S'*δH)


	if mod(t, enkf.Δtshuff) == 0
		U = rdnortho(Ne)
	else
		U = I
	end

	# ens.state.S .= x̄ + Xf*(w + √(Ne-1)*sqrt(Th)*U')
	state .= x̄ .+ Xf*(w .+ √(Ne-1)*ϕ * Diagonal( 1 ./ sqrt.(λ)) * ϕ'*U)
	return X
end
