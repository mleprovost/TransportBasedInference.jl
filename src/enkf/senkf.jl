export StochEnKF, SeqStochEnKF

"""
$(TYPEDEF)

A structure for the stochastic ensemble Kalman filter (sEnKF)

References:

Evensen, G. (1994). Sequential data assimilation with a nonlinear quasi‐geostrophic model using Monte Carlo methods to forecast error statistics. Journal of Geophysical Research: Oceans, 99(C5), 10143-10162.
## Fields
$(TYPEDFIELDS)
"""

struct StochEnKF<:SeqFilter
    "Filter function"
    G::Function

    "Standard deviations of the measurement noise distribution"
    ϵy::AdditiveInflation

    "Time step dynamic"
    Δtdyn::Float64

    "Time step observation"
    Δtobs::Float64

	"Boolean: is the covariance matrix localized"
	islocal::Bool

    "Boolean: is state vector filtered"
    isfiltered::Bool
end

function StochEnKF(G::Function, ϵy::AdditiveInflation,
    Δtdyn, Δtobs; islocal = false, isfiltered = false)
    @assert norm(mod(Δtobs, Δtdyn))<1e-12 "Δtobs should be an integer multiple of Δtdyn"

    return StochEnKF(G, ϵy, Δtdyn, Δtobs, islocal, isfiltered)
end

# If no filtering function is provided, use the identity in the constructor.
function StochEnKF(ϵy::AdditiveInflation,
    Δtdyn, Δtobs; islocal = false)
    @assert norm(mod(Δtobs, Δtdyn))<1e-12 "Δtobs should be an integer multiple of Δtdyn"

    return StochEnKF(x -> x, ϵy, Δtdyn, Δtobs, islocal, false)
end



function Base.show(io::IO, enkf::StochEnKF)
	println(io,"Stochastic EnKF  with filtered = $(enkf.isfiltered)")
end


function (enkf::StochEnKF)(X, ystar::Array{Float64,1}, t::Float64; laplace::Bool=false)

    Ny = size(ystar,1)
    Nx = size(X,1)-Ny
    Ne = size(X, 2)

    meas  = viewmeas(X,Ny,Nx)
    state = viewstate(X,Ny,Nx)

    x̄ = copy(mean(X[Ny+1:Ny+Nx,:],dims=2)[:,1])
    Xf = copy(X[Ny+1:Ny+Nx,:])
    Xf .-= x̄
	rmul!(Xf, 1/sqrt(Ne-1))

	# Need the covariance to perform the localisation
	u = zeros(Ny, Ne)
	if laplace == false
		u .= enkf.ϵy.σ*randn(Ny, Ne) .+ enkf.ϵy.m
	else
		u .= sqrt(2.0)*enkf.ϵy.σ*rand(Laplace(),(Ny, Ne)) .+ enkf.ϵy.m
	end
	# ū = mean(u,dims=2)[:,1]
	Yf = copy(meas .- mean(meas; dims=2)[:,1])

	rmul!(Yf,1/sqrt(Ne-1))

	"Analysis step with representers, Evensen, Leeuwen et al. 1998"

	b = (Yf*Yf' + enkf.ϵy.Σ) \ (ystar .+ (u - meas))

	Bᵀb = (Xf*Yf')*b

	state .+= Bᵀb

	return X
end


function (enkf::StochEnKF)(X, ystar::Array{Float64,1}, ȳf::Array{Float64,1}; laplace::Bool=false)
    Ny = size(ystar,1)
    Nx = size(X,1)-Ny
    Ne = size(X, 2)

    meas  = viewmeas(X,Ny,Nx)
    state = viewstate(X,Ny,Nx)

    x̄ = copy(mean(X[Ny+1:Ny+Nx,:],dims=2)[:,1])

    Xf = copy(X[Ny+1:Ny+Nx,:])
    Xf .-= x̄

	# Need the covariance to perform the localisation
	u = zeros(Ny, Ne)
	if laplace == false
		u .= enkf.ϵy.σ*randn(Ny, Ne) .+ enkf.ϵy.m
	else
		u .= sqrt(2.0)*enkf.ϵy.σ*rand(Laplace(),(Ny, Ne)) .+ enkf.ϵy.m
	end
	ū = mean(u,dims=2)[:,1]

	Yf = copy(meas) .- ȳf

	"Analysis step with representers, Evensen, Leeuwen et al. 1998"
		b = (Yf*transpose(Yf) + (Ne-1)*enkf.ϵy.Σ)\(ystar .+ (u - meas))
		Bᵀb = (Xf*Yf')*b
		# @show norm(Bᵀb)
		state .+= Bᵀb

	return X
end

##### Sequential and localized ensemble Kalman filter

struct SeqStochEnKF<:SeqFilter
    "Filter function"
    G::Function

	"Local observation function"
	h::Function

	"Multiplicative inflation"
	β::MultiplicativeInflation

    "Standard deviations of the measurement noise distribution"
    ϵy::AdditiveInflation

	"Observation dimension"
	Ny::Int64

	"State dimension"
	Nx::Int64

    "Time step dynamic"
    Δtdyn::Float64

    "Time step observation"
    Δtobs::Float64

	"Localization"
	Loc::Union{Nothing, Localization}

	"Distance matrix"
	dist::Array{Float64,2}

	"Index of measurement"
	idx::Array{Int64,2}
	# idx contains the dictionnary of the mapping
	# First line contains the range of integer 1:Ny
	# Second line contains the associated indice of each measurement

	# Define cache
	"Cache for the inference"
	cache::Array{Float64,2}

	"Boolean: is state vector filtered"
    isfiltered::Bool

	"Boolean: is assimilation localized"
	islocalized::Bool
end

function SeqStochEnKF(G::Function, h::Function, β::Float64, ϵy::AdditiveInflation,
					  Ny, Nx, Ne, Δtdyn, Δtobs, Loc::Union{Nothing, Localization}, 
					  dist::Array{Float64,2}, idx::Array{Int64,2};
					  isfiltered = false)
    @assert norm(mod(Δtobs, Δtdyn))<1e-12 "Δtobs should be an integer multiple of Δtdyn"

	if typeof(Loc) <: Nothing
		islocalized = false
	else
		islocalized = true
	end
    return SeqStochEnKF(G, h, MultiplicativeInflation(β), ϵy, Ny, Nx, Δtdyn, Δtobs, 
	       Loc, dist, idx, zeros(Nx+1, Ne), isfiltered, islocalized)
end


function Base.show(io::IO, enkf::SeqStochEnKF)
	println(io,"Sequential Stochastic EnKF  with filtered = $(enkf.isfiltered)")
end


# """
#     Define action of SeqStocEnKF on EnsembleStateMeas
# """
# Baptista, Spantini, Marzouk 2019

# Assimilate the entire observation vector
function (enkf::SeqStochEnKF)(X, ystar, t)
	Ny = enkf.Ny
	@assert enkf.Ny==size(ystar,1) "Wrong dimension of the observation"

	@inbounds for i=1:Ny
		enkf(X, ystar[i], t, enkf.idx[:,i])
	end
	return X
end

# Version for a scalar observation
function (enkf::SeqStochEnKF)(X, ystar::Float64, t, idx::Array{Int64, 1})
	idx1, idx2 = idx
	Nx = enkf.Nx
	Ny = enkf.Ny
	Na = Nx+1
	NxX, Ne = size(X)
	cache = enkf.cache
	fill!(cache, 0.0)
	# We add a +1 such that the scalar observation will remain the first entry
	# perm = sortperm(view(enkf.dist,:,idx2))
	# @show idx1, perm
	Xinfl = copy(X)

	# Apply the multiplicative inflation
	Aβ = enkf.β
	Aβ(Xinfl, Ny+1, Ny+Nx)

	# Generate samples from local likelihood
	@inbounds for i=1:Ne
		col = Xinfl[Ny+1:Ny+Nx, i]
		cache[1,i] = enkf.h(col, t)[idx1] + enkf.ϵy.m[idx1] + dot(enkf.ϵy.σ[idx1,:], randn(Ny))
	end

	cache[2:Na, :] .= copy(Xinfl[Ny+1:Ny+Nx,:])

	XYf = copy(cache) .- mean(cache; dims = 2)[:,1]
	rmul!(XYf, 1/sqrt(Ne-1))

	Yf = copy(XYf[1,:])
	Xf = copy(XYf[2:Nx+1, :])

	#Generate local-likelihood samples with uninflated samples
	@inbounds for i=1:Ne
		col = X[Ny+1:Ny+Nx, i]
		cache[1,i] = enkf.h(col, t)[idx1] + enkf.ϵy.m[idx1] + dot(enkf.ϵy.σ[idx1,:], randn(Ny))
	end

	@view(cache[2:Na,:]) .= X[Ny+1:Ny+Nx,:]

	# Since Yf is a vector (Yf Yf') is reduced to a scalar
	"Analysis step with representers, Evensen, Leeuwen et al. 1998"
	if enkf.islocalized == true
		K = view(locXY,:,idx1).* (Xf*Yf)/(dot(Yf, Yf))
	else
		K = (Xf*Yf)/(dot(Yf, Yf))
	end

	for i=1:Ne
		xi = X[Ny+1:Ny+Nx,i]
		X[Ny+1:Ny+Nx,i] .= xi - K*(cache[1,i] - ystar)
		# @show K./dot(Yf,Yf)
		# K ./= dot(Yf, Yf)
		# Bᵀb = K*(ylocal .- view(enkf.ensa.S,1,:))
		#In-place analysis step with gemm!
		# BLAS.gemm!('N', 'T', -1/dot(Yf,Yf), K, view(cache,1,:) .- ylocal, 1.0, view(X, Ny+1:Ny+Nx, :))
	end
	return X
end


# """
#     Define action of StocEnKF on EnsembleStateMeas
# """
# Bocquet Data assimilation p.160 Chapter 6 Stochastic EnKF
# function (enkf::StochEnKF)(ens::EnsembleStateMeas{Nx, Ny, Ne}, ystar::Array{Float64,1}, t::Float64; laplace::Bool=false) where {Nx, Ny, Ne}
#
# 	Xf = deviation(ens.state).S
# 	rmul!(Xf, 1/sqrt(Ne-1))
#
# 	# Need the covariance to perform the localisation
# 	u = zeros(Ny, Ne)
# 	if laplace == false
# 	u .= enkf.ϵy.σ*randn(Ny, Ne) .+ enkf.ϵy.m
# 	else
# 	u .= sqrt(2.0)*enkf.ϵy.σ*rand(Laplace(),(Ny, Ne)) .+ enkf.ϵy.m
# 	end
# 	# ū = mean(u,dims=2)[:,1]
# 	Yf = deviation(ens.meas).S
# 	rmul!(Yf,1/sqrt(Ne-1))
#
# 	"Analysis step with representers, Evensen, Leeuwen et al. 1998"
# 	b = (Yf*Yf' + enkf.ϵy.Σ) \ (ystar .+ (u - ens.meas.S))
#
# 	Bᵀb = (Xf*Yf')*b
#
# 	ens.state.S .+= Bᵀb
#
# 	return ens
# end
#
# # Bocquet Data assimilation p.160 Chapter 6 Stochastic EnKF
# function (enkf::StochEnKF)(ens::EnsembleStateMeas{Nx, Ny, Ne}, ystar::Array{Float64,1}, t::Float64; laplace::Bool=false) where {Nx, Ny, Ne}
#
# 	Xf = deviation(ens.state).S
# 	rmul!(Xf, 1/sqrt(Ne-1))
#
# 	# Need the covariance to perform the localisation
# 	u = zeros(Ny, Ne)
# 	if laplace == false
# 	u .= enkf.ϵy.σ*randn(Ny, Ne) .+ enkf.ϵy.m
# 	else
# 	u .= sqrt(2.0)*enkf.ϵy.σ*rand(Laplace(),(Ny, Ne)) .+ enkf.ϵy.m
# 	end
# 	# ū = mean(u,dims=2)[:,1]
# 	Yf = deviation(ens.meas).S
# 	rmul!(Yf,1/sqrt(Ne-1))
#
# 	"Analysis step with representers, Evensen, Leeuwen et al. 1998"
# 	b = (Yf*Yf' + enkf.ϵy.Σ) \ (ystar .+ (u - ens.meas.S))
#
# 	Bᵀb = (Xf*Yf')*b
#
# 	ens.state.S .+= Bᵀb
#
# 	return ens
# end

# # Version with localization
# function (enkf::StochEnKF)(ens::EnsembleStateMeas{Nx, Ny, Ne}, ystar::Array{Float64,1}, t::Float64, Loc::Localization) where {Nx, Ny, Ne}
#
# 	Xf = deviation(ens.state).S
# 	rmul!(Xf, 1/sqrt(Ne-1))
#
# 	# Localize covariance
# 	locXY = Locgaspari((Nx, Ny), Loc.L, Loc.Gxy)
# 	locYY = Locgaspari((Ny, Ny), Loc.L, Loc.Gyy)
#
# 	# Need the covariance ot peform the localisation
# 	u = zeros(Ny, Ne)
# 	u .= enkf.ϵy.σ*randn(Ny, Ne) .+ enkf.ϵy.m
# 	ū = mean(u,dims=2)[:,1]
#
# 	Yf = deviation(ens.meas).S
# 	rmul!(Yf,1/sqrt(Ne-1))
#
# 	"Analysis step with representers, Evensen, Leeuwen et al. 1998"
# 	b = (locYY .* (Yf*Yf') + enkf.ϵy.Σ) \ ((ū + ystar) .- (u + ens.meas.S))
# 	Bᵀb = (locXY .* (Xf*Yf'))*b
#
#
# 	@inbounds for i = 1:Ne
# 		Bᵀbi = view(Bᵀb,:,i)
# 		ens.state.S[:,i] .+= Bᵀbi
# 	end
# 	return ens
# end
