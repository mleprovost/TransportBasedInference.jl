export StochEnKF#, SeqStochEnKF

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


function Base.show(io::IO, enkf::StochEnKF)
	println(io,"Stochastic EnKF  with filtered = $(enkf.isfiltered)")
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


function (enkf::StochEnKF)(X, ystar::Array{Float64,1}, t::Float64; laplace::Bool=false)

    Ny = size(ystar,1)
    Nx = size(X,1)-Ny
    Ne = size(X, 2)

    meas  = viewmeas(X,Ny,Nx)
    state = viewstate(X,Ny,Nx)

    x̄ = deepcopy(mean(X[Ny+1:Ny+Nx,:],dims=2)[:,1])
    Xf = deepcopy(X[Ny+1:Ny+Nx,:])
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

    x̄ = deepcopy(mean(X[Ny+1:Ny+Nx,:],dims=2)[:,1])

    Xf = deepcopy(X[Ny+1:Ny+Nx,:])
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
#
#
# ##### Sequential and localized ensemble Kalman filter
#
# struct SeqStochEnKF<:SeqFilter
#     "Filter function"
#     G::Function
#
# 	"Local observation function"
# 	h::Function
#
#     "Standard deviations of the measurement noise distribution"
#     ϵy::AdditiveInflation
#
# 	"Multiplicative inflation"
# 	β::Float64
#
#     "Time step dynamic"
#     Δtdyn::Float64
#
#     "Time step observation"
#     Δtobs::Float64
#
# 	"Boolean: is the covariance matrix localized"
# 	islocal::Bool
#     "Boolean: is state vector filtered"
#     isfiltered::Bool
#
# 	# Define cache
# 	"EnsembleState to hold the augmented state"
# 	ensa::EnsembleState
#
# 	"Index of measurement"
# 	idx::Array{Int64,2}
# 	# idx contains the dictionnary of the mapping
# 	# First line contains the range of integer 1:Ny
# 	# Second line contains the associated indice of each measurement
#
# end
#
# function SeqStochEnKF(G::Function, h::Function, ϵy::AdditiveInflation,
#     β, Δtdyn, Δtobs, ensa::EnsembleState{Na, Ne}; islocal = false, isfiltered = false) where {Na, Ne}
#     @assert norm(mod(Δtobs, Δtdyn))<1e-12 "Δtobs should be an integer multiple of Δtdyn"
#
#     return SeqStochEnKF(G, h, ϵy, β, Δtdyn, Δtobs, islocal, isfiltered, ensa)
# end
#
#
# function Base.show(io::IO, enkf::SeqStochEnKF)
# 	println(io,"Sequential Stochastic EnKF  with filtered = $(enkf.isfiltered)")
# end
#
#
# # """
# #     Define action of SeqStocEnKF on EnsembleStateMeas
# # """
# # Baptista, Spantini, Marzouk 2019
#
#
# # Version with localization
# function (enkf::SeqStochEnKF)(ens::EnsembleStateMeas{Nx, Ny, Ne}, ystar, t, Loc::Localization) where {Nx, Ny, Ne}
# 	h = enkf.h
#
# 	# Localize covariance
# 	locXY = Locgaspari((Nx, Ny), Loc.L, Loc.Gxy)
# 	# locYY = Locgaspari((Ny, Ny), Loc.L, Loc.Gyy)
# 	# @show view(locXY,:,1)
# 	# Sequential assimilation of the observations
# 	@inbounds for i=1:Ny
# 		idx1, idx2 = enkf.idx[:,i]
# 		ylocal = ystar[idx1]
#
# 		# Inflate state
# 		enkf.ensa.S[2:end,:] .= deepcopy(ens.state.S)
# 		Aβ = MultiplicativeInflation(enkf.β)
# 		Aβ(enkf.ensa)
#
#
# 		# Generate samples from local likelihood with inflated ensemble
# 		@inbounds for i=1:Ne
# 			col = view(enkf.ensa.S,2:Nx+1,i)
# 			enkf.ensa.S[1,i] =  h(t, col)[idx1] + enkf.ϵy.m[idx1] + dot(enkf.ϵy.σ[idx1,:], randn(Ny))
# 		end
# 		# enkf.ensa.S[1,:] .= [6.610555270713447;   6.610557962530371;   6.610555098242182;   6.610553119289472;   6.610555739404843;   6.610553456891982;   6.610552597219042;  6.610555864033318]
# 		# @show enkf.ensa.S[1,:]
# 		# @show enkf.ensa.S[2,:]
#
# 		XYf = deepcopy(deviation(enkf.ensa).S)
# 		rmul!(XYf, 1/sqrt(Ne-1))
#
# 		Yf = view(XYf,1,:)
# 		Xf = view(XYf,2:Nx+1, :)
#
# 		#Construct the observation with the un-inflated measurement
# 		@inbounds for i=1:Ne
# 			col = view(ens.state.S,:,i)
# 			enkf.ensa.S[1,i] =  h(t, col)[idx1] + enkf.ϵy.m[idx1] + dot(enkf.ϵy.σ[idx1,:], randn(Ny))
# 		end
# 		# @show dot(Yf,Yf)
# 		# Since Yf is a vector (Yf Yf') is reudced to a scalar
# 		"Analysis step with representers, Evensen, Leeuwen et al. 1998"
# 		K = (view(locXY,:,idx1).* (Xf*Yf))
# 		# @show K./dot(Yf,Yf)
# 		# K ./= dot(Yf, Yf)
# 		# Bᵀb = K*(ylocal .- view(enkf.ensa.S,1,:))
# 		#In-place analysis step with gemm!
# 		BLAS.gemm!('N', 'T', 1/dot(Yf,Yf), K, ylocal .- view(enkf.ensa.S,1,:), 1.0, ens.state.S)
# 	end
# 	return ens
# end
