export SparseRadialSMF, AdaptiveSparseRadialSMF

# Structure for the stochastic transport map
struct SparseRadialSMF<:SeqFilter
	"Filter function"
	G::Function

	"Observation operator"
	h::Function

	"Multiplicative inflation"
	β::Float64

	"Inflation for the measurement noise distribution"
	ϵy::AdditiveInflation

	"Sparse radial map"
	S::SparseRadialMap

	"Observation dimension"
	Ny::Int64

	"State dimension"
	Nx::Int64

    "Time step dynamic"
    Δtdyn::Float64

    "Time step observation"
    Δtobs::Float64

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


function SparseRadialSMF(G::Function, h::Function, β::Float64, ϵy::AdditiveInflation,
						 p::Array{Array{Int64,1},1}, γ, λ, δ, κ,
						 Ny, Nx, Ne,
				         Δtdyn::Float64, Δtobs::Float64,
						 dist::Array{Float64,2}, idx::Array{Int64,2};
						 isfiltered::Bool = false, islocalized::Bool = true)
	#Create the map with scalar assimlation of the data
	if islocalized == true
		S = SparseRadialMap(Nx+1, p; γ = γ, λ = λ, δ =  δ, κ = κ)
	else
		S = SparseRadialMap(Ny+Nx, p; γ = γ, λ = λ, δ =  δ, κ = κ)
	end
	return SparseRadialSMF(G, h, β, ϵy, S, Ny, Nx, Δtdyn, Δtobs, dist, idx,
	                       zeros(Nx+1, Ne), isfiltered, islocalized)
end

# Assimilate a scalar observation
function (smf::SparseRadialSMF)(X, ystar::Float64, t, idx::Array{Int64,1}; P::Parallel=serial)
	idx1, idx2 = idx
	Nx = smf.Nx
	Ny = smf.Ny
	Na = Nx+1
	NxX, Ne = size(X)
	cache = smf.cache
	fill!(cache, 0.0)
	# We add a +1 such that the scalar observation will remain the first entry
	perm = sortperm(view(smf.dist,:,idx2))

	Xinfl = copy(X)

	# Apply the multiplicative inflation
	Aβ = MultiplicativeInflation(smf.β)
	Aβ(Xinfl, Ny+1, Ny+Nx)

	# Generate samples from local likelihood
	@inbounds for i=1:Ne
		col = view(Xinfl, Ny+1:Ny+Nx, i)
		cache[1,i] = smf.h(col, t)[idx1] + smf.ϵy.m[idx1] + dot(smf.ϵy.σ[idx1,:], randn(Ny))
	end

	@view(cache[2:Na, :]) .= Xinfl[Ny .+ perm,:]

	# Update the LinearTransform smf.S.L
	smf.S.L.μ .= mean(cache; dims = 2)[:,1]
	smf.S.L.L.diag .= std(cache; dims = 2)[:,1]

	optimize(smf.S, cache, nothing; apply_rescaling=true, start = 2, P = P)

	#Generate local-likelihood samples with uninflated samples
	@inbounds for i=1:Ne
		col = view(X, Ny+1:Ny+Nx, i)
		cache[1,i] = smf.h(col, t)[idx1] + smf.ϵy.m[idx1] + dot(smf.ϵy.σ[idx1,:], randn(Ny))
	end

	@view(cache[2:Na,:]) .= X[Ny .+ perm,:]
	Sx = smf.S(cache; apply_rescaling = true, start = 2)

	if typeof(P) <: Serial
		inverse!(cache, Sx, smf.S, [ystar]; apply_rescaling = true)
	elseif typeof(P) <: Thread
		@inbounds Threads.@threads for i=1:Ne
			col = view(cache,:,i)
			inverse(col, view(Sx,:,i), smf.S, ystar; apply_rescaling = true)
		end
	end


	@view(X[Ny .+ perm,:]) .= cache[2:Na,:]
end

# Assimilate the entire observation vector
function (smf::SparseRadialSMF)(X, ystar, t; P::Parallel = serial, localized::Bool = true)
	Ny = smf.Ny
	@assert smf.Ny==size(ystar,1) "Wrong dimension of the observation"

	if smf.islocalized == true
		@inbounds for i=1:Ny
			smf(X, ystar[i], t, smf.idx[:,i]; P = P)
		end
	else
		# Perturbation of the measurements
		smf.ϵy(X, 1, Ny)

		optimize(smf.S, X, nothing; apply_rescaling=true, start = Ny+1)

		# Evaluate the transport map
		F = evaluate(smf.S, X; apply_rescaling=true, start = Ny+1)

		# Generate the posterior samples by partial inversion of the map

		inverse!(X, F, smf.S, ystar; appply_rescaling, start = Ny+1)
	end
	return X
end


## Adaptive Sparse Radial Map filter


# Structure for the stochastic transport map
struct AdaptiveSparseRadialSMF<:SeqFilter
	"Filter function"
	G::Function

	"Observation operator"
	h::Function

	"Multiplicative inflation"
	β::Float64

	"Inflation for the measurement noise distribution"
	ϵy::AdditiveInflation

	"Sparse radial map"
	S::SparseRadialMap

	"Observation dimension"
	Ny::Int64

	"State dimension"
	Nx::Int64

    "Time step dynamic"
    Δtdyn::Float64

    "Time step observation"
    Δtobs::Float64

	"Time step between reparametrization of the map"
	Δtfresh::Float64

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

function AdaptiveSparseRadialSMF(G::Function, h::Function, β::Float64, ϵy::AdditiveInflation,
						 p::Array{Array{Int64,1},1}, γ, λ, δ, κ,
						 Ny, Nx, Ne,
				         Δtdyn::Float64, Δtobs::Float64, Δtrefresh::Float64,
						 dist::Array{Float64,2}, idx::Array{Int64,2};
						 isfiltered::Bool = false, islocalized::Bool = true)
	#Create the map with scalar assimlation of the data
	if islocalized == true
		S = SparseRadialMap(Nx+1, p; γ = γ, λ = λ, δ =  δ, κ = κ)
	else
		S = SparseRadialMap(Ny+Nx, p; γ = γ, λ = λ, δ =  δ, κ = κ)
	end
	return AdaptiveSparseRadialSMF(G, h, β, ϵy, S, Ny, Nx, Δtdyn, Δtobs, Δtrefresh, dist, idx,
	                       zeros(Nx+1, Ne), isfiltered, islocalized)
end

# Assimilate a scalar observation
function (smf::AdaptiveSparseRadialSMF)(X, ystar::Float64, t, idx::Array{Int64,1}; P::Parallel=serial)
	idx1, idx2 = idx
	Nx = smf.Nx
	Ny = smf.Ny
	Na = Nx+1
	NxX, Ne = size(X)
	cache = smf.cache
	fill!(cache, 0.0)
	# We add a +1 such that the scalar observation will remain the first entry
	perm = sortperm(view(smf.dist,:,idx2))

	Xinfl = copy(X)

	# Apply the multiplicative inflation
	Aβ = MultiplicativeInflation(smf.β)
	Aβ(Xinfl, Ny+1, Ny+Nx)

	# Generate samples from local likelihood
	@inbounds for i=1:Ne
		col = view(Xinfl, Ny+1:Ny+Nx, i)
		cache[1,i] = smf.h(col, t)[idx1] + smf.ϵy.m[idx1] + dot(smf.ϵy.σ[idx1,:], randn(Ny))
	end

	@view(cache[2:Na, :]) .= Xinfl[Ny .+ perm,:]

	if abs(round(Int64,  t / smf.Δtfresh) - t / smf.Δtfresh)<1e-6

		order = fill(-1, Nx+1)
		nonid_rad = 15
		order[1] = 2
		order[2] = 1
		order[3] = 1
		fill!(view(order, 4:nonid_rad), 0)


		# Sgreedy = SparseRadialMp(Nx+1, -1; λ = smf.S.λ, δ = smf.S.δ, γ = smf.S.γ)
		Sgreedy = SparseRadialMap(cache, -1; λ = smf.S.λ, δ = smf.S.δ, γ = smf.S.γ)
		optimize(Sgreedy, cache, 2, order, "kfolds"; apply_rescaling=true, start = 2, verbose = false)
	else
		# Update the LinearTransform smf.S.L
		smf.S.L.μ .= mean(cache; dims = 2)[:,1]
		smf.S.L.L.diag .= std(cache; dims = 2)[:,1]

		optimize(Sgreedy, cache, nothing, nothing, nothing; apply_rescaling=true, start = 2, verbose = false)
	end

	#Generate local-likelihood samples with uninflated samples
	@inbounds for i=1:Ne
		col = view(X, Ny+1:Ny+Nx, i)
		cache[1,i] = smf.h(col, t)[idx1] + smf.ϵy.m[idx1] + dot(smf.ϵy.σ[idx1,:], randn(Ny))
	end

	@view(cache[2:Na,:]) .= X[Ny .+ perm,:]

	Sx = Sgreedy(cache; apply_rescaling = true, start = 2)

	if typeof(P) <: Serial
		inverse!(cache, Sx, Sgreedy, [ystar]; apply_rescaling = true)

	elseif typeof(P) <: Thread
		@inbounds Threads.@threads for i=1:Ne
			col = view(cache,:,i)
			inverse(col, view(Sx,:,i), Sgreedy, ystar; apply_rescaling = true)
		end
	end


	@view(X[Ny .+ perm,:]) .= cache[2:Na,:]
end

# Assimilate the entire observation vector
function (smf::AdaptiveSparseRadialSMF)(X, ystar, t; P::Parallel = serial, localized::Bool = true)
	Ny = smf.Ny
	@assert smf.Ny==size(ystar,1) "Wrong dimension of the observation"

	if smf.islocalized == true
		@inbounds for i=1:Ny
			smf(X, ystar[i], t, smf.idx[:,i]; P = P)
		end
	else
		# Perturbation of the measurements
		smf.ϵy(X, 1, Ny)

		optimize(smf.S, X, nothing; apply_rescaling=true, start = Ny+1)

		# Evaluate the transport map
		F = evaluate(smf.S, X; apply_rescaling=true, start = Ny+1)

		# Generate the posterior samples by partial inversion of the map

		inverse!(X, F, smf.S, ystar; appply_rescaling, start = Ny+1)
	end
	return X
end




















#
# function (T::SparseRadialSMF)(ens::EnsembleStateMeas{Nx, Ny, Ne}, ystar, t; P::Parallel = serial) where {Nx, Ny, Ne}
# 	@assert Ny==size(ystar,1) "Wrong dimension of the observation"
# 	# idx contains the dictionnary of the mapping
# 	# First line contains the range of integer 1:Ny
# 	# Second line contains the associated indice of each measurement
# 	@inbounds for i=1:Ny
# 		assimilate_scalar_obs(T, ens, T.idx[:,i], ystar[i], t; P = P)
# 	end
#
# 	return ens
# end
#
# function (smf::SparseRadialSMF)(X, ystar, t; P::Parallel = serial)
# 	Ny = smf.Ny
# 	@assert smf.Ny==size(ystar,1) "Wrong dimension of the observation"
# 	# idx contains the dictionnary of the mapping
# 	# First line contains the range of integer 1:Ny
# 	# Second line contains the associated indice of each measurement
# 	@inbounds for i=1:Ny
# 		smf(X, ystar[i], t, smf.idx[:,i]; P = P)
# 	end
#
# 	return X
# end

# Without localization of the observations
# function (smf::SparseRadialSMF)(X, ystar, t; P::Parallel = serial)
# 	Ny = smf.Ny
# 	@assert smf.Ny==size(ystar,1) "Wrong dimension of the observation"
#
# 	# Perturbation of the measurements
# 	smf.ϵy(X, 1, Ny)
#
# 	optimize(smf.S, X; start = Ny+1)
#
# 	# Evaluate the transport map
# 	F = evaluate(smf.S, X; start = Ny+1)
#
# 	# Generate the posterior samples by partial inversion of the map
#
# 	inverse!(X, F, smf.S, ystar; start = Ny+1)
#
# 	return X
# end
#
# ## Full map without local observation
#
#
# # Structure for the stochastic transport map
# struct RadialSMF<:SeqFilter
#
#     "Filter function"
#     G::Function

	# 	"Knothe-Rosenblatt rearrangement"
	# 	S::RadialMap
#
#     "Standard deviations of the measurement noise distribution"
#     ϵy::AdditiveInflation
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
# end
#
#
# function RadialSMF(Nx, Ny, p, γ, λ, δ, κ, G::Function,
# 	ϵy::AdditiveInflation, Δtdyn::Float64, Δtobs::Float64, islocal::Bool, isfiltered::Bool)
# 	#Create the map
# 	S = RadialMap(Nx+Ny, p; γ = γ, λ = λ, δ =  δ, κ = κ)
# 	return RadialSMF(S, G, ϵy, Δtdyn, Δtobs, islocal, isfiltered)
# end
#
#
# function (T::RadialSMF)(ens::EnsembleStateMeas{Nx, Ny, Ne}, ystar) where {Nx, Ny, Ne}
#
# # Perturb each observation
# T.ϵy(ens.meas)
# augmented_ens = EnsembleState(Nx+Ny, Ne)
# augmented_ens.S[1:Ny,:] = deepcopy(ens.meas.S)
# augmented_ens.S[Ny+1:Ny+Nx,:] = deepcopy(ens.state.S)
#
# run_optimization(T.S, augmented_ens, start = Ny+1)
#
# #Evaluate the map
# Sval  = T.S(augmented_ens)
#
#
# #Invert the map and update each ensemble member
# zplus = zeros(Ny+Nx)
# zplus[1:Ny] = deepcopy(ystar)
#
# @inbounds for i=1:Ne
# zplus[Ny+1:end] = zeros(Nx)
# invert_S(T.S, view(Sval,:,i), ystar, zplus)
# ens.state.S[:,i] = deepcopy(zplus[Ny+1:end])
# end
#
# return ens
#
# end

###################
# End of the code
################################################################################################################################################################

#
# function assimilate_scalar_obs(T::SparseRadialSMF, ens::EnsembleStateMeas{Nx, Ny, Ne}, idx, y, t) where {Nx, Ny, Ne}
# Na,_ = size(T.ensa)
# @assert Na == Nx+1 "Wrong dimension for the augmented EnsembleStateMeas"
#
# # We add a +1 such that the scalar observation will remain the first entry
# perm = sortperm(view(T.dist,:,idx))
#
# meas = zeros(Ne)
# #Generate local-likelihood samples with uninflated samples
# @inbounds for i=1:Ne
# 	col = view(ens.state.S,:,i)
# 	meas[i] = T.dyn.h(t, col)[idx] + T.ϵy.m[idx] + dot(T.ϵy.σ[idx,:], randn(Nx))
# end
#
# # Apply the multiplicative inflation
# Ŝ = deepcopy(mean(ens.state))
#
# @inbounds for i=1:Ne
# 	col = view(ens.state.S, :, i)
# 	T.ensa.S[2:Nx+1, i] .= Ŝ + T.β*(col - Ŝ)
# end
#
# # Generate samples from local likelihood with inflated state
# @inbounds for i=1:Ne
# 	col = view(T.ensa.S,2:Nx+1,i)
# 	T.ensa.S[1,i] = T.dyn.h(t, col)[idx] + T.ϵy.m[idx] + dot(T.ϵy.σ[idx,:], randn(Nx))
# end
#
#
# # Permute in-place the lines 2 to Nx+1 of ensa according to perm
# @inbounds for i=1:Ne
# 	permute!(view(T.ensa.S,2:Nx+1,i),perm)
# end
#
# #Run optimization
# run_optimization(T.S, T.ensa; start = 2)
#
# T.ensa.S[1,:] .= meas
# T.ensa.S[2:Nx+1,:] .= deepcopy(ens.state.S[perm,:])
#
# Sval = T.S(T.ensa)
#
# @inbounds for i=1:Ne
# zplus = view(T.ensa.S,:,i)
# invert_S(T.S, view(Sval,:,i), y, zplus)
# end
#
# ens.state.S[perm,:] = T.ensa.S[2:Nx+1,:];
# end
