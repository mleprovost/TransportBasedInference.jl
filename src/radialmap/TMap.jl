export SparseTMap, assimilate_scalar_obs, TMap

# Structure for the stochastic transport map
struct SparseTMap<:SeqFilter
	"Knothe-Rosenblatt rearrangement"
	S::SparseKRmap

	"Distance matrix"
	dist::Matrix

	"Dynamical system"
	dyn::DynamicalSystem

    "Filter function"
    G::Function

	"Multiplicative inflation"
	β::Float64

    "Inflation for the measurement noise distribution"
    ϵy::AdditiveInflation

    "Time step dynamic"
    Δtdyn::Float64

    "Time step observation"
    Δtobs::Float64

    "Boolean: is state vector filtered"
    isfiltered::Bool

	# Define cache
	"EnsembleState to hold the augmented state"
	ensa::EnsembleState

	"Index of measurement"
	idx::Array{Int64,2}
	# idx contains the dictionnary of the mapping
	# First line contains the range of integer 1:Ny
	# Second line contains the associated indice of each measurement



end


function SparseTMap(Nx, Ny, Ne, p::Array{Array{Int64,1}}, γ, λ, δ, κ,
					dist::Matrix, dyn::DynamicalSystem,
                    G::Function, β, ϵy::AdditiveInflation,
				    Δtdyn::Float64, Δtobs::Float64, isfiltered::Bool, idx::Array{Int64,2})
	#Create the map with scalar assimlation of the data
	S = SparseKRmap(Nx+1, p; γ = γ, λ = λ, δ =  δ, κ = κ)
	return SparseTMap(S, dist, dyn, G, β, ϵy, Δtdyn, Δtobs, isfiltered,  EnsembleState(Nx+1, Ne), idx)
end




function assimilate_scalar_obs(T::SparseTMap, ens::EnsembleStateMeas{Nx, Ny, Ne}, idx::Array{Int64,1}, y, t; P::Parallel=serial) where {Nx, Ny, Ne}

idx1, idx2 = idx

Na = Nx+1
# We add a +1 such that the scalar observation will remain the first entry
perm = sortperm(view(T.dist,:,idx2))

ensinfl = deepcopy(ens.state)

# Apply the multiplicative inflation
Aβ = MultiplicativeInflation(T.β)
Aβ(ensinfl)


# Generate samples from local likelihood
@inbounds for i=1:Ne
	col = view(ensinfl.S,:,i)
	T.ensa.S[1,i] = T.dyn.h(t, col)[idx1] + T.ϵy.m[idx1] + dot(T.ϵy.σ[idx1,:], randn(Ny))
end


T.ensa.S[2:Na,:] .= deepcopy(ensinfl.S[perm,:])

#Run optimization
@time run_optimization(T.S, T.ensa; start = 2, P = P)

#Generate local-likelihood samples with uninflated samples
@time @inbounds for i=1:Ne
	col = view(ens.state.S,:,i)
	T.ensa.S[1,i] = T.dyn.h(t, col)[idx1] + T.ϵy.m[idx1] + dot(T.ϵy.σ[idx1,:], randn(Ny))
end

T.ensa.S[2:Na,:] .= ens.state.S[perm,:]

Sval = T.S(T.ensa)
# @show Sval[:,1]

# T.ensa.S[:,1]

if typeof(P)==Serial
	@inbounds for i=1:Ne
		zplus = view(T.ensa.S,:,i)
		invert_S(T.S, view(Sval,:,i), y, zplus)
	end
else
	@inbounds Threads.@threads for i=1:Ne
		zplus = view(T.ensa.S,:,i)
		invert_S(T.S, view(Sval,:,i), y, zplus)
	end
end

ens.state.S[perm,:] = T.ensa.S[2:Na,:];
end

function (T::SparseTMap)(ens::EnsembleStateMeas{Nx, Ny, Ne}, ystar, t; P::Parallel = serial) where {Nx, Ny, Ne}
	@assert Ny==size(ystar,1) "Wrong dimension of the observation"
	# idx contains the dictionnary of the mapping
	# First line contains the range of integer 1:Ny
	# Second line contains the associated indice of each measurement
	@inbounds for i=1:Ny
		assimilate_scalar_obs(T, ens, T.idx[:,i], ystar[i], t; P = P)
	end

	return ens
end

function (T::SparseTMap)(X, ystar, t; P::Parallel = serial)
	@assert Ny==size(ystar,1) "Wrong dimension of the observation"
	# idx contains the dictionnary of the mapping
	# First line contains the range of integer 1:Ny
	# Second line contains the associated indice of each measurement
	@inbounds for i=1:Ny
		assimilate_scalar_obs(T, X, T.idx[:,i], ystar[i], t; P = P)
	end

	return X
end

## Full map without local observation

# Structure for the stochastic transport map
struct TMap<:SeqFilter
	"Knothe-Rosenblatt rearrangement"
	S::KRmap

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


function TMap(Nx, Ny, p, γ, λ, δ, κ, G::Function,
	ϵy::AdditiveInflation, Δtdyn::Float64, Δtobs::Float64, islocal::Bool, isfiltered::Bool)
	#Create the map
	S = KRmap(Nx+Ny, p; γ = γ, λ = λ, δ =  δ, κ = κ)
	return TMap(S, G, ϵy, Δtdyn, Δtobs, islocal, isfiltered)
end


function (T::TMap)(ens::EnsembleStateMeas{Nx, Ny, Ne}, ystar) where {Nx, Ny, Ne}

# Perturb each observation
T.ϵy(ens.meas.S)
augmented_ens = EnsembleState(Nx+Ny, Ne)
augmented_ens.S[1:Ny,:] = deepcopy(ens.meas.S)
augmented_ens.S[Ny+1:Ny+Nx,:] = deepcopy(ens.state.S)

run_optimization(T.S, augmented_ens, start = Ny+1)

#Evaluate the map
Sval  = T.S(augmented_ens)


#Invert the map and update each ensemble member
zplus = zeros(Ny+Nx)
zplus[1:Ny] = deepcopy(ystar)

@inbounds for i=1:Ne
zplus[Ny+1:end] = zeros(Nx)
invert_S(T.S, view(Sval,:,i), ystar, zplus)
ens.state.S[:,i] = deepcopy(zplus[Ny+1:end])
end

return ens

end



#
# function assimilate_scalar_obs(T::SparseTMap, ens::EnsembleStateMeas{Nx, Ny, Ne}, idx, y, t) where {Nx, Ny, Ne}
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
