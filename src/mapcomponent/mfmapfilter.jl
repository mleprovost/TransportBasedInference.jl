export MfMapFilter

#multifidelity version of the non-adaptive stochastic map filter
#starting out with implementing the regular stochastic map filter with Linear/RBF parametrization
#what even is a RBF?
struct MfMapFilter<:SeqFilter
		"Filter function"
		G::Function

		"Standard deviations of the measurement noise distribution"
		ϵy::AdditiveInflation

        "HermiteMap" #TODO change to non-Hermite map
        M::HermiteMap

        "Observation dimension"
        Ny::Int64

        "State dimension"
        Nx::Int64

        "Time step dynamic"
        Δtdyn::Float64

        "Time step observation"
        Δtobs::Float64

		#removed Δtfresh

		"Boolean: is state vector filtered"
		isfiltered::Bool
end

function Base.show(io::IO, mfmf::MfMapFilter)
	println(io,"Multifidelity Map Filter with filtered = $(mfmf.isfiltered)")
end

function (mfmf::MfMapFilter)(X, ystar::Array{Float64,1}, t::Float64)
	Ny = mfmf.Ny
	Nx = mfmf.Nx
	@show t

	Nystar = size(ystar, 1)
	Nypx, Ne = size(X)

	@assert Nystar == Ny "Size of ystar is not consistent with Ny"
	@assert Nypx == Ny + Nx "Size of X is not consistent with Ny or Nx"

	# Perturbation of the measurements
	mfmf.ϵy(X, 1, Ny)

	L = LinearTransform(X; diag = true) #this computes a standardization for X (mean and diagonal stddev)

	M = HermiteMap(10, Ny+Nx, L, mfmf.M.C)

	@show getcoeff(M[6])
	# # Re-optimize the map with kfolds
	# if mod(t, mfmf.Δtfresh) == 0
	#
	#
	# 	# Perform a kfold optimization of the map
	# 	optimize(M, X, "kfolds"; withconstant = false, withqr = true,
	# 			   verbose = false, start = Ny+1, P = serial, hessprecond = true)

	#else
	# Optimize the existing coefficients of the basis (this estimates the KR rearrangement given a certain basis, doesn't adaptively choose functions)
	optimize(M, X, nothing; withconstant = false, withqr = true,
				   verbose = false, start = Ny+1, P = serial, hessprecond = true)
	#end

	# Evaluate the transport map
	F = evaluate(M, X; apply_rescaling = true, start = Ny+1, P = serial) #push pa (to standard normal)

	# Rescale ystar
	ystar .-= view(M.L.μ,1:Ny)
	ystar ./= M.L.L.diag[1:Ny]

	# Generate the posterior samples by partial inversion of the map
	inverse!(X, F, M, ystar; start = Ny+1, P = serial) #shove ma (to posterior)
	# @show getcoeff(M[Nypx])
	# @show "after inversion"
	# @show norm(X)
	return X
end

#
# function assimilate_obs(M::HermiteMap, X, ystar, Ny, Nx; withconstant::Bool = false,
#                         withqr::Bool = false, verbose::Bool = false, P::Parallel = serial)
#
#         Nystar = size(ystar, 1)
#         Nypx, Ne = size(X)
#
#         @assert Nystar == Ny "Size of ystar is not consistent with Ny"
#         @assert Nypx == Ny + Nx "Size of X is not consistent with Ny or Nx"
#         @assert Ne == Neystar "Size of X and Ystar are not consistent"
#
#         # Optimize the transport map
#         M = optimize(M, X, "kfold"; withconstant = withconstant, withqr = withqr,
#                                verbose = verbose, start = Ny+1, P = P)
#
#         # Evaluate the transport map
#         F = evaluate(M, X; start = Ny+1, P = P)
#
#         inverse!(X, F, M, ystar; start = Ny+1, P = P)
#
#         return X
# end
