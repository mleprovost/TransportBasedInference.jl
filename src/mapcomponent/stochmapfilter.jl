export StochMapFilter

struct StochMapFilter<:SeqFilter
		"Filter function"
		G::Function

		"Standard deviations of the measurement noise distribution"
		ϵy::AdditiveInflation

        "HermiteMap"
        M::HermiteMap

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

		"Boolean: is state vector filtered"
		isfiltered::Bool
end

function Base.show(io::IO, smf::StochMapFilter)
	println(io,"Stochastic Map Filter with filtered = $(smf.isfiltered)")
end

function (smf::StochMapFilter)(X, ystar::Array{Float64,1}, t::Float64)
	Ny = smf.Ny
	Nx = smf.Nx
	@show t

	Nystar = size(ystar, 1)
	Nypx, Ne = size(X)

	@assert Nystar == Ny "Size of ystar is not consistent with Ny"
	@assert Nypx == Ny + Nx "Size of X is not consistent with Ny or Nx"

	# Perturbation of the measurements
	smf.ϵy(X, 1, Ny)

	L = LinearTransform(X; diag = true)

	M = HermiteMap(10, Ny+Nx, L, smf.M.C)

	@show getcoeff(M[6])
	# Re-optimize the map with kfolds
	if mod(t, smf.Δtfresh) == 0


		# Perform a kfold optimization of the map
		optimize(M, X, "kfolds"; withconstant = false, withqr = true,
				   verbose = false, start = Ny+1, P = serial, hessprecond = true)

	else
		# Only optimize the existing coefficients of the basis
		optimize(M, X, nothing; withconstant = false, withqr = true,
				   verbose = false, start = Ny+1, P = serial, hessprecond = true)
	end

	# Evaluate the transport map
	F = evaluate(M, X; apply_rescaling = true, start = Ny+1, P = serial)

	# Rescale ystar
	ystar .-= view(M.L.μ,1:Ny)
	ystar ./= M.L.L.diag[1:Ny]

	# Generate the posterior samples by partial inversion of the map
	inverse!(X, F, M, ystar; start = Ny+1, P = serial)
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
