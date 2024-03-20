export HermiteSMF, FixedHermiteSMF

struct HermiteSMF<:SeqFilter
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

function Base.show(io::IO, smf::HermiteSMF)
	println(io,"Hermite Stochastic Map Filter with filtered = $(smf.isfiltered)")
end

function (smf::HermiteSMF)(X, ystar::Array{Float64,1}, t::Float64)
	Ny = smf.Ny
	Nx = smf.Nx

	Nystar = size(ystar, 1)
	Nypx, Ne = size(X)

	@assert Nystar == Ny "Size of ystar is not consistent with Ny"
	@assert Nypx == Ny + Nx "Size of X is not consistent with Ny or Nx"

	# Perturbation of the measurements
	smf.ϵy(X, 1, Ny)

	μYX = mean(X; dims = 2)[:,1]
	σYX = std(X; dims = 2)[:,1]

	X̃ = Diagonal(σYX) \ (X .- μYX)
	# if abs(round(Int64,  t / smf.Δtfresh) - t / smf.Δtfresh)<1e-6
	# M = HermiteMap(30, X̃; diag = true, b = "CstLinProHermiteBasis")
	# Perform a kfold optimization of the map
	# optimize(M, X̃, "split"; maxterms = 20, withconstant = false, withqr = true,
	# verbose = false, start = Ny+1, P = serial, hessprecond = true)

	M = totalordermap(X̃, 2; b = "CstLinProHermiteBasis")
	optimize(M, X̃, nothing; withconstant = false, withqr = true,
			 verbose = false, start = Ny+1, P = serial, hessprecond = false)

	# Evaluate the transport map
	F = evaluate(M, X̃; apply_rescaling = false, start = Ny+1, P = serial)

	# Rescale ystar
	ỹstar = Diagonal(σYX[1:Ny])\(copy(ystar) - μYX[1:Ny])

	# Generate the posterior samples by partial inversion of the map
	hybridinverse!(X̃, F, M, ỹstar; start = Ny+1, P = serial)
	# @show getcoeff(M[Nypx])
	# @show "after inversion"
	# @show norm(X)
	X[Ny+1:Ny+Nx,:] .= μYX[Ny+1:Ny+Nx] .+ Diagonal(σYX[Ny+1:Ny+Nx])*X̃[Ny+1:Ny+Nx,:]
	return X
end



struct FixedHermiteSMF<:SeqFilter
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

		"Boolean: is state vector filtered"
		isfiltered::Bool
end

function Base.show(io::IO, smf::FixedHermiteSMF)
	println(io,"Fixed Order Hermite Stochastic Map Filter with filtered = $(smf.isfiltered)")
end

function (smf::FixedHermiteSMF)(X, ystar::Array{Float64,1}, t::Float64)
	Ny = smf.Ny
	Nx = smf.Nx

	Nystar = size(ystar, 1)
	Nypx, Ne = size(X)

	@assert Nystar == Ny "Size of ystar is not consistent with Ny"
	@assert Nypx == Ny + Nx "Size of X is not consistent with Ny or Nx"

	# Perturbation of the measurements
	smf.ϵy(X, 1, Ny)

	L = LinearTransform(X; diag = true)
	M = HermiteMap(smf.M.m, smf.M.Nx, L, smf.M.C)
	clearcoeff!(M)

	M = totalordermap(X, 2; b = "CstLinProHermiteBasis")

	optimize(M, X, nothing; withconstant = false, withqr = true,
			 verbose = false, start = Ny+1, P = serial, hessprecond = true)

	# Evaluate the transport map
	F = evaluate(M, X; apply_rescaling = true, start = Ny+1, P = serial)

	# Generate the posterior samples by partial inversion of the map
	hybridinverse!(X, F, M, ystar; start = Ny+1, P = serial)

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
