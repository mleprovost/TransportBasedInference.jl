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

	"Boolean: is state vector filtered"
	isfiltered::Bool
end

function Base.show(io::IO, smf::StochMapFilter)
	println(io,"Stochastic Map Filter  with filtered = $(smf.isfiltered)")
end

function (smf::StochMapFilter)(X, ystar::Array{Float64,1}; laplace::Bool=false)
	Ny = smf.Ny
	Nx = smf.Nx
	M = smf.M

	Nystar = size(ystar, 1)
	Nypx, Ne = size(X)

	@assert Nystar == Ny "Size of ystar is not consistent with Ny"
	@assert Nypx == Ny + Nx "Size of X is not consistent with Ny or Nx"
	@assert Ne == Neystar "Size of X and Ystar are not consistent"
	@show getcoeff(M[end])

	# Perturbation of the measurements
	smf.ϵy(X, 1, Ny; laplace = laplacw)
	# Optimize the transport map
	M.L = LinearTransform(X; diag = true)
	# M.L needs to be re-evaluated at every time step

	optimize(M, X, nothing; withconstant = withconstant, withqr = withqr,
			   verbose = verbose, start = Ny+1, P = P)
	# Evaluate the transport map
	F = evaluate(M, X; start = Ny+1, P = P)
	inverse!(X, F, M, ystar; start = Ny+1, P = P)
	@show getcoeff(M[end])

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
