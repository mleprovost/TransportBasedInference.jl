export LowRankStochEnKF

"""
$(TYPEDEF)

A structure for the stochastic ensemble Kalman filter (sEnKF)

References:

Evensen, G. (1994). Sequential data assimilation with a nonlinear quasi‐geostrophic model using Monte Carlo methods to forecast error statistics. Journal of Geophysical Research: Oceans, 99(C5), 10143-10162.
## Fields
$(TYPEDFIELDS)
"""

struct LowRankStochEnKF<:SeqFilter
    "Filter function"
    G::Function

    "Standard deviations of the measurement noise distribution"
    ϵy::AdditiveInflation

	"Observation dimension"
	Ny::Int64

	"State dimension"
	Nx::Int64

	"Rank of the observation space ry"
	ry::Int64

	"Rank of the state space rx"
	rx::Int64

    "Time step dynamic"
    Δtdyn::Float64

    "Time step observation"
    Δtobs::Float64

	"cachenoise"
	cachenoise::Array{Float64, 2}

	"cachey"
	cachey::Array{Float64, 2}

	"cachex"
	cachex::Array{Float64, 2}

    "Boolean: is state vector filtered"
    isfiltered::Bool


end

function LowRankStochEnKF(G::Function, ϵy::AdditiveInflation, Ny, Nx, ry, rx, Ne, Δtdyn, Δtobs;
	                      isfiltered = false)
    @assert norm(mod(Δtobs, Δtdyn))<1e-12 "Δtobs should be an integer multiple of Δtdyn"

    return LowRankStochEnKF(G, ϵy, Ny, Nx, ry, rx, Δtdyn, Δtobs,
	       zeros(Ny, Ne), zeros(ry, Ne), zeros(rx, Ne),
		   isfiltered)
end

# If no filtering function is provided, use the identity in the constructor.
function LowRankStochEnKF(ϵy::AdditiveInflation, Ny, Nx, ry, rx, Ne, Δtdyn, Δtobs;
	                      isfiltered = false)
    @assert norm(mod(Δtobs, Δtdyn))<1e-12 "Δtobs should be an integer multiple of Δtdyn"

	return LowRankStochEnKF(x->x, ϵy, Ny, Nx, ry, rx, Δtdyn, Δtobs,
		   zeros(Ny, Ne), zeros(ry, Ne), zeros(rx, Ne),
		   isfiltered)
end



function Base.show(io::IO, enkf::LowRankStochEnKF)
	println(io,"Low-rank Stochastic EnKF  with filtered = $(enkf.isfiltered)")
end


function (enkf::LowRankStochEnKF)(X, ystar::Array{Float64,1}, t::Float64, U, V; laplace::Bool=false)

    Ny = size(ystar, 1)
    Nx = size(X, 1)-Ny
    Ne = size(X, 2)

    meas  = X[1:Ny,:]
    state = X[Ny+1:Ny+Nx,:]

	Lx = LinearTransform(state)

	# Need the covariance to perform the localisation
	if laplace == false
		enkf.cachenoise .= enkf.ϵy.σ*randn(Ny, Ne) .+ enkf.ϵy.m
	else
		enkf.cachenoise .= sqrt(2.0)*enkf.ϵy.σ*rand(Laplace(),(Ny, Ne)) .+ enkf.ϵy.m
	end
	enkf.cachey .= U'*(enkf.ϵy.σ\(meas .- copy(mean(meas; dims = 2)[:,1])))
	enkf.cachey .-= mean(enkf.cachey; dims = 2)[:,1]
	enkf.cachey .*= 1/sqrt(Ne-1)

	enkf.cachex .= V'*(Lx.L\(state .- Lx.μ))
	enkf.cachex .-= mean(enkf.cachex; dims = 2)[:,1]
	enkf.cachex .*= 1/sqrt(Ne-1)

	ϵbrevepert = U'*(enkf.ϵy.σ\(enkf.cachenoise .- mean(enkf.cachenoise; dims =2)[:,1]))
	ϵbrevepert .-= mean(ϵbrevepert; dims = 2)[:,1]
	ϵbrevepert .*= 1/sqrt(Ne-1)

	"Low-rank analysis step with representers, Evensen, Leeuwen et al. 1998"
	b̆ = (enkf.cachey*enkf.cachey' + ϵbrevepert*ϵbrevepert')\(U'*(enkf.ϵy.σ\(ystar .- (meas + enkf.cachenoise))))
	view(X,Ny+1:Ny+Nx,:) .+= Lx.L*V*(enkf.cachex*enkf.cachey')*b̆

	return X
end
