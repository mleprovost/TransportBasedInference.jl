export seqassim

# Create a function to perform the sequential assimilation for any sequential filter SeqFilter
function seqassim(F::StateSpace, data::SyntheticData, J::Int64, ϵx::InflationType, algo::SeqFilter, X, Ny, Nx, t0::Float64)

Ne = size(X, 2)

step = ceil(Int, algo.Δtobs/algo.Δtdyn)
statehist = Array{Float64,2}[]
push!(statehist, deepcopy(X[Ny+1:Ny+Nx,:]))

n0 = ceil(Int64, t0/algo.Δtobs) + 1
Acycle = n0:n0+J-1
tspan = (t0, t0 + algo.Δtobs)
# Define the dynamical system
prob = ODEProblem(F.f, zeros(Nx), tspan)

# Run particle filter
for i=1:length(Acycle)
    # Forecast
	tspan = (t0+(i-1)*algo.Δtobs, t0+i*algo.Δtobs)
	prob = remake(prob; tspan=tspan)

	prob_func(prob,i,repeat) = ODEProblem(prob.f, X[Ny+1:Ny+Nx,i],prob.tspan)

	ensemble_prob = EnsembleProblem(prob,output_func = (sol,i) -> (sol[end], false),
	prob_func=prob_func)
	sim = solve(ensemble_prob, Tsit5(), EnsembleThreads(),trajectories = Ne,
				dense = false, save_everystep=false);

	@inbounds for i=1:Ne
	    X[Ny+1:Ny+Nx, i] .= deepcopy(sim[i])
	end


    # Assimilation # Get real measurement # Fix this later # Things are shifted in data.yt
    ystar = data.yt[:,Acycle[i]]
	# Replace at some point by realobserve(model.h, t0+i*model.Δtobs, ens)
	# Perform inflation for each ensemble member
	ϵx(X, Ny+1, Ny+Nx)

	# Compute measurements
	observe(F.h, t0+i*algo.Δtobs, X, Ny, Nx)

    # Generate posterior samples.
	# Note that the additive inflation of the observation is applied within the sequential filter.
    X = algo(X, ystar)#, t0+i*algo.Δtobs)

	# Filter state
	if algo.isfiltered == true
		for i=1:Ne
			statei = view(X, Ny+1:Ny+Nx, i)
			statei .= algo.G(statei)
		end
	end

    push!(statehist, deepcopy(X[Ny+1:Ny+Nx,:]))
	end

return statehist
end

#Version with Localization metric

# Create a function to perform the sequential assimilation for any sequential filter SeqFilter
function seqassim(F::StateSpace, data::SyntheticData, J::Int64, ϵx::InflationType, algo::SeqFilter, X, Ny, Nx,t0::Float64, Loc::Localization)

step = ceil(Int, algo.Δtobs/algo.Δtdyn)
statehist = Array{Float64,2}[]
push!(statehist, deepcopy(viewstate(X, Ny, Nx)))


n0 = ceil(Int64, t0/algo.Δtobs) + 1;
Acycle = n0:n0+J-1
tspan = (t0, t0 + algo.Δtobs)
prob = ODEProblem(F.f, zeros(Nx), tspan)

# Run particle filter
@inbounds for i=1:length(Acycle)
    # Forecast
	tspan = (t0+(i-1)*algo.Δtobs, t0+i*algo.Δtobs)
	prob = remake(prob; tspan=tspan)

	prob_func(prob,i,repeat) = ODEProblem(prob.f,view(X,Ny+1:Ny+Nx,i),prob.tspan)

	ensemble_prob = EnsembleProblem(prob, output_func = (sol,i) -> (sol[end], false),
	prob_func=prob_func)
	sim = solve(ensemble_prob, Tsit5(), EnsembleThreads(),trajectories = Ne,
				dense = false, save_everystep=false);

	@inbounds for i=1:Ne
		view(X, Ny+1:Ny+Nx, i) .= sim[i]
	end

    # Assimilation # Get real measurement # Fix this later # Things are shifted in data.yt
    ystar = deepcopy(view(data.yt,:,Acycle[i]))
	# Replace at some point by realobserve(model.h, t0+i*model.Δtobs, ens)
	# Perform inflation for each ensemble member
	ϵx(X, Ny+1, Ny+Nx)
	# Compute measurements
	observe(F.h, t0+i*algo.Δtobs, X, Ny, Nx)
    # Generate posterior samples
    X = algo(X, ystar, t0+i*algo.Δtobs, Loc)

	# Filter state
	if algo.isfiltered == true
		for i=1:Ne
			statei = view(X, Ny+1:Ny+Nx, i)
			statei .= algo.G(statei)
		end
	end

    push!(statehist, deepcopy(X[Ny+1:Ny+Nx,:]))
end

return statehist
end
