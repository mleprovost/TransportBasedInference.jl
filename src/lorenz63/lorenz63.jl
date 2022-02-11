export lorenz63!, setup_lorenz63, generate_lorenz63, benchmark_lorenz63, benchmark_srmf_lorenz63, benchmark_sadaptivermf_lorenz63

"""
    lorenz63!(du,u,p,t)

Compute in-place the right-hand-side of the Lorenz-63 system for a state `u` at time `t`,
and store it in `du`. `p` is vector of user-defined parameters.
"""
function lorenz63!(du,u,p,t)
    du[1] = 10.0*(u[2]-u[1])
    du[2] = u[1]*(28.0-u[3]) - u[2]
    du[3] = u[1]*u[2] - (8/3)*u[3]
    return du
end


function generate_lorenz63(model::Model, x0::Array{Float64,1}, J::Int64)

    @assert model.Nx == size(x0,1) "Error dimension of the input"
    xt = zeros(model.Nx,J)
    x = deepcopy(x0)
    yt = zeros(model.Ny,J)
    tt = zeros(J)

    t0 = 0.0

    step = ceil(Int, model.Δtobs/model.Δtdyn)
    tspan = (t0, t0 + model.Δtobs)
    prob = ODEProblem(model.F.f,x,tspan)

    for i=1:J
    	# Run dynamics and save results
    	tspan = (t0 + (i-1)*model.Δtobs, t0 + i*model.Δtobs)
    	prob = remake(prob, tspan = tspan)

    	sol = solve(prob, Tsit5(), dt = model.Δtdyn, adaptive = false, dense = false, save_everystep = false)
    	x .= deepcopy(sol.u[end])
    	# for j=1:step
    	# 	t = t0 + (i-1)*algo.Δtobs+(j-1)*algo.Δtdyn
        # 	_, x = model.f(t+(i-1)*model.Δtdyn, x)
    	# end
    	model.ϵx(x)

    	# Collect observations
    	tt[i] = deepcopy(i*model.Δtobs)
    	xt[:,i] = deepcopy(x)
    	yt[:,i] = deepcopy(model.F.h(x, tt[i]))
        # model.ϵy(yt[:,i])
		yt[:,i] .+= model.ϵy.m + model.ϵy.σ*randn(model.Ny)
    end
    	return SyntheticData(tt, model.Δtdyn, x0, xt, yt)
end

function spin_lorenz63(model::Model, data::SyntheticData, Ne::Int64, path::String)
	@assert path[1]=='/' && path[end]=='/' "This function expects a / at the extremities of path"

	# Set initial condition
	X = zeros(model.Ny + model.Nx, Ne)
	X[model.Ny+1:model.Ny+model.Nx,:] .= rand(model.π0, Ne)#sqrt(model.C0)*randn(model.Nx, Ne) .+ model.m0

	J = model.Tspinup
	t0 = 0.0
	F = model.F
	enkf = StochEnKF(x->x, model.ϵy, model.Δtdyn, model.Δtobs, false, false)

	statehist = seqassim(F, data, J, model.ϵx, enkf, X, model.Ny, model.Nx, t0)

    save(path*"set_up_Ne"*string(Ne)*".jld", "state", statehist, "Ne", Ne, "x0", data.x0, "tt", data.tt, "xt", data.xt, "yt", data.yt)


    # return statehist
	_,_,rmse_mean,_ = metric_hist(rmse, data.xt[:,1:J], statehist[2:end])
	println("Ne "*string(Ne)* " RMSE: "*string(rmse_mean))
	# Save data
	save(path*"set_up_Ne"*string(Ne)*".jld", "X", statehist[end], "Ne", Ne, "x0", data.x0, "tt", data.tt, "xt", data.xt, "yt", data.yt)
end


function setup_lorenz63(path::String, Ne_array::Array{Int64,1})
    Nx = 3
    Ny = 3
    Δtdyn = 0.05
    Δtobs = 0.1

    σx = 1e-6
    σy = 2.0

    ϵx = AdditiveInflation(Nx, zeros(Nx), σx)
    ϵy = AdditiveInflation(Ny, zeros(Ny), σy)
    tspinup = 200.0
    Tspinup = 2000
    tmetric = 400.0
    Tmetric = 4000
    t0 = 0.0
    tf = 600.0
    Tf = 6000

    Tburn = 2000
    Tstep = 4000
    Tspinup = 2000

    f = lorenz63!
    h(x, t) = x
    # Create a local version of the observation operator
    h(x, t, idx) = x[idx]
	F = StateSpace(lorenz63!, h)

    model = Model(Nx, Ny, Δtdyn, Δtobs, ϵx, ϵy, MvNormal(zeros(Nx), Matrix(1.0*I, Nx, Nx)), Tburn, Tstep, Tspinup, F);

    # Set initial condition
    x0 = rand(model.π0)
    # x0 = [0.645811242103507;  -1.465126216973632;   0.385227725149822];

    # Run dynamics and generate data
    data = generate_lorenz63(model, x0, model.Tspinup+model.Tstep);


    for Ne in Ne_array
        spin_lorenz63(model, data, Ne, path)
    end

    return model, data
end


function benchmark_lorenz63(model::Model, data::SyntheticData, path::String, Ne_array::Array{Int64,1}, β_array::Array{Float64,1})
@assert path[1]=='/' && path[end]=='/' "This function expects a / at the extremities of path"

#Store all the metric per number of ensemble members
metric_list = []

@showprogress for Ne in Ne_array
    metric_Ne = Metrics[]
    for β in β_array
    @show Ne, β
    # Load file
    X0 = load(path*"set_up_Ne"*string(Ne)*".jld", "X")

    X = zeros(model.Ny + model.Nx, Ne)
    X[model.Ny+1:model.Ny+model.Nx,:] .= copy(X0)
    J = model.Tstep
    t0 = model.Tspinup*model.Δtobs
    F = model.F
    enkf = StochEnKF(x->x, model.ϵy, model.Δtdyn, model.Δtobs, false, false)

    # Use this multi-additive inflation
    ϵx = MultiAddInflation(model.Nx, β, zeros(model.Nx), model.ϵx.Σ, model.ϵx.σ)

    # @time enshist = seqassim(dyn, data, J, ϵx, enkf, ens, t0)
	@time statehist = seqassim(F, data, J, model.ϵx, enkf, X, model.Ny, model.Nx, t0);

    metric = post_process(data, model, J, statehist)
    push!(metric_Ne, deepcopy(metric))
    println("Ne "*string(Ne)*"& β "*string(β)*" RMSE: "*string(metric.rmse_mean))
    end
    push!(metric_list, deepcopy(metric_Ne))
end

return metric_list
end


# Routine to benchmark the stochastic radial map filter
function benchmark_srmf_lorenz63(model::Model, data::SyntheticData, path::String, Ne_array::Array{Int64,1}, β_array::Array{Float64,1}, p::Int64)
@assert path[1]=='/' && path[end]=='/' "This function expects a / at the extremities of path"

#Store all the metric per number of ensemble members
metric_hist = []

@showprogress for Ne in Ne_array
    metric_Ne = Metrics[]
    for β in β_array

	@show Ne, β
    # Load file
    X0 = load(path*"set_up_Ne"*string(Ne)*".jld", "X")

    X = zeros(model.Ny + model.Nx, Ne)
    X[model.Ny+1:model.Ny+model.Nx,:] .= copy(X0)

    J = model.Tstep
    t0 = model.Tspinup*model.Δtobs
    F = model.F
	dist = Float64.(metric_lorenz(model.Nx))
	idx = vcat(collect(1:model.Ny)',collect(1:model.Ny)')

    order = [[-1], [p; p], [-1; p; 0], [-1; p; p; 0]]

	γ = 2.0
	λ = 0.0
	δ = 1e-8
	κ = 10.0

	smf = SparseRadialSMF(x->x, F.h, β, model.ϵy, order, γ, λ, δ, κ,
                      model.Ny, model.Nx, Ne,
                      model.Δtdyn, model.Δtobs,
                      dist, idx; islocalized = true)

    @time statehist = seqassim(F, data, J, model.ϵx, smf, X, model.Ny, model.Nx, t0);

    metric = post_process(data, model, J, statehist)
    push!(metric_Ne, deepcopy(metric))
    println("Ne "*string(Ne)*"& β "*string(β)*" RMSE: "*string(metric.rmse_mean))
    end
    push!(metric_hist, deepcopy(metric_Ne))
end

return metric_hist
end


# Routine to benchmark the stochastic adaptive radial map filter
function benchmark_sadaptivermf_lorenz63(model::Model, data::SyntheticData, path::String, Ne_array::Array{Int64,1}, β_array::Array{Float64,1}, p::Int64)
@assert path[1]=='/' && path[end]=='/' "This function expects a / at the extremities of path"

# Perform a free-run of the model to identify the structure

xfreerun = rand(model.π0)
tfreerun = 20000
Tfreerun = ceil(Int64, tfreerun/model.Δtobs)
data_freerun = generate_lorenz63(model, xfreerun, Tfreerun)

Tdiscard = 2000
Δdiscard = 40

#Store all the metric per number of ensemble members
metric_hist = []


@showprogress for Ne in Ne_array
    metric_Ne = Metrics[]
    for β in β_array

	@show Ne, β

	γ = 2.0
	λ = 0.0
	δ = 1e-8
	κ = 10.0

	J = model.Tstep
	t0 = model.Tspinup*model.Δtobs
	F = model.F
	dist = Float64.(metric_lorenz(model.Nx))
	idx = vcat(collect(1:model.Ny)',collect(1:model.Ny)')

	Slist = SparseRadialMap[]
		for i=1:model.Ny
			idx1, idx2 = idx[:,i]
			perm = sortperm(view(dist,:,idx2))
			Xi = vcat(data_freerun.yt[i:i,Tdiscard:Δdiscard:end], data_freerun.xt[perm, Tdiscard:Δdiscard:end])
			Si = SparseRadialMap(Xi, -1; λ = λ, δ = δ, γ = γ)
			poff = p
			order = [p; 0; 0]
			# maxfeatures = ceil(Int64, (sqrt(Ne)-(findmin(order)[2]+1))/(poff+1))

			optimize(Si, Xi, poff, order, 2; apply_rescaling = true, start = 2)#, maxfeatures = maxfeatures)
			push!(Slist, Si)
		end
    # Load file
    X0 = load(path*"set_up_Ne"*string(Ne)*".jld", "X")

    X = zeros(model.Ny + model.Nx, Ne)
    X[model.Ny+1:model.Ny+model.Nx,:] .= copy(X0)

    smf = AdaptiveSparseRadialSMF(x->x, F.h, β, model.ϵy, Slist,
                        model.Ny, model.Nx,
                        model.Δtdyn, model.Δtobs, Inf,
                        dist, idx, zeros(model.Nx+1, Ne), false, true)
    @time statehist = seqassim(F, data, J, model.ϵx, smf, X, model.Ny, model.Nx, t0);

    metric = post_process(data, model, J, statehist)
    push!(metric_Ne, deepcopy(metric))
    println("Ne "*string(Ne)*"& β "*string(β)*" RMSE: "*string(metric.rmse_mean))
    end
    push!(metric_hist, deepcopy(metric_Ne))
end

return metric_hist
end
