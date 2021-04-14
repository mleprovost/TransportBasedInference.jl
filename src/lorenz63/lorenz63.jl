export lorenz63!, setup_lorenz63, generate_lorenz63, benchmark_lorenz63

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

    	sol = solve(prob, Tsit5(), dense = false, save_everystep = false)
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
    	return SyntheticData(tt, x0, xt, yt)
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
	save(path*"set_up_Ne"*string(Ne)*".jld", "ens", statehist[end], "Ne", Ne, "x0", data.x0, "tt", data.tt, "xt", data.xt, "yt", data.yt)
end


function setup_lorenz63(path::String, Ne_array::Array{Int64,1})
    Nx = 3
    Ny = 3
    Δtdyn = 0.05
    Δtobs = 0.1

    σx = 1e-2#1e-6#1e-2
    σy = 2.0#1e-6#2.0

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

    model = Model(Nx, Ny, Δtdyn, Δtobs, ϵx, ϵy, MvNormal(zeros(Nx), mMatrix(1.0*I, Nx, Nx)), Tburn, Tstep, Tspinup, F);

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
Metric_list = []

@showprogress for Ne in Ne_array
    Metric_Ne = Metrics[]
    for β in β_array
    @show Ne, β
    # Load file
    X0 = load(path*"set_up_Ne"*string(Ne)*".jld", "ens")

    X = zeros(model.Ny + model.Nx, Ne)
    X[model.Ny+1:model.Ny+model.Nx,:] .= deepcopy(X0)
    J = model.Tstep
    t0 = model.Tspinup*model.Δtobs
    F = model.F
    enkf = StochEnKF(x->x, model.ϵy, model.Δtdyn, model.Δtobs, false, false)

    # Use this multi-additive inflation
    ϵx = MultiAddInflation(model.Nx, β, zeros(model.Nx), model.ϵx.Σ, model.ϵx.σ)

    # @time enshist = seqassim(dyn, data, J, ϵx, enkf, ens, t0)
	@time statehist = seqassim(F, data, J, model.ϵx, enkf, X, model.Ny, model.Nx, t0);

    Metric = post_process(data, model, J, statehist)
    push!(Metric_Ne, deepcopy(Metric))
    println("Ne "*string(Ne)*"& β "*string(β)*" RMSE: "*string(Metric.rmse_mean))
    end
    push!(Metric_list, deepcopy(Metric_Ne))
end

return Metric_list
end
#
#
# function benchmark_TMap_lorenz63(model::Model, data::SyntheticData, path::String, Ne_array::Array{Int64,1}, β_array::Array{Float64,1})
# @assert path[1]=='/' && path[end]=='/' "This function expects a / at the extremities of path"
#
# #Store all the metric per number of ensemble members
# Metric_hist = []
#
# @showprogress for Ne in Ne_array
#     Metric_Ne = Metrics[]
#     for β in β_array
#     @show Ne, β
#     # Load file
#     ens0 = load(path*"set_up_Ne"*string(Ne)*".jld", "ens")
#
#     ens = EnsembleStateMeas(3,3,Ne)
#     ens.state.S .= ens0
#
#     J = model.Tstep
#     t0 = model.Tspinup*model.Δtobs
#     dyn = DynamicalSystem(model.f, model.h)
#     dist = metric_lorenz(3)
#     p = 0
#     # order = [[-1], [1; 1], [-1; 1; 0], [-1; 1; 1; 0]]
#     order = [[-1], [p; p], [-1; p; 0], [-1; p; p; 0]]
#     T = SparseTMap(3, 3, Ne, order, 2.0, 0.1, 1e-8, 10.0,  dist, dyn, x->x, β, model.ϵy, model.Δtdyn, model.Δtobs, false);
#
#     @time enshist = seqassim(dyn, data, J, model.ϵx, T, ens, t0)
#
#     Metric = post_process(data, model, J, enshist)
#     push!(Metric_Ne, deepcopy(Metric))
#     println("Ne "*string(Ne)*"& β "*string(β)*" RMSE: "*string(Metric.rmse_mean))
#     end
#     push!(Metric_hist, deepcopy(Metric_Ne))
# end
#
# return Metric_hist
# end
