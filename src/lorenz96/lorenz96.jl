export lorenz96!, generate_lorenz96, spin_lorenz96, setup_lorenz96, benchmark_lorenz96, setup_order, benchmark_srmf_lorenz96, benchmark_sadaptivermf_lorenz96


"""
    lorenz96!(du,u,p,t)

Compute in-place the right-hand-side of the Lorenz-96 system for a state `u` at time `t`,
and store it in `du`. `p` is vector of user-defined parameters.
"""
function lorenz96!(du,u,p,t)
    F = 8.0
    n = size(u,1)
    du[1] = (u[2]-u[end-1])*u[end] - u[1] + F
    du[2] = (u[3]-u[end])*u[1] - u[2] + F
    du[end] = (u[1] - u[end-2])*u[end-1] - u[end] + F

    @inbounds for i=3:n-1
        du[i] = (u[i+1] - u[i-2])*u[i-1] - u[i] + F
    end
    return du
end


function generate_lorenz96(model::Model, x0::Array{Float64,1}, J::Int64)

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

    	sol = solve(prob, Tsit5(), dt = model.Δtdyn, adaptive = false,
		            dense = false, save_everystep = false)
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
		yt[:,i] .+= model.ϵy.m + model.ϵy.σ*randn(model.Ny)
    end
    	return SyntheticData(tt, x0, xt, yt)
end

function spin_lorenz96(model::Model, data::SyntheticData, Ne::Int64, path::String)
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

function setup_lorenz96(path::String, Ne_array::Array{Int64,1})
    Nx = 40
    Ny = 20

    # The state is measured every fourth
    Δ = 2
    yidx = 1:Δ:Nx

    Δtdyn = 0.01
    Δtobs = 0.4

    σx = 1e-6#1e-2#1e-6#1e-2
    σy = sqrt(0.5)#1e-6#2.0

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

    f = lorenz96!
    h(x, t) = x[yidx]
    # Create a local version of the observation operator
    h(x, idx, t) = x[idx]
	F = StateSpace(f, h)

    model = Model(Nx, Ny, Δtdyn, Δtobs, ϵx, ϵy, MvNormal(zeros(Nx), Matrix(1.0*I, Nx, Nx)), Tburn, Tstep, Tspinup, F);

    # Set initial condition
    x0 = rand(model.π0)
    # x0 = [0.645811242103507;  -1.465126216973632;   0.385227725149822];

    # Run dynamics and generate data
    data = generate_lorenz96(model, x0, model.Tspinup+model.Tstep);


    for Ne in Ne_array
        spin_lorenz96(model, data, Ne, path)
    end

    return model, data
end

function benchmark_lorenz96(model::Model, data::SyntheticData, path::String, Ne_array::Array{Int64,1},
	                        β_array::Array{Float64,1}, Lrad_array::Array{Float64,1})
@assert path[1]=='/' && path[end]=='/' "This function expects a / at the extremities of path"

#Store all the metric per number of ensemble members
metric_list = []

@showprogress for Ne in Ne_array
    metric_Ne = Metrics[]
    for β in β_array
		for Lrad in Lrad_array
		    @show Ne, β, Lrad
		    # Load file
		    X0 = load(path*"set_up_Ne"*string(Ne)*".jld", "X")

		    X = zeros(model.Ny + model.Nx, Ne)
		    X[model.Ny+1:model.Ny+model.Nx,:] .= copy(X0)
		    J = model.Tstep
		    t0 = model.Tspinup*model.Δtobs
		    F = model.F

			Δ = 2
			yidx = 1:Δ:model.Nx
			idx = vcat(collect(1:length(yidx))', collect(yidx)')

			enkf = SeqStochEnKF(x->x, F.h, MultiplicativeInflation(β), model.ϵy, model.Ny, model.Nx,
			                    model.Δtdyn, model.Δtobs,
			                    idx, zeros(model.Nx+1, Ne), false, true)
		    # Use this multi-additive inflation
		    ϵx = MultiAddInflation(model.Nx, β, zeros(model.Nx), model.ϵx.Σ, model.ϵx.σ)

			# Create Localization structure
		    Gxy(i,j) = periodicmetric!(i,yidx[j], model.Nx)
		    Gyy(i,j) = periodicmetric!(yidx[i],yidx[j], model.Nx)
		    Loc = Localization(Lrad, Gxy, Gyy)

			@time statehist = seqassim(F, data, J, model.ϵx, enkf, X, model.Ny, model.Nx, t0, Loc);

		    metric = post_process(data, model, J, statehist)
		    push!(metric_Ne, deepcopy(metric))
			println("Ne "*string(Ne)*"& β "*string(β)*"& Lrad "*string(Lrad)*" RMSE: "*string(metric.rmse_mean))
		end
	end
	push!(metric_list, deepcopy(metric_Ne))
end

return metric_list
end


# Nx size of the state
# off_p contains the order for the off-diagonal component
# off_rad contains the radius for the localization of the map
# nonid_rad contains the number of non id components
# dist matrix of the distances between the variables
function setup_order(Nx, diagobs_p, diagunobs_p, off_p, off_rad, nonid_rad, dist::Array{Float64,2})

	dist = convert(Array{Int64,2}, dist)
    perm = sortperm(view(dist,:,1))

    dist_to_order = fill(-1, Nx)
    # fill!(view(dist_to_order,1:ceil(Int64,off_rad)), off_p)
    fill!(view(dist_to_order,1:ceil(Int64,min(off_rad, Nx))), off_p)
    order = [fill(-1,i) for i=1:Nx+1]
    nonid = Int64[]
    for i=1:Nx
        node_i = perm[i]
        dist_i1 = dist[1, node_i]

        # Vector to store order, offset of 1
        # because we have one scalar observation
        order_i = zeros(Int64, i-1)#view(order,i+1)
        if dist_i1 <= nonid_rad
            if i>1
            for j=1:i-1

                # extract distance to node i
                node_j = perm[j]
                dist_ij = dist[node_i, node_j]

                # compute order for variable based on distance
                order_i[j] = dist_to_order[dist_ij]
            end
            end
            # Define order for diagonal TM components and observation component
            if i==1 # observed mode
                order[2][2] = diagobs_p
                order[2][1] = off_p
            else # unobserved mode
                order[i+1][2:i] .= deepcopy(order_i)
                order[i+1][i+1] = diagunobs_p
                order[i+1][1] = -1
            end
        end
    end
    return order
end

# Routine to benchmark the stochastic radial map filter
function benchmark_srmf_lorenz96(model::Model, data::SyntheticData, path::String, Ne_array::Array{Int64,1}, β_array::Array{Float64,1},
	                             Lrad_array::Array{Float64,1}, nonid_array::Array{Float64,1}, p::Int64)
@assert path[1]=='/' && path[end]=='/' "This function expects a / at the extremities of path"

#Store all the metric per number of ensemble members
metric_hist = []

@showprogress for Ne in Ne_array
    metric_Ne = Metrics[]
    for β in β_array
		for Lrad in Lrad_array
			for nonid_rad in nonid_array
					@show Ne, β, Lrad, nonid_rad
				    # Load file
				    X0 = load(path*"set_up_Ne"*string(Ne)*".jld", "X")

				    X = zeros(model.Ny + model.Nx, Ne)
				    X[model.Ny+1:model.Ny+model.Nx,:] .= copy(X0)

				    J = model.Tstep
				    t0 = model.Tspinup*model.Δtobs
				    F = model.F

					# parameters of the radial map
					γ = 2.0
					λ = 0.01
					δ = 1e-8
					κ = 4.0


					dist = Float64.(metric_lorenz(model.Nx))
					idx = vcat(collect(1:model.Ny)',collect(1:2:model.Nx)')
					order = setup_order(model.Nx, p, 0, p, Lrad, nonid_rad, dist)

					# Define index of the measurement

					smf = SparseRadialSMF(x->x, F.h, β, model.ϵy, order, γ, λ, δ, κ,
				                      model.Ny, model.Nx, Ne,
				                      model.Δtdyn, model.Δtobs,
				                      dist, idx; islocalized = true)

				    @time statehist = seqassim(F, data, J, model.ϵx, smf, X, model.Ny, model.Nx, t0);

				    metric = post_process(data, model, J, statehist)
				    push!(metric_Ne, deepcopy(metric))
				    println("Ne "*string(Ne)*" & β "*string(β)*" & Lrad "*string(Lrad)*" & nonid_rad "*string(nonid_rad)*" & RMSE: "*string(metric.rmse_mean))
				end
			end
	    end
	    push!(metric_hist, deepcopy(metric_Ne))
	end
	return metric_hist
end


# Routine to benchmark the stochastic adaptive radial map filter
function benchmark_sadaptivermf_lorenz96(model::Model, data::SyntheticData, path::String, Ne_array::Array{Int64,1},
	                                     β_array::Array{Float64,1}, nonid_array::Array{Float64,1}, p::Int64)
@assert path[1]=='/' && path[end]=='/' "This function expects a / at the extremities of path"

# Perform a free-run of the model to identify the structure

xfreerun = rand(model.π0)
tfreerun = 10000
Tfreerun = ceil(Int64, tfreerun/model.Δtobs)
data_freerun = generate_lorenz96(model, xfreerun, Tfreerun)

Tdiscard = 1000
Δdiscard = 10

#Store all the metric per number of ensemble members
metric_hist = []


@showprogress for Ne in Ne_array
    metric_Ne = Metrics[]
    for β in β_array
		for nonid_rad in nonid_array

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
		    println("Ne "*string(Ne)*"& β "*string(β)*" & nonid_rad "*string(nonid_rad)*" RMSE: "*string(metric.rmse_mean))
		    end
		end
		push!(metric_hist, deepcopy(metric_Ne))
	end

return metric_hist
end
