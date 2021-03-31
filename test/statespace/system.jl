
@testset "Test Propagation of the Lorenz equation I" begin
    u0 = [10.0; -5.0; 2.0]
    t0 = 0.0
    tf = 5.0
    Δt = 1e-2

    J = ceil(Int64, (tf-t0)/Δt)


    h(u, t) = [sum(u)]
    F = StateSpace(lorenz63!, h)

    prob = ODEProblem(F.f,u0,(t0,tf))
    sol = solve(prob, RK4(), dt = Δt, adaptive = false);


#     Run it for the different ensemble members
    Nx = 3
    Ny = 1
    Ne = 5
    X = zeros(Ny+Nx, Ne)

    viewstate(X, Ny, Nx) .= repeat(u0, 1, Ne)

    statehist = Array{Float64,2}[]
    push!(statehist, deepcopy(X[Ny+1:Ny+Nx,:]))

    for i = 1:J
        # Forecast
        tspan = (t0+(i-1)*Δt, t0+i*Δt)
        prob = remake(prob; tspan=tspan)

        prob_func(prob,i,repeat) = ODEProblem(prob.f,view(X,Ny+1:Ny+Nx,i),prob.tspan)

        ensemble_prob = EnsembleProblem(prob,output_func = (sol,i) -> (sol[end], false),
        prob_func=prob_func)
        sim = solve(ensemble_prob, RK4(), dt = Δt, adaptive = false, EnsembleThreads(),trajectories = Ne,
        dense = false, save_everystep=false);

        @inbounds for i=1:Ne
            view(X, Ny+1:Ny+Nx,i) .= sim[i]
        end

        push!(statehist, deepcopy(viewstate(X, Ny, Nx)))
    end

    observe(F.h, X, 5.0, Ny, Nx)

    for i=1:Ne
        @test norm(view(statehist[end], :,i) - sol(5.0))<1e-11
        @test norm(view(X, 1:Ny,i) - h(5.0, sol(5.0)))<1e-11
        @test norm(view(X, Ny+1:Ny+Nx,i) - sol(5.0))<1e-11

    end
end
