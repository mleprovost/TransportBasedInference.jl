export assimilate_obs


function assimilate_obs(M::HermiteMap, X, Ystar, Ny, Nx; withconstant::Bool = false,
                        withqr::Bool = false, verbose::Bool = false, P::Parallel = serial)

        Nystar, Neystar = size(Ystar)
        Nypx, Ne = size(X)

        @assert Nystar == Ny "Size of ystar is not consistent with Ny"
        @assert Nypx == Ny + Nx "Size of X is not consistent with Ny or Nx"
        @assert Ne == Neystar "Size of X and Ystar are not consistent"

        # Optimize the transport map
        M = optimize(M, X, 10; withconstant = withconstant, withqr = withqr,
                               verbose = verbose, start = Ny+1, P = P)

        # Evaluate the transport map
        F = evaluate(M, X; start = Ny+1, P = P)

        inverse!(F, M, X, Ystar; start = Ny+1, P = P)

        return X
end
