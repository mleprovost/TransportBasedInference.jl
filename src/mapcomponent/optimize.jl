
export optimize


function optimize(C::MapComponent, X, maxterms::Union{Nothing, Int64, String}; verbose::Bool = false)

    m = C.m
    Nx = C.Nx

    if typeof(maxterms) <: Nothing
        S = Storage(C.I.f, X)

        # Optimize coefficients
        coeff0 = getcoeff(C)
        precond = zeros(ncoeff(C), ncoeff(C))
        precond!(precond, coeff0, S, C, X)

        res = Optim.optimize(Optim.only_fg!(negative_log_likelihood!(S, C, X)), coeff0,
              Optim.LBFGS(; m = 20, P = Preconditioner(precond)))

        setcoeff!(C, Optim.minimizer(res))

        error = res.minimum

    elseif typeof(maxterms) <: Int64
        C, error =  greedyfit(m, Nx, X, maxterms; verbose = verbose)

    elseif maxterms ∈ ("kfold", "Kfold", "Kfolds")
        # Define cross-validation splits of data
        n_folds = 5
        folds = kfolds(1:size(X,2), k = n_folds)

        # Run greedy approximation
        max_iter = ceil(Int64, sqrt(size(X,2)))

        valid_error = zeros(max_iter, n_folds)
        @inbounds for i=1:n_folds
            idx_train, idx_valid = folds[i]

            C, error = greedyfit(m, Nx, X[:,idx_train], X[:,idx_valid], max_iter; verbose  = verbose)

            # error[2] contains the histroy of the validation error
            valid_error[:,i] .= deepcopy(error[2])
        end
        # Find optimal numbers of terms
        mean_valid_error = mean(valid_error, dims  = 2)[:,1]

        _, opt_nterms = findmin(mean_valid_error)

        # Run greedy fit up to opt_nterms on all the data
        C, error = greedyfit(m, Nx, X, opt_nterms; verbose  = verbose)

    elseif maxterms ∈ ("split", "Split")
        nvalid = ceil(Int64, floor(0.2*size(X,2)))
        X_train = X[:,nvalid+1:end]
        X_valid = X[:,1:nvalid]

        # Set maximum patience for optimization
        maxpatience = 20

        # Run greedy approximation
        max_iter = ceil(Int64, sqrt(size(X,2)))

        C, error = greedyfit(m, Nx, X_train, X_valid, max_iter;
                                       maxpatience = maxpatience, verbose  = verbose)
    else
        error("Argument max_terms is not recognized")
    end
    return C, error
end


function optimize(L::LinMapComponent, X::Array{Float64,2}, maxterms::Union{Nothing, Int64, String}; verbose::Bool = false)

    transform!(L.L, X)
    C = L.C
    C_opt, error = optimize(C, X, maxterms; verbose = verbose)

    itransform!(L.L, X)

    return LinMapComponent(L.L, C_opt), error
end
