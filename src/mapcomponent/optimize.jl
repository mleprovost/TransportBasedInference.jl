
export optimize


function optimize(Hk::HermiteMapk{m, Nψ, k}, X, maxterms::Union{Nothing, Int64, String}; verbose::Bool = false) where {m, Nψ, k}

    if typeof(maxterms) <: Nothing
        S = Storage(Hk.I.f, X)

        # Optimize coefficients
        coeff0 = getcoeff(Hk)
        precond = zeros(ncoeff(Hk), ncoeff(Hk))
        precond!(precond, coeff0, S, Hk, X)

        res = Optim.optimize(Optim.only_fg!(negative_log_likelihood!(S, Hk, X)), coeff0,
              Optim.LBFGS(; m = 20, P = Preconditioner(precond)))

        setcoeff!(Hk, Optim.minimizer(res))

        error = res.minimum

    elseif typeof(maxterms) <: Int64
        Hk, error =  greedyfit(m, k, X, maxterms; verbose = verbose)

    elseif maxterms ∈ ("kfold", "Kfold", "Kfolds")
        # Define cross-validation splits of data
        n_folds = 5
        folds = kfolds(1:size(X,2), k = n_folds)

        # Run greedy approximation
        max_iter = ceil(Int64, sqrt(size(X,2)))

        valid_error = zeros(max_iter, n_folds)
        @inbounds for i=1:n_folds
            idx_train, idx_valid = folds[i]

            Hk, error = greedyfit(m, k, X[:,idx_train], X[:,idx_valid], max_iter; verbose  = verbose)

            # error[2] contains the histroy of the validation error
            valid_error[:,i] .= deepcopy(error[2])
        end
        # Find optimal numbers of terms
        mean_valid_error = mean(valid_error, dims  = 2)[:,1]

        _, opt_nterms = findmin(mean_valid_error)

        # Run greedy fit up to opt_nterms on all the data
        Hk, error = greedyfit(m, k, X, opt_nterms; verbose  = verbose)

    elseif maxterms ∈ ("split", "Split")
        nvalid = ceil(Int64, floor(0.2*size(X,2)))
        X_train = X[:,nvalid+1:end]
        X_valid = X[:,1:nvalid]

        # Set maximum patience for optimization
        maxpatience = 20

        # Run greedy approximation
        max_iter = ceil(Int64, sqrt(size(X,2)))

        Hk, error = greedyfit(m, k, X_train, X_valid, max_iter;
                                       maxpatience = maxpatience, verbose  = verbose)
    else
        error("Argument max_terms is not recognized")
    end
    return Hk, error
end


function optimize(Lk::LinHermiteMapk{m, Nψ, k}, X::Array{Float64,2}, maxterms::Union{Nothing, Int64, String}; verbose::Bool = false) where {m, Nψ, k}

    transform!(Lk.L, X)
    Hk = Lk.H
    Hk_opt, error = optimize(Hk, X, maxterms; verbose = verbose)

    itransform!(Lk.L, X)

    return LinHermiteMapk(Lk.L, Hk_opt), error
end
