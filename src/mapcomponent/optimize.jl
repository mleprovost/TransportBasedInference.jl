export optimize


function optimize(C::MapComponent, X, maxterms::Union{Nothing, Int64, String};
                  withconstant::Bool = false, withqr::Bool = false,
                  maxpatience::Int64 = 10^5, verbose::Bool = false,
                  hessprecond = true, P::Parallel = serial)

    m = C.m
    Nx = C.Nx

    # algo = Optim.BFGS()
    # algo = Optim.LBFGS(; m = 10)

    if typeof(maxterms) <: Nothing
        S = Storage(C.I.f, X)

        # Optimize coefficients
        if withqr == false
            coeff0 = getcoeff(C)


            if hessprecond == true
                precond = zeros(ncoeff(C), ncoeff(C))
                precond!(precond, coeff0, S, C, X)
                res = Optim.optimize(Optim.only_fg!(negative_log_likelihood(S, C, X)), coeff0,
                                     Optim.LBFGS(; m = 10, P = Preconditioner(precond)))
            else
                res = Optim.optimize(Optim.only_fg!(negative_log_likelihood(S, C, X)), coeff0,
                                     Optim.LBFGS(; m = 10))
            end

            setcoeff!(C, Optim.minimizer(res))
            error = res.minimum
        else
            F = QRscaling(S)
            coeff0 = getcoeff(C)
            mul!(coeff0, F.U, coeff0)

            # mul!(S.ψoffψd0, S.ψoffψd0, F.Uinv)
            # mul!(S.ψoffdψxd, S.ψoffdψxd, F.Uinv)
            if hessprecond == true
                qrprecond = zeros(ncoeff(C), ncoeff(C))
                qrprecond!(qrprecond, coeff0, F, S, C, X)

                res = Optim.optimize(Optim.only_fg!(qrnegative_log_likelihood(F, S, C, X)), coeff0,
                                     Optim.LBFGS(; m = 10, P = Preconditioner(qrprecond)))
            else
                res = Optim.optimize(Optim.only_fg!(qrnegative_log_likelihood(F, S, C, X)), coeff0,
                                     Optim.LBFGS(; m = 10))
            end

            mul!(view(C.I.f.f.coeff,:), F.Uinv, Optim.minimizer(res))

            error = res.minimum

            # Compute initial loss on training set
            # mul!(S.ψoffψd0, S.ψoffψd0, F.Uinv)
            # mul!(S.ψoffdψxd, S.ψoffdψxd, F.Uinv)
        end

    elseif typeof(maxterms) <: Int64
        C, error =  greedyfit(m, Nx, X, maxterms; withconstant = withconstant,
                              withqr = withqr, maxpatience = maxpatience,
                              verbose = verbose, hessprecond = hessprecond)

    elseif maxterms ∈ ("kfold", "kfolds", "Kfold", "Kfolds")
        # Define cross-validation splits of data
        n_folds = 5
        folds = kfolds(1:size(X,2), k = n_folds)

        # Run greedy approximation
        max_iter = min(m-1, ceil(Int64, sqrt(size(X,2))))

        valid_error = zeros(max_iter+1, n_folds)
        if typeof(P) <: Serial
            @inbounds for i=1:n_folds
                idx_train, idx_valid = folds[i]

                C, error = greedyfit(m, Nx, X[:,idx_train], X[:,idx_valid], max_iter;
                                     withconstant = withconstant, withqr = withqr, verbose  = verbose,
                                     hessprecond = hessprecond)

                # error[2] contains the history of the validation error
                valid_error[:,i] .= deepcopy(error[2])
            end
        elseif typeof(P) <: Thread
            @inbounds  Threads.@threads for i=1:n_folds
                idx_train, idx_valid = folds[i]

                C, error = greedyfit(m, Nx, X[:,idx_train], X[:,idx_valid], max_iter;
                                     withconstant = withconstant, withqr = withqr, verbose  = verbose,
                                     hessprecond = hessprecond)

                # error[2] contains the history of the validation error
                valid_error[:,i] .= deepcopy(error[2])
            end
        end

        # Find optimal numbers of terms
        mean_valid_error = mean(valid_error, dims  = 2)[:,1]

        _, opt_nterms = findmin(mean_valid_error)

        # Run greedy fit up to opt_nterms on all the data
        C, error = greedyfit(m, Nx, X, opt_nterms; withqr = withqr, verbose  = verbose, hessprecond = hessprecond)

    elseif maxterms ∈ ("split", "Split")
        nvalid = ceil(Int64, floor(0.2*size(X,2)))
        X_train = X[:,nvalid+1:end]
        X_valid = X[:,1:nvalid]

        # Set maximum patience for optimization
        maxpatience = 20

        # Run greedy approximation
        max_iter =  min(m, ceil(Int64, sqrt(size(X,2))))

        C, error = greedyfit(m, Nx, X_train, X_valid, max_iter;
                             withconstant = withconstant, withqr = withqr,
                             maxpatience = maxpatience, verbose  = verbose,
                             hessprecond = hessprecond)
    else
        println("Argument max_terms is not recognized")
        error()
    end
    return C, error
end


function optimize(L::LinMapComponent, X::Array{Float64,2}, maxterms::Union{Nothing, Int64, String};
                  withconstant::Bool = false, withqr::Bool = false, maxpatience::Int64=20, verbose::Bool = false, hessprecond::Bool = true)

    transform!(L.L, X)
    C = L.C
    C_opt, error = optimize(C, X, maxterms; withconstant = withconstant, withqr = withqr, maxpatience = maxpatience,
                            verbose = verbose, hessprecond = hessprecond)

    itransform!(L.L, X)

    return LinMapComponent(L.L, C_opt), error
end
