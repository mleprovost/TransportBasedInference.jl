export greedyfit, update_component, update_coeffs

"""
$(TYPEDSIGNATURES)

Performs a greedy feature selection of an `HermiteMapComponent` up to `maxterms` features, based on the ensemble matrix `X`.

The following optionial settings can be tuned:
* `α::Float64 = αreg`: a Tikkonov regularization parameter of the cost function
* `withconstant::Bool = false`: the option to remove the constant feature in the greedy optimization routine, if the zero feature is the constant function for the basis of interest.
* `withqr::Bool = false`: improve the conditioning of the optimization problem with a QR factorization of the feature basis (recommended option)
* `maxpatience::Int64 = 10^5`: for `optimkind = split`, the maximum number of extra terms that can be added without decreasing the validation error before the greedy optimmization get stopped.
* `verbose::Bool = false`: prints details of the optimization procedure, current component optimize, training and validation errors, number of features added.
* `hessprecond::Bool=true`: use a preconditioner based on the Gauss-Newton of the Hessian of the loss function to accelerate the convergence.
* `b::String="CstProHermiteBasis"`: several bases for the feature expansion are available, see src/hermitemap/basis.jl for more details
* `ATMcriterion::String="gradient"`: sensitivty criterion used to select the feature to add to the expansion. The default uses the derivative of the cost function with respect to the coefficient
   of the features in the reduced margin of the current set of features.
"""
function greedyfit(m::Int64, Nx::Int64, X, maxterms::Int64; α::Float64 = αreg, withconstant::Bool = false, withqr::Bool = false,
                   maxpatience::Int64 = 10^5, verbose::Bool = true, hessprecond::Bool=true,
                   b::String="CstProHermiteBasis", ATMcriterion::String="gradient")

    @assert maxterms >=1 "maxterms should be >= 1"
    best_valid_error = Inf
    patience = 0

    train_error = Float64[]

    # Initialize map C to identity
    C = HermiteMapComponent(m, Nx; α = α, b = b);

    # Compute storage # Add later storage for validation S_valid
    S = Storage(C.I.f, X)

    # Remove or not the constant i.e the multi-index [0 0 0]
    if withconstant == false && iszerofeatureactive(C.I.f.B.B) == false
        # Compute the reduced margin
        reduced_margin = getreducedmargin(getidx(C))
        f = ExpandedFunction(C.I.f.B, reduced_margin, zeros(size(reduced_margin,1)))
        C = HermiteMapComponent(f; α = αreg)
        S = Storage(C.I.f, X)
        coeff0 = getcoeff(C)
        dJ = zero(coeff0)

        negative_log_likelihood!(nothing, dJ, coeff0, S, C, X)
        _, opt_dJ_coeff_idx = findmax(abs.(dJ))

        opt_idx = reduced_margin[opt_dJ_coeff_idx:opt_dJ_coeff_idx,:]

        f = ExpandedFunction(C.I.f.B, opt_idx, zeros(size(opt_idx,1)))
        C = HermiteMapComponent(f; α = α)
        S = Storage(C.I.f, X)
    end

    if withqr == true
        F = QRscaling(S)
    end

    # Compute initial loss on training set
    push!(train_error, negative_log_likelihood!(0.0, nothing, getcoeff(C), S, C, X))

    if verbose == true
        println(string(ncoeff(C))*" terms - Training error: "*string(train_error[end]))
    end

    # Compute the reduced margin
    reduced_margin = getreducedmargin(getidx(C))

    while ncoeff(C) < maxterms
        coeff_old = getcoeff(C)
        idx_old = getidx(C)

        idx_new, reduced_margin = update_component!(C, X, reduced_margin, S; ATMcriterion = ATMcriterion)

        # Update storage with the new feature
        S = update_storage(S, X, idx_new[end:end,:])

        # Update C
        C = HermiteMapComponent(IntegratedFunction(S.f); α = C.α)
        # Optimize coefficients
        if withqr == false
            coeff0 = getcoeff(C)
            if hessprecond == true
                precond = zeros(ncoeff(C), ncoeff(C))
                precond!(precond, coeff0, S, C, X)
                precond_chol = cholesky(Symmetric(precond); check = false)

                if issuccess(precond_chol) == true
                    res = Optim.optimize(Optim.only_fg!(negative_log_likelihood(S, C, X)), coeff0,
                          Optim.LBFGS(; m = 10, P = Preconditioner(Symmetric(precond), precond_chol)))
                elseif cond(Diagonal(precond)) < 10^6
                    res = Optim.optimize(Optim.only_fg!(negative_log_likelihood(S, C, X)), coeff0,
                          Optim.LBFGS(; m = 10, P = Diagonal(precond)))
                else
                    res = Optim.optimize(Optim.only_fg!(negative_log_likelihood(S, C, X)), coeff0,
                          Optim.LBFGS(; m = 10))
                end
            else

            res = Optim.optimize(Optim.only_fg!(negative_log_likelihood(S, C, X)), coeff0,
                                 Optim.LBFGS(; m = 10))
            end

            if Optim.converged(res) == true
                setcoeff!(C, Optim.minimizer(res))
                # Compute new loss on training set
                push!(train_error, negative_log_likelihood!(0.0, nothing, getcoeff(C), S, C, X))
            else
                # if optimization wasn't successful, return map
                setcoeff!(C, Optim.minimizer(res))
                append!(train_error, Inf*ones(maxterms - size(getcoeff(C), 1) + 1))
                println("Optimization wasn't successful")
                break
            end
        else

            coeff0 = getcoeff(C)
            # F = QRscaling(S)
            F = updateQRscaling(F, S)

            # Check condition number of basis evaluations: cond(ψoff ⊗ ψd) = cond(Q R) = cond(R) as Q is an orthogonal matrix
            if cond(F.R) > 1e9 || size(F.R,1) < size(F.R,2)
                println("Warning: Condition number reached")
                append!(train_error, Inf*ones(maxterms - size(getcoeff(C), 1) + 1))
                # Revert to the previous map component
                C = HermiteMapComponent(ExpandedFunction(C.I.f.B, idx_old, coeff_old); α = C.α)
                break
            end
            mul!(coeff0, F.U, coeff0)

            if hessprecond == true
                qrprecond = zeros(ncoeff(C), ncoeff(C))
                qrprecond!(qrprecond, coeff0, F, S, C, X)
                qrprecond_chol = cholesky(Symmetric(qrprecond); check = false)

                if issuccess(qrprecond_chol) == true
                    res = Optim.optimize(Optim.only_fg!(qrnegative_log_likelihood(F, S, C, X)), coeff0,
                                         Optim.LBFGS(; m = 10, P = Preconditioner(Symmetric(qrprecond), qrprecond_chol)))
                elseif cond(Diagonal(qrprecond)) < 10^6
                    res = Optim.optimize(Optim.only_fg!(qrnegative_log_likelihood(F, S, C, X)), coeff0,
                                         Optim.LBFGS(; m = 10, P = Diagonal(qrprecond)))
                else
                    res = Optim.optimize(Optim.only_fg!(qrnegative_log_likelihood(F, S, C, X)), coeff0,
                                         Optim.LBFGS(; m = 10))
                end
                # mul!(S.ψoffψd0, S.ψoffψd0, F.Uinv)
                # mul!(S.ψoffdψxd, S.ψoffdψxd, F.Uinv)
            else
                res = Optim.optimize(Optim.only_fg!(qrnegative_log_likelihood(F, S, C, X)), coeff0,
                                     Optim.LBFGS(; m = 10))
            end

            # Reverse to the non-QR space and update in-place the coefficients
            C.I.f.coeff .= F.Uinv*Optim.minimizer(res)

            if Optim.converged(res) == true
                # The computation is the non-QR space is slightly faster,
                # even if we prefer the QR form to optimize the coefficients
                push!(train_error, negative_log_likelihood!(0.0, nothing, getcoeff(C), S, C, X))
            else
                println("Optimization wasn't successful")
                append!(train_error, Inf*ones(maxterms - size(getcoeff(C), 1) + 1))
                break
            end
        end

        if verbose == true
            println(string(ncoeff(C))*" terms - Training error: "*string(train_error[end]))
        end
    end

    return C, train_error
end

"""
$(TYPEDSIGNATURES)

Compute the sensitivity of the loss function with respect to the multi-indices in the reduced-margin,
determines the feature to add to the exsiting set based on the `ATMcriterion`,
and finally returns the new set of multi-indices and the updated redcued margin.
"""
function update_component!(C::HermiteMapComponent, X, reduced_margin::Array{Int64,2}, S::Storage; ATMcriterion::String="gradient")
    m = C.m
    Nψ = C.Nψ
    idx_old = getidx(C)

    idx_new = vcat(idx_old, reduced_margin)

    # Define updated map
    f_new = ExpandedFunction(C.I.f.B, idx_new, vcat(getcoeff(C), zeros(size(reduced_margin,1))))
    C_new = HermiteMapComponent(f_new; α = αreg)

    # Set coefficients based on previous optimal solution
    coeff_new, coeff_idx_added, idx_added = update_coeffs(C, C_new)

    # Compute gradient after adding the new elements
    S = update_storage(S, X, reduced_margin)
    dJ = zero(coeff_new)
    negative_log_likelihood!(nothing, dJ, coeff_new, S, C_new, X)
    # Find function in the reduced margin most correlated with the residual

    if ATMcriterion == "gradient"
        _, opt_dJ_coeff_idx = findmax(abs.(dJ[coeff_idx_added]))
    elseif ATMcriterion == "normalized gradient"
        # The definition of ψnorm has a rescaling by 1/√Ne,
        # but it doesn't matter as all the entries are multiplied by the same constant
        _, opt_dJ_coeff_idx = findmax(abs.(dJ[coeff_idx_added]).^2 ./S.ψnorm[coeff_idx_added].^2)
    else
        error(ATMcriterion*" is not implemented.")
    end

    opt_idx = idx_added[opt_dJ_coeff_idx,:]

    # Update multi-indices and the reduced margins based on opt_idx
    # reducedmargin_opt_coeff_idx = Bool[opt_idx == x for x in eachslice(reduced_margin; dims = 1)]

    idx_new, reduced_margin = updatereducedmargin(idx_old, reduced_margin, opt_dJ_coeff_idx)

    return idx_new, reduced_margin
end

"""
$(TYPEDSIGNATURES)

Returns the value and index of the new coefficients, as well as their associated multi-indices.
"""
function update_coeffs(Cold::HermiteMapComponent, Cnew::HermiteMapComponent)

    Nψ = Cold.Nψ
    idx_old = getidx(Cold)
    idx_new = getidx(Cnew)

    coeff_old = getcoeff(Cold)

    # Declare vectors for new coefficients and to track added terms
    Nψnew = Cnew.Nψ
    coeff_new = zeros(Nψnew)
    coeff_added = ones(Nψnew)

    # Update coefficients
    @inbounds for i = 1:Nψ
        idx_i = Bool[idx_old[i,:] == x for x in eachslice(idx_new; dims = 1)]
        coeff_new[idx_i] .= coeff_old[i]
        coeff_added[idx_i] .= 0.0
    end

    # Find indices of added coefficients and corresponding multi-indices
    coeff_idx_added = findall(coeff_added .> 0.0)
    idx_added = idx_new[coeff_idx_added,:]

    return coeff_new, coeff_idx_added, idx_added
end

"""
$(TYPEDSIGNATURES)

Performs a greedy feature selection of an `HermiteMapComponent` up to `maxterms` features, based on the pair of ensemble matrices `X` (training set) and `Xvalid` (validation set).

The following optionial settings can be tuned:
* `α::Float64 = αreg`: a Tikkonov regularization parameter of the cost function
* `withconstant::Bool = false`: the option to remove the constant feature in the greedy optimization routine, if the zero feature is the constant function for the basis of interest.
* `withqr::Bool = false`: improve the conditioning of the optimization problem with a QR factorization of the feature basis (recommended option)
* `maxpatience::Int64 = 10^5`: for `optimkind = split`, the maximum number of extra terms that can be added without decreasing the validation error before the greedy optimmization get stopped.
* `verbose::Bool = false`: prints details of the optimization procedure, current component optimize, training and validation errors, number of features added.
* `hessprecond::Bool=true`: use a preconditioner based on the Gauss-Newton of the Hessian of the loss function to accelerate the convergence.
* `b::String="CstProHermiteBasis"`: several bases for the feature expansion are available, see src/hermitemap/basis.jl for more details
* `ATMcriterion::String="gradient"`: sensitivty criterion used to select the feature to add to the expansion. The default uses the derivative of the cost function with respect to the coefficient
   of the features in the reduced margin of the current set of features.
"""
function greedyfit(m::Int64, Nx::Int64, X, Xvalid, maxterms::Int64; α::Float64 = αreg, withconstant::Bool = false,
                   withqr::Bool = false, maxpatience::Int64 = 10^5, verbose::Bool = true, hessprecond::Bool=true,
                   b::String="CstProHermiteBasis", ATMcriterion::String="gradient")

    best_valid_error = Inf
    patience = 0

    train_error = Float64[]
    valid_error = Float64[]

    # Initialize map C to identity
    C = HermiteMapComponent(m, Nx; α = α, b = b);

    # Compute storage # Add later storage for validation S_valid
    S = Storage(C.I.f, X)
    Svalid = Storage(C.I.f, Xvalid)

    # Remove or not the constant i.e the multi-index [0 0 0]
    if withconstant == false && iszerofeatureactive(C.I.f.B.B) == false

        # Compute the reduced margin
        reduced_margin = getreducedmargin(getidx(C))
        f = ExpandedFunction(C.I.f.B, reduced_margin, zeros(size(reduced_margin,1)))
        C = HermiteMapComponent(f; α = α)
        S = Storage(C.I.f, X)
        coeff0 = getcoeff(C)
        dJ = zero(coeff0)

        negative_log_likelihood!(nothing, dJ, coeff0, S, C, X)
        _, opt_dJ_coeff_idx = findmax(abs.(dJ))

        opt_idx = reduced_margin[opt_dJ_coeff_idx:opt_dJ_coeff_idx,:]

        f = ExpandedFunction(C.I.f.B, opt_idx, zeros(size(opt_idx,1)))
        C = HermiteMapComponent(f; α = αreg)
        S = Storage(C.I.f, X)
        Svalid = Storage(C.I.f, Xvalid)
    end

    if withqr == true
        F = QRscaling(S)
    end

    # Compute initial loss on training set
    push!(train_error, negative_log_likelihood!(0.0, nothing, getcoeff(C), S, C, X))
    push!(valid_error, negative_log_likelihood!(0.0, nothing, getcoeff(C), Svalid, C, Xvalid))

    if verbose == true
        println(string(ncoeff(C))*" terms - Training error: "*
        string(train_error[end])*", Validation error: "*string(valid_error[end]))
    end

    # Compute the reduced margin
    reduced_margin = getreducedmargin(getidx(C))

    while ncoeff(C) < maxterms
        coeff_old = getcoeff(C)
        idx_old = getidx(C)
        idx_new, reduced_margin = update_component!(C, X, reduced_margin, S; ATMcriterion = ATMcriterion)

        # Update storage with the new feature
        S = update_storage(S, X, idx_new[end:end,:])
        Svalid = update_storage(Svalid, Xvalid, idx_new[end:end,:])

        # Update C
        C = HermiteMapComponent(IntegratedFunction(S.f); α = C.α)

        # Optimize coefficients
        if withqr == false
            coeff0 = getcoeff(C)
            if hessprecond == true
                precond = zeros(ncoeff(C), ncoeff(C))
                precond!(precond, coeff0, S, C, X)
                precond_chol = cholesky(Symmetric(precond); check = false)

                if issuccess(precond_chol) == true
                    res = Optim.optimize(Optim.only_fg!(negative_log_likelihood(S, C, X)), coeff0,
                          Optim.LBFGS(; m = 10, P = Preconditioner(Symmetric(precond), precond_chol)))
                elseif cond(Diagonal(precond)) < 10^6
                    res = Optim.optimize(Optim.only_fg!(negative_log_likelihood(S, C, X)), coeff0,
                          Optim.LBFGS(; m = 10, P = Diagonal(precond)))
                else #no preconditioner
                    res = Optim.optimize(Optim.only_fg!(negative_log_likelihood(S, C, X)), coeff0,
                          Optim.LBFGS(; m = 10))
                end
            else
                res = Optim.optimize(Optim.only_fg!(negative_log_likelihood(S, C, X)), coeff0,
                      Optim.LBFGS(; m = 10))
            end

            setcoeff!(C, Optim.minimizer(res))

            if Optim.converged(res) == true
                # Compute new loss on training and validation sets
                push!(train_error, negative_log_likelihood!(0.0, nothing, getcoeff(C), S, C, X))
                push!(valid_error, negative_log_likelihood!(0.0, nothing, getcoeff(C), Svalid, C, Xvalid))
            else
                println("Optimization wasn't successful")
                # Compute new loss on training and validation sets
                append!(train_error, Inf*ones(maxterms - size(getcoeff(C), 1) + 1))
                append!(valid_error, Inf*ones(maxterms - size(getcoeff(C), 1) + 1))
                break
            end

        else
            coeff0 = getcoeff(C)
            F = updateQRscaling(F, S)
            # F = QRscaling(S)

            # Check condition number of basis evaluations: cond(ψoff ⊗ ψd) = cond(Q R) = cond(R) as Q is an orthogonal matrix
            if cond(F.R) > 1e9 || size(F.R,1) < size(F.R,2)
                println("Warning: Condition number reached")
                append!(train_error, Inf*ones(maxterms - size(getcoeff(C), 1) + 1))
                append!(valid_error, Inf*ones(maxterms - size(getcoeff(C), 1) + 1))
                # Revert to the previous map component
                C = HermiteMapComponent(ExpandedFunction(C.I.f.B, idx_old, coeff_old); α = C.α)
                break
            end

            mul!(coeff0, F.U, coeff0)
            if hessprecond == true
                qrprecond = zeros(ncoeff(C), ncoeff(C))
                qrprecond!(qrprecond, coeff0, F, S, C, X)
                qrprecond_chol = cholesky(Symmetric(qrprecond); check = false)

                if issuccess(qrprecond_chol) == true
                    res = Optim.optimize(Optim.only_fg!(qrnegative_log_likelihood(F, S, C, X)), coeff0,
                                         Optim.LBFGS(; m = 10, P = Preconditioner(Symmetric(qrprecond), qrprecond_chol)))
                elseif cond(Diagonal(qrprecond)) < 10^6
                    res = Optim.optimize(Optim.only_fg!(qrnegative_log_likelihood(F, S, C, X)), coeff0,
                                         Optim.LBFGS(; m = 10, P = Diagonal(qrprecond)))
                else
                    res = Optim.optimize(Optim.only_fg!(qrnegative_log_likelihood(F, S, C, X)), coeff0,
                                         Optim.LBFGS(; m = 10))
                end
            else
                res = Optim.optimize(Optim.only_fg!(qrnegative_log_likelihood(F, S, C, X)), coeff0,
                                     Optim.LBFGS(; m = 10))
            end
            # Reverse to the non-QR space and update in-place the coefficients
            mul!(view(C.I.f.coeff,:), F.Uinv, Optim.minimizer(res))

            if Optim.converged(res) == true
                # The computation is the non-QR space is slightly faster,
                # even if we prefer the QR form to optimize the coefficients
                push!(train_error, negative_log_likelihood!(0.0, nothing, getcoeff(C), S, C, X))
                push!(valid_error, negative_log_likelihood!(0.0, nothing, getcoeff(C), Svalid, C, Xvalid))
            else
                println("Optimization wasn't successful")
                # Compute new loss on training and validation sets
                append!(train_error, Inf*ones(maxterms - size(getcoeff(C), 1) + 1))
                append!(valid_error, Inf*ones(maxterms - size(getcoeff(C), 1) + 1))
                break
            end

        end

        if verbose == true
            println(string(ncoeff(C))*" terms - Training error: "*
            string(train_error[end])*", Validation error: "*string(valid_error[end]))
        end

        # Update patience
        if valid_error[end] >= best_valid_error
            patience += 1
        else
            best_valid_error = deepcopy(valid_error[end])
            patience = 0
        end

        # Check if patience exceeded maximum patience
        if patience >= maxpatience
            println("Patience has been exceeded")
            break
        end
    end

    return C, (train_error, valid_error)
end
