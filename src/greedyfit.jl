export greedyfit, update_component, update_coeffs



function greedyfit(Hk::HermiteMapk{m, Nψ, k}, X::Array{Float64,2}, maxterms::Int64; maxpatience::Int64 = 10^5, verbose::Bool = true) where {m, Nψ, k}

    best_valid_error = Inf
    patience = 0

    Xvalid =  zero(X)

    train_error = Float64[]
    valid_error = Float64[]

    # Initialize map Hk to identity
    Hk = HermiteMapk(m, k; α = 1e-6);

    # Compute storage # Add later storage for validation S_valid
    S = Storage(Hk.I.f, X)
    Svalid = Storage(Hk.I.f, Xvalid)

    # Compute initial loss on training set
    push!(train_error, negative_log_likelihood!(S, Hk, X)(0.0, nothing, getcoeff(Hk)))
    push!(valid_error, negative_log_likelihood!(Svalid, Hk, Xvalid)(0.0, nothing, getcoeff(Hk)))

    # Compute the reduced margin
    reduced_margin = getreducedmargin(getidx(Hk))

    while ncoeff(Hk) < maxterms
        idx_new, reduced_margin = update_component(Hk, X, reduced_margin, S)

        # Update storage with the new feature
        S = update_storage(S, X, idx_new[end:end,:])
        Svalid = update_storage(Svalid, Xvalid, idx_new[end:end,:])


        # Update Hk
        Hk = HermiteMapk(IntegratedFunction(S.f); α = Hk.α)

        # Optimize coefficients
        coeff0 = getcoeff(Hk)
        res = Optim.optimize(Optim.only_fg!(negative_log_likelihood!(S, Hk, X)), coeff0, Optim.BFGS())
        setcoeff!(Hk, Optim.minimizer(res))

        # Compute new loss on the training and validation set
        push!(train_error, negative_log_likelihood!(S, Hk, X)(0.0, nothing, getcoeff(Hk)))
        push!(valid_error, negative_log_likelihood!(Svalid, Hk, Xvalid)(0.0, nothing, getcoeff(Hk)))

        if verbose == true
            println(string(ncoeff(Hk)-1)*" terms - Training error: "*
            string(train_error[end])*", Validation error: "*string(valid_error[end]))
        end

        # Update patience
        if valid_error[end] >= best_valid_error
            patience +=1
        else
            best_valid_error = deepcopy(valid_error[end])
            patience = 0
        end

        # Check if patience exceeded maximum patience
        if patience >= maxpatience
            break
        end
    end

    return Hk, train_error, valid_error
end


function update_component(Hk::HermiteMapk{m, Nψ, k}, X::Array{Float64,2}, reduced_margin::Array{Int64,2}, S::Storage{m, Nψ, k}) where {m, Nψ, k}
    idx_old = getidx(Hk)
    idx_new = vcat(idx_old, reduced_margin)

    # Define updated map
    f_new = ExpandedFunction(Hk.I.f.f.B, idx_new, vcat(getcoeff(Hk), zeros(size(reduced_margin,1))))
    Hk_new = HermiteMapk(f_new; α = 1e-6)

    # Set coefficients based on previous optimal solution
    coeff_new, coeff_idx_added, idx_added = update_coeffs(Hk, Hk_new)

    # Compute gradient after adding the new elements
    S = update_storage(S, X, reduced_margin)
    dJ = zero(coeff_new)
    negative_log_likelihood!(nothing, dJ, coeff_new, S, Hk_new, X)

    # Find function in the reduced margin most correlated with the residual
    _, opt_dJ_coeff_idx = findmax(abs.(dJ[coeff_idx_added]))

    opt_idx = idx_added[opt_dJ_coeff_idx,:]

    # Update multi-indices and the reduced margins based on opt_idx
    # reducedmargin_opt_coeff_idx = Bool[opt_idx == x for x in eachslice(reduced_margin; dims = 1)]

    idx_new, reduced_margin = updatereducedmargin(idx_old, reduced_margin, opt_dJ_coeff_idx)

    return idx_new, reduced_margin
end

function update_coeffs(Hkold::HermiteMapk{m, Nψ, k}, Hknew::HermiteMapk{m, Nψnew, k}) where {m, Nψ, Nψnew, k}

    idx_old = getidx(Hkold)
    idx_new = getidx(Hknew)

    coeff_old = getcoeff(Hkold)

    # Declare vectors for new coefficients and to track added terms

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
