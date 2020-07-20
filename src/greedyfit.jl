export greedyfit, update_component, update_coeffs



function greedyfit(Hk::HermiteMapk{m, Nψ, k}, X::Array{Float64,2}, maxterms::Int64; maxpatience::Int64 = 10^5) where {m, Nψ, k}

    best_valid_error = Inf
    patience = 0

    # Initialize map Hk to identity
    Hk = HermiteMapk(k, m; α = 1e-6);

    # Compute storage # Add later storage for validation S_valid
    S = Storage(Hk, X)

    # Compute initial loss on training set
    train_err = negative_log_likelihood!(S, Hk, X)(0.0, nothing, getcoeff(Hk))

    # Compute the reduced margin
    reduced_margin = getreducedmargin(getidx(Hk))

    while ncoeff(Hk) < maxterms




    end
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
