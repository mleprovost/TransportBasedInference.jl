export greedyfit, update_component, update_coeffs


# function greedyfit(m::Int64, Nx::Int64, X::Array{Float64,2}, Xvalid::Array{Float64,2}, maxterms::Int64; maxpatience::Int64 = 10^5, verbose::Bool = true)# where {m, Nψ, Nx}

function greedyfit(m::Int64, Nx::Int64, X, Xvalid, maxterms::Int64; withconstant::Bool = false,
                   maxpatience::Int64 = 10^5, verbose::Bool = true)

    best_valid_error = Inf
    patience = 0

    train_error = Float64[]
    valid_error = Float64[]

    # Initialize map C to identity
    C = MapComponent(m, Nx; α = 1e-6);

    # Compute storage # Add later storage for validation S_valid
    S = Storage(C.I.f, X)
    Svalid = Storage(C.I.f, Xvalid)

    # Compute initial loss on training set
    push!(train_error, negative_log_likelihood!(0.0, nothing, getcoeff(C), S, C, X))
    push!(valid_error, negative_log_likelihood!(0.0, nothing, getcoeff(C), Svalid, C, Xvalid))

    if verbose == true
        println(string(ncoeff(C))*" terms - Training error: "*
        string(train_error[end])*", Validation error: "*string(valid_error[end]))
    end

    # Remove or not the constant i.e the multi-index [0 0 0], and
    # optimize for the first non-zero index
    if withconstant == false
        # Compute the reduced margin
        reduced_margin = getreducedmargin(getidx(C))
        f = ExpandedFunction(C.I.f.f.B, reduced_margin, zeros(size(reduced_margin,1)))
        C = MapComponent(f; α = 1e-6)
        S = Storage(C.I.f, X)
        coeff0 = getcoeff(C)
        dJ = zero(coeff0)

        negative_log_likelihood!(nothing, dJ, coeff0, S, C, X)
        _, opt_dJ_coeff_idx = findmax(abs.(dJ))

        opt_idx = reduced_margin[opt_dJ_coeff_idx:opt_dJ_coeff_idx,:]

        f = ExpandedFunction(C.I.f.f.B, opt_idx, zeros(size(opt_idx,1)))
        C = MapComponent(f; α = 1e-6)
        S = Storage(C.I.f, X)
        Svalid = Storage(C.I.f, Xvalid)

    end

    # Optimize C with the first idx: = zeros(Int64,1,C.Nx) or a non-zero one if withconstant == false
    coeff0 = getcoeff(C)
    precond = zeros(ncoeff(C), ncoeff(C))
    precond!(precond, coeff0, S, C, X)
    res = Optim.optimize(Optim.only_fg!(negative_log_likelihood(S, C, X)), coeff0,
          Optim.LBFGS(; m = 20, P = Preconditioner(precond)))
          
    setcoeff!(C, Optim.minimizer(res))

    # Compute initial loss on training set
    push!(train_error, negative_log_likelihood!(0.0, nothing, getcoeff(C), S, C, X))
    push!(valid_error, negative_log_likelihood!(0.0, nothing, getcoeff(C), Svalid, C, Xvalid))

    if verbose == true
        println(string(ncoeff(C))*" terms - Training error: "*
        string(train_error[end])*", Validation error: "*string(valid_error[end]))
    end


    # Compute the reduced margin
    reduced_margin = getreducedmargin(getidx(C))

    while  ncoeff(C) <= maxterms-1
        idx_new, reduced_margin = update_component(C, X, reduced_margin, S)

        # Update storage with the new feature
        S = update_storage(S, X, idx_new[end:end,:])
        Svalid = update_storage(Svalid, Xvalid, idx_new[end:end,:])

        # Update C
        C = MapComponent(IntegratedFunction(S.f); α = C.α)

        # Optimize coefficients
        coeff0 = getcoeff(C)
        precond = zeros(ncoeff(C), ncoeff(C))
        precond!(precond, coeff0, S, C, X)

        res = Optim.optimize(Optim.only_fg!(negative_log_likelihood(S, C, X)), coeff0,
              Optim.LBFGS(; m = 20, P = Preconditioner(precond)))

        setcoeff!(C, Optim.minimizer(res))

        # Compute new loss on training and validation sets
        push!(train_error, negative_log_likelihood!(0.0, nothing, getcoeff(C), S, C, X))
        push!(valid_error, negative_log_likelihood!(0.0, nothing, getcoeff(C), Svalid, C, Xvalid))

        if verbose == true
            println(string(ncoeff(C))*" terms - Training error: "*
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

    return C, (train_error, valid_error)
end


# function greedyfit(m::Int64, Nx::Int64, X::Array{Float64,2}, maxterms::Int64; maxpatience::Int64 = 10^5, verbose::Bool = true)# where {m, Nψ, Nx}

function greedyfit(m::Int64, Nx::Int64, X, maxterms::Int64; withconstant::Bool = false, maxpatience::Int64 = 10^5, verbose::Bool = true)# where {m, Nψ, Nx}

    @assert maxterms >=1 "maxterms should be >= 1"
    best_valid_error = Inf
    patience = 0

    train_error = Float64[]

    # Initialize map C to identity
    C = MapComponent(m, Nx; α = 1e-6);

    # Compute storage # Add later storage for validation S_valid
    S = Storage(C.I.f, X)

    # Compute initial loss on training set
    push!(train_error, negative_log_likelihood!(0.0, nothing, getcoeff(C), S, C, X))

    if verbose == true
        println(string(ncoeff(C))*" terms - Training error: "*string(train_error[end]))
    end

    # Remove or not the constant i.e the multi-index [0 0 0]
    if withconstant == false
        # Compute the reduced margin
        reduced_margin = getreducedmargin(getidx(C))
        f = ExpandedFunction(C.I.f.f.B, reduced_margin, zeros(size(reduced_margin,1)))
        C = MapComponent(f; α = 1e-6)
        S = Storage(C.I.f, X)
        coeff0 = getcoeff(C)
        dJ = zero(coeff0)

        negative_log_likelihood!(nothing, dJ, coeff0, S, C, X)
        _, opt_dJ_coeff_idx = findmax(abs.(dJ))

        opt_idx = reduced_margin[opt_dJ_coeff_idx:opt_dJ_coeff_idx,:]

        f = ExpandedFunction(C.I.f.f.B, opt_idx, zeros(size(opt_idx,1)))
        C = MapComponent(f; α = 1e-6)
        S = Storage(C.I.f, X)
    end

    coeff0 = getcoeff(C)
    precond = zeros(ncoeff(C), ncoeff(C))
    precond!(precond, coeff0, S, C, X)
    res = Optim.optimize(Optim.only_fg!(negative_log_likelihood(S, C, X)), coeff0,
          Optim.LBFGS(; m = 20, P = Preconditioner(precond)))

    setcoeff!(C, Optim.minimizer(res))

    # Compute initial loss on training set
    push!(train_error, negative_log_likelihood!(0.0, nothing, getcoeff(C), S, C, X))

    if verbose == true
        println(string(ncoeff(C))*" terms - Training error: "*string(train_error[end]))
    end

    # Compute the reduced margin
    reduced_margin = getreducedmargin(getidx(C))

    while ncoeff(C) <= maxterms-1
        idx_new, reduced_margin = update_component(C, X, reduced_margin, S)

        # Update storage with the new feature
        S = update_storage(S, X, idx_new[end:end,:])

        # Update C
        C = MapComponent(IntegratedFunction(S.f); α = C.α)

        # Optimize coefficients
        coeff0 = getcoeff(C)
        precond = zeros(ncoeff(C), ncoeff(C))
        precond!(precond, coeff0, S, C, X)

        res = Optim.optimize(Optim.only_fg!(negative_log_likelihood(S, C, X)), coeff0,
              Optim.LBFGS(; m = 20, P = Preconditioner(precond)))

        setcoeff!(C, Optim.minimizer(res))

        # Compute new loss on the training and validation set
        push!(train_error, negative_log_likelihood!(0.0, nothing, getcoeff(C), S, C, X))

        if verbose == true
            println(string(ncoeff(C))*" terms - Training error: "*string(train_error[end]))
        end
    end

    return C, train_error
end

# function update_component(C::MapComponent{m, Nψ, Nx}, X::Array{Float64,2}, reduced_margin::Array{Int64,2}, S::Storage{m, Nψ, Nx}) where {m, Nψ, Nx}

function update_component(C::MapComponent, X, reduced_margin::Array{Int64,2}, S::Storage)
    m = C.m
    Nψ = C.Nψ
    idx_old = getidx(C)

    idx_new = vcat(idx_old, reduced_margin)

    # Define updated map
    f_new = ExpandedFunction(C.I.f.f.B, idx_new, vcat(getcoeff(C), zeros(size(reduced_margin,1))))
    C_new = MapComponent(f_new; α = 1e-6)


    # Set coefficients based on previous optimal solution
    coeff_new, coeff_idx_added, idx_added = update_coeffs(C, C_new)

    # Compute gradient after adding the new elements
    S = update_storage(S, X, reduced_margin)
    dJ = zero(coeff_new)
    negative_log_likelihood!(nothing, dJ, coeff_new, S, C_new, X)

    # Find function in the reduced margin most correlated with the residual
    _, opt_dJ_coeff_idx = findmax(abs.(dJ[coeff_idx_added]))

    opt_idx = idx_added[opt_dJ_coeff_idx,:]

    # Update multi-indices and the reduced margins based on opt_idx
    # reducedmargin_opt_coeff_idx = Bool[opt_idx == x for x in eachslice(reduced_margin; dims = 1)]

    idx_new, reduced_margin = updatereducedmargin(idx_old, reduced_margin, opt_dJ_coeff_idx)

    return idx_new, reduced_margin
end

function update_coeffs(Cold::MapComponent, Cnew::MapComponent)

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
