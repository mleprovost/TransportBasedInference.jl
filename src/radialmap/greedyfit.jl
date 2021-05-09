export greedyfit, gradient_off!, update_component!


"""
$(TYPEDSIGNATURES)

An adaptive routine to estimate a sparse approximation of an `SparseRadialMapComponent` based on  the pair of ensemble matrices `X` (training set) and `Xvalid` (validation set).
"""
function greedyfit(Nx::Int64, poff::Int64, pdiag::Int64,  X::AbstractMatrix{Float64}, Xvalid::AbstractMatrix{Float64}, maxfamilies::Int64,
                   λ::Float64, δ::Float64, γ::Float64; maxpatience::Int64 = 10^5, verbose::Bool=false)

    train_error = Float64[]
    valid_error = Float64[]

    NxX, Ne = size(X)
    # The widths and centers are computed on the entire set.
    Xsort = deepcopy(sort(hcat(X, Xvalid); dims = 2))
    @assert poff > -1 || pdiag > -1 "The order poff and pdiag of the features must be > 0"
    @assert λ == 0 "Greedy fit is only implemented for λ = 0"
    @assert NxX == Nx "Wrong dimension of the ensemble matrix `X`"

    # Initialize a sparse radial map component C with only a diagonal term of order p
    order = fill(-1, Nx)
    order[end] = pdiag
    C = SparseRadialMapComponent(Nx, order)

    center_std!(C, Xsort; γ = γ)
    x_diag = optimize(C, X, λ, δ)
    modify_a!(C, x_diag)

    # Compute loss on training and validation sets
    push!(train_error, copy(negative_likelihood(C, X)))
    push!(valid_error, copy(negative_likelihood(C, Xvalid)))

    if verbose == true
        println(string(size(C.activedim,1))*" active dimensions  - Training error: "*
        string(train_error[end])*", Validation error: "*string(valid_error[end]))
    end

    if Nx>1 || maxfamilies>0

        best_valid_error = Inf
        patience = 0

        # Create a radial map with order p for all the entries
        Cfull = SparseRadialMapComponent(Nx, vcat(poff*ones(Int64, Nx-1), pdiag))

        # Compute centers and widths
        center_std!(Cfull, Xsort; γ = γ)

        # Create weights for the training and validation sets
        ψ_off, ψ_diag, dψ_diag = compute_weights(Cfull, X)
        ψ_offvalid, ψ_diagvalid, dψ_diagvalid = compute_weights(Cfull, Xvalid)

        n_off = size(ψ_off,2)
        n_diag = size(ψ_diag,2)


        Asqrt = zero(ψ_diag)
        # Normalize diagtone basis functions
        μψ = copy(mean(ψ_diag, dims=1)[1,:])
        σψ = copy(std(ψ_diag, dims = 1, corrected = false)[1,:])
        # σψ = copy(norm.(eachcol(ψ_diag)))
        ψ_diagscaled = copy(ψ_diag)
        dψ_diagscaled = copy(dψ_diag)

        ψ_diagscaled .-= μψ'
        ψ_diagscaled ./= σψ'
        dψ_diagscaled ./= σψ'

        lhd = LHD(zeros(n_diag,n_diag), dψ_diagscaled, λ, δ)

        ψ_offscaled = copy(ψ_off)
        μψ_off = copy(mean(ψ_off, dims = 1)[1,:])
        # σψ_off = copy(std(ψ_off, dims = 2, corrected = false)
        # σψ_off = copy(norm.(eachcol(ψ_off)))
        σψ_off = copy(std(ψ_off, dims = 1, corrected = false)[1,:])

        ψ_offscaled .-= μψ_off'
        ψ_offscaled ./= σψ_off'

        # rhs = -ψ_diag x_diag
        rhs = zeros(Ne)
        mul!(rhs, ψ_diag, x_diag[2:n_diag+1])
        rhs .+= x_diag[1]
        rmul!(rhs, -1.0)

        # Create updatable QR factorization
        # For the greedy optimization, we don't use a L2 regularization λ||x||^2,
        # since we are already making a greedy selection of the features
        F = qrfactUnblocked(zeros(0,0))
        # Off-diagonal coefficients will be permuted based on the greedy procedure

        candidates = collect(1:Nx-1)
        # Unordered off-diagonal active dimensions
        offdims = Int64[]

        # Compute the gradient of the different basis
        dJ = zeros((poff+1)*(Nx-1))

        x_off = zeros((poff+1)*(Nx-1))
        x_offsparse = Float64[]
        tmp_off = Float64[]
        tmp_diag = zeros(n_diag)


        budget = min(maxfamilies, Nx-1)
        # Compute the norm of the different candidate features
        sqnormfeatures = map(i-> norm(view(ψ_off, (i-1)*(poff+1)+1:i*(poff+1)))^2, candidates)
        cache = zeros(Ne)
        @inbounds for i=1:budget
            # Compute the gradient of the different basis (use the unscaled basis evaluations)
            mul!(rhs, ψ_diag, view(x_diag,2:n_diag+1))
            rhs .+= x_diag[1]
            rmul!(rhs, -1.0)
            gradient_off!(dJ, cache, ψ_off, x_off, rhs, Ne)

            _, max_dim = findmax(map(k-> norm(view(dJ, (k-1)*(poff+1)+1:k*(poff+1)))^2/sqnormfeatures[k], candidates))
            new_dim = candidates[max_dim]
            push!(offdims, copy(new_dim))
            append!(tmp_off, zeros(poff+1))
            append!(x_offsparse, zeros(poff+1))

            # Update storage in C
            update_component!(C, poff, new_dim)

            # Compute center and std for this new family of features
            # The centers and widths have already been computed in Cfull
            copy!(C.ξ[new_dim], Cfull.ξ[new_dim])
            copy!(C.σ[new_dim], Cfull.σ[new_dim])

            # Update qr factorization with the new family of features
            if i == 1
                F = qrfactUnblocked(ψ_offscaled[:,(new_dim-1)*(poff+1)+1:new_dim*(poff+1)])
            else
                F = updateqrfactUnblocked!(F, view(ψ_offscaled,:,(new_dim-1)*(poff+1)+1:new_dim*(poff+1)))
            end
            Asqrt .= ψ_diagscaled - fast_mul2(ψ_diagscaled, F.Q, Ne, (poff+1)*size(offdims, 1))
            lhd.A .= (1/Ne)*Asqrt'*Asqrt

            if C.p[Nx] == 0
                @assert size(lhd.A)==(1,1) "Quadratic matrix should be a scalar."
                # This equation is equation (A.9) Couplings for nonlinear ensemble filtering
                # uNx(z) = c + α z so α = 1/√κ*
                fill!(tmp_diag, 1.0)
                tmp_diag[1] = sqrt(1/lhd.A[1,1])
            else
                # Update A and b of the loss function lhd
                # Add L-2 regularization

                fill!(tmp_diag, 1.0)

                @inbounds for i=1:n_diag
                    lhd.A[i,i] += (λ/Ne)
                end
                # Update b coefficient
                lhd.b .= δ*sum(lhd.A, dims=2)[:,1]

                tmp_diag,_,_,_ = projected_newton(tmp_diag, lhd, "TrueHessian")
                tmp_diag .+= δ
            end

            cache .= ψ_diagscaled*tmp_diag
            tmp_off .= -F.R\((F.Q'*cache)[1:(poff+1)*size(offdims, 1)])
            # Rescale coefficients
            for j=1:n_diag

                x_diag[j+1] = tmp_diag[j]/σψ[j]
            end

            for (j, offdimj) in enumerate(offdims)
                tmp_off[(j-1)*(poff+1)+1:j*(poff+1)] ./= σψ_off[(offdimj-1)*(poff+1)+1:offdimj*(poff+1)]
            end

            # Compute and store the constant term (expectation of selected terms)
            x_diag[1] = -dot(μψ, x_diag[2:n_diag+1]) #-dot(μψ_off, tmp_off)

            # Make sure that active dim are in the right order when we affect coefficient.
            # For the split and kfold compute the training and validation losses.
            fill!(x_offsparse, 0.0)
            # Compute the permutation associated to offdims
            perm = sortperm(offdims)

            for (j, offdimj) in enumerate(offdims)
                tmp_offj = tmp_off[(j-1)*(poff+1)+1:j*(poff+1)]
                x_diag[1] -= dot(view(μψ_off, (offdimj-1)*(poff+1)+1:offdimj*(poff+1)), tmp_offj)
                view(x_off, (offdimj-1)*(poff+1)+1:offdimj*(poff+1)) .= tmp_offj
            end

            for (j, offdimj) in enumerate(offdims)
                tmp_offj = tmp_off[(perm[j]-1)*(poff+1)+1:perm[j]*(poff+1)]
                view(x_offsparse, (j-1)*(poff+1)+1:j*(poff+1)) .= tmp_offj
            end

            @assert norm(x_off[x_off .!= 0.0] - x_offsparse[x_offsparse .!= 0.0])<1e-10  "Error in x_off"

            modify_a!(C, vcat(x_offsparse, x_diag))
            filter!(x-> x!= new_dim, candidates)

            # Compute loss on training and validation sets
            push!(train_error, copy(negative_likelihood(C, X)))
            push!(valid_error, copy(negative_likelihood(C, Xvalid)))

            if verbose == true
                println(string(size(C.activedim,1))*" active dimensions  - Training error: "*
                string(train_error[end])*", Validation error: "*string(valid_error[end]))
            end

            # Update patience
            if valid_error[end] >= best_valid_error
                patience +=1
            else
                best_valid_error = copy(valid_error[end])
                patience = 0
            end

            # Check if patience exceeded maximum patience
            if patience >= maxpatience
                break
            end
        end
    end
    return C, (train_error, valid_error)
end

greedyfit(Nx::Int64, p::Int64, X::AbstractMatrix{Float64}, Xvalid::AbstractMatrix{Float64},
          maxfamilies::Int64, λ::Float64, δ::Float64, γ::Float64;
          maxpatience::Int64 = 10^5, verbose::Bool=false) =
          greedyfit(Nx, p, p, X, Xvalid, maxfamilies, λ, δ, γ; maxpatience = maxpatience, verbose = verbose)





"""
$(TYPEDSIGNATURES)

An adaptive routine to estimate a sparse approximation of an `SparseRadialMapComponent` based on  the ensemble matrix `X`.
"""
function greedyfit(Nx::Int64, poff::Int64, pdiag::Int64, X::AbstractMatrix{Float64},
                   maxfamilies::Int64, λ::Float64, δ::Float64, γ::Float64; maxpatience::Int64 = 10^5, verbose::Bool = false)

    NxX, Ne = size(X)
    Xsort = deepcopy(sort(X; dims = 2))
    train_error = Float64[]
    @assert poff > -1 || pdiag > -1 "The order poff and pdiag of the features must be > 0"
    @assert λ == 0 "Greedy fit is only implemented for λ = 0"
    @assert NxX == Nx "Wrong dimension of the ensemble matrix `X`"

    # Initialize a sparse radial map component C with only a diagonal term of order p
    order = fill(-1, Nx)
    order[end] = pdiag
    C = SparseRadialMapComponent(Nx, order)

    center_std!(C, Xsort; γ = γ)
    x_diag = optimize(C, X, λ, δ)
    modify_a!(C, x_diag)
    push!(train_error, copy(negative_likelihood(C, X)))

    if verbose == true
        # Compute loss on training and validation sets
        println(string(size(C.activedim,1))*" active dimensions  - Training error: "*
                string(train_error[end]))
    end

    if Nx>1 || maxfamilies>0

        best_train_error = Inf
        patience = 0

        # Create a radial map with order p for all the entries
        Cfull = SparseRadialMapComponent(Nx, vcat(poff*ones(Int64, Nx-1), pdiag))

        # Compute centers and widths
        center_std!(Cfull, Xsort; γ = γ)

        # Create weights
        ψ_off, ψ_diag, dψ_diag = compute_weights(Cfull, X)
        n_off = size(ψ_off,2)
        n_diag = size(ψ_diag,2)


        Asqrt = zero(ψ_diag)
        # Normalize diagtone basis functions
        μψ = copy(mean(ψ_diag, dims=1)[1,:])
        σψ = copy(std(ψ_diag, dims = 1, corrected = false)[1,:])
        # σψ = copy(norm.(eachcol(ψ_diag)))
        ψ_diagscaled = copy(ψ_diag)
        dψ_diagscaled = copy(dψ_diag)

        ψ_diagscaled .-= μψ'
        ψ_diagscaled ./= σψ'
        dψ_diagscaled ./= σψ'

        lhd = LHD(zeros(n_diag,n_diag), dψ_diagscaled, λ, δ)


        ψ_offscaled = copy(ψ_off)
        μψ_off = copy(mean(ψ_off, dims = 1)[1,:])
        # σψ_off = copy(std(ψ_off, dims = 2, corrected = false)
        # σψ_off = copy(norm.(eachcol(ψ_off)))
        σψ_off = copy(std(ψ_off, dims = 1, corrected = false)[1,:])

        ψ_offscaled .-= μψ_off'
        ψ_offscaled ./= σψ_off'

        # rhs = -ψ_diag x_diag
        rhs = zeros(Ne)
        mul!(rhs, ψ_diag, x_diag[2:n_diag+1])
        rhs .+= x_diag[1]
        rmul!(rhs, -1.0)

        # Create updatable QR factorization
        # For the greedy optimization, we don't use a L2 regularization λ||x||^2,
        # since we are already making a greedy selection of the features
        F = qrfactUnblocked(zeros(0,0))
        # Off-diagonal coefficients will be permuted based on the greedy procedure

        candidates = collect(1:Nx-1)
        # Unordered off-diagonal active dimensions
        offdims = Int64[]

        # Compute the gradient of the different basis
        dJ = zeros((poff+1)*(Nx-1))

        x_off = zeros((poff+1)*(Nx-1))
        x_offsparse = Float64[]
        tmp_off = Float64[]
        tmp_diag = zeros(n_diag)


        budget = min(maxfamilies, Nx-1)
        # Compute the norm of the different candidate features
        sqnormfeatures = map(i-> norm(view(ψ_off, (i-1)*(poff+1)+1:i*(poff+1)))^2, candidates)
        cache = zeros(Ne)
        @inbounds for i=1:budget
            # Compute the gradient of the different basis (use the unscaled basis evaluations)
            mul!(rhs, ψ_diag, view(x_diag,2:n_diag+1))
            rhs .+= x_diag[1]
            rmul!(rhs, -1.0)
            gradient_off!(dJ, cache, ψ_off, x_off, rhs, Ne)

            _, max_dim = findmax(map(k-> norm(view(dJ, (k-1)*(poff+1)+1:k*(poff+1)))^2/sqnormfeatures[k], candidates))
            new_dim = candidates[max_dim]
            push!(offdims, copy(new_dim))
            append!(tmp_off, zeros(poff+1))
            append!(x_offsparse, zeros(poff+1))

            # Update storage in C
            update_component!(C, poff, new_dim)

            # Compute center and std for this new family of features
            # The centers and widths have already been computed in Cfull
            copy!(C.ξ[new_dim], Cfull.ξ[new_dim])
            copy!(C.σ[new_dim], Cfull.σ[new_dim])

            # Update qr factorization with the new family of features
            if i == 1
                F = qrfactUnblocked(ψ_offscaled[:,(new_dim-1)*(poff+1)+1:new_dim*(poff+1)])
            else
                F = updateqrfactUnblocked!(F, view(ψ_offscaled,:,(new_dim-1)*(poff+1)+1:new_dim*(poff+1)))
            end
            Asqrt .= ψ_diagscaled - fast_mul2(ψ_diagscaled, F.Q, Ne, (poff+1)*size(offdims, 1))
            lhd.A .= (1/Ne)*Asqrt'*Asqrt

            if C.p[Nx] == 0
                @assert size(lhd.A)==(1,1) "Quadratic matrix should be a scalar."
                # This equation is equation (A.9) Couplings for nonlinear ensemble filtering
                # uNx(z) = c + α z so α = 1/√κ*
                fill!(tmp_diag, 1.0)
                tmp_diag[1] = sqrt(1/lhd.A[1,1])
            else
                # Update A and b of the loss function lhd
                # Add L-2 regularization

                fill!(tmp_diag, 1.0)

                @inbounds for i=1:n_diag
                    lhd.A[i,i] += (λ/Ne)
                end
                # Update b coefficient
                lhd.b .= δ*sum(lhd.A, dims=2)[:,1]

                tmp_diag,_,_,_ = projected_newton(tmp_diag, lhd, "TrueHessian")
                tmp_diag .+= δ
            end

            cache .= ψ_diagscaled*tmp_diag
            tmp_off .= -F.R\((F.Q'*cache)[1:(poff+1)*size(offdims, 1)])
            # Rescale coefficients
            for j=1:n_diag
                x_diag[j+1] = tmp_diag[j]/σψ[j]
            end

            for (j, offdimj) in enumerate(offdims)
                tmp_off[(j-1)*(poff+1)+1:j*(poff+1)] ./= σψ_off[(offdimj-1)*(poff+1)+1:offdimj*(poff+1)]
            end

            # Compute and store the constant term (expectation of selected terms)
            x_diag[1] = -dot(μψ, x_diag[2:n_diag+1]) #-dot(μψ_off, tmp_off)

            # Make sure that active dim are in the right order when we affect coefficient.
            # For the split and kfold compute the training and validation losses.
            fill!(x_offsparse, 0.0)
            # Compute the permutation associated to offdims
            perm = sortperm(offdims)

            for (j, offdimj) in enumerate(offdims)
                tmp_offj = tmp_off[(j-1)*(poff+1)+1:j*(poff+1)]
                x_diag[1] -= dot(view(μψ_off, (offdimj-1)*(poff+1)+1:offdimj*(poff+1)), tmp_offj)
                view(x_off, (offdimj-1)*(poff+1)+1:offdimj*(poff+1)) .= tmp_offj
            end

            for (j, offdimj) in enumerate(offdims)
                tmp_offj = tmp_off[(perm[j]-1)*(poff+1)+1:perm[j]*(poff+1)]
                view(x_offsparse, (j-1)*(poff+1)+1:j*(poff+1)) .= tmp_offj
            end

            @assert norm(x_off[x_off .!= 0.0] - x_offsparse[x_offsparse .!= 0.0])<1e-10  "Error in x_off"

            modify_a!(C, vcat(x_offsparse, x_diag))
            push!(train_error, copy(negative_likelihood(C, X)))
            filter!(x-> x!= new_dim, candidates)

            if verbose == true
                # Compute loss on training and validation sets
                println(string(size(C.activedim,1))*" active dimensions  - Training error: "*
                        string(train_error[end]))
            end

            # Update patience
            if train_error[end] >= best_train_error
                patience +=1
            else
                best_train_error = copy(train_error[end])
                patience = 0
            end

            # Check if patience exceeded maximum patience
            if patience >= maxpatience
                break
            end
        end
    end
    return C, train_error
end

greedyfit(Nx::Int64, p::Int64, X::AbstractMatrix{Float64}, maxfamilies::Int64,
          λ::Float64, δ::Float64, γ::Float64; maxpatience::Int64 = 10^5, verbose::Bool=false) =
          greedyfit(Nx, p, p, X, maxfamilies, λ, δ, γ; maxpatience = maxpatience, verbose = verbose)


function gradient_off!(dJ::AbstractVector{Float64}, cache::AbstractVector{Float64}, ψ_off::AbstractMatrix{Float64}, x_off, rhs, Ne::Int64)
    fill!(dJ, 0.0)
    cache .= ψ_off*x_off - rhs
    dJ .= (1/Ne)*ψ_off'*cache
end

gradient_off!(dJ::AbstractVector{Float64}, ψ_off::AbstractMatrix{Float64}, x_off, rhs, Ne::Int64) =
              gradient_off!(dJ, zeros(Ne), ψ_off, x_off, rhs, Ne)


function update_component!(C::SparseRadialMapComponent, p::Int64, new_dim::Int64)
    @assert new_dim <= C.Nx
    if new_dim == C.Nx
        if p == -1
            C.p[new_dim] = p
            C.ξ[new_dim] = Float64[]
            C.σ[new_dim] = Float64[]
            C.a[new_dim] = Float64[]
        elseif p == 0
            C.p[new_dim] = p
            push!(C.activedim, new_dim)
            sort!(C.activedim)
            C.ξ[new_dim] = zeros(p)
            C.σ[new_dim] = zeros(p)
            C.a[new_dim] = zeros(p+2)
        else
            C.p[new_dim] = p
            push!(C.activedim, new_dim)
            sort!(C.activedim)
            C.ξ[new_dim] = zeros(p+2)
            C.σ[new_dim] = zeros(p+2)
            C.a[new_dim] = zeros(p+3)
        end
    else
        if p == -1
            C.p[new_dim] = p
            C.ξ[new_dim] = Float64[]
            C.σ[new_dim] = Float64[]
            C.a[new_dim] = Float64[]
        elseif p == 0
            C.p[new_dim] = p
            push!(C.activedim, new_dim)
            sort!(C.activedim)
            C.ξ[new_dim] = zeros(p)
            C.σ[new_dim] = zeros(p)
            C.a[new_dim] = zeros(p+1)
        else
            C.p[new_dim] = p
            push!(C.activedim, new_dim)
            sort!(C.activedim)
            C.ξ[new_dim] = zeros(p)
            C.σ[new_dim] = zeros(p)
            C.a[new_dim] = zeros(p+1)
        end
    end
end



    # maxfmaily is the maximal number of
    # if p == 0
    #     maxfamily = ceil(Int64, (sqrt(Ne)-(p+1))/(p+1))
    # elseif p > 0
    #     maxfamily = ceil(Int64, (sqrt(Ne)-(p+3))/(p+1))
    # else
    #     error("Wrong value for p")
    # end
