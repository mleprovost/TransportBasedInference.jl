export optimize

function optimize(C::SparseRadialMapComponent, X, poff::Union{Nothing, Int64}, pdiag::Union{Int64, Nothing}, maxfamiliesoff::Union{Nothing, Int64, String},
                  λ::Float64, δ::Float64, γ::Float64;
                  maxpatience::Int64 = 10^5, verbose::Bool = false)
    Nx = C.Nx
    # By default the diagonal component is selected
    if typeof(maxfamiliesoff) <: Nothing
        C, error = optimize(C, X, nothing, λ, δ)
    elseif typeof(maxfamiliesoff) <: Int64
        C, error =  greedyfit(Nx, poff, pdiag, X, maxfamiliesoff, λ, δ, γ;
                              verbose = verbose)

    elseif maxfamiliesoff ∈ ("kfold", "kfolds", "Kfold", "Kfolds")
        # Define cross-validation splits of data
        n_folds = 5
        folds = kfolds(1:size(X,2), k = n_folds)

        # Run greedy approximation
        if pdiag == 0
            maxfamily = min(Nx-1, ceil(Int64, (sqrt(size(X,2))-(pdiag+1))/(poff+1)))
        elseif pdiag > 0
            maxfamily = min(Nx-1, ceil(Int64, (sqrt(size(X,2))-(pdiag+3))/(poff+1)))
        else
            error("Wrong value for pdiag")
        end

        valid_error = zeros(maxfamily+1, n_folds)
        # if typeof(P) <: Serial
        @inbounds for i=1:n_folds
            if verbose == true
                println("Fold "*string(i)*":")
            end
            idx_train, idx_valid = folds[i]

            C, error = greedyfit(Nx, poff, pdiag, X[:,idx_train], X[:,idx_valid], maxfamily,
                                 λ, δ, γ;
                                 maxpatience = maxpatience, verbose  = verbose)

            # error[2] contains the history of the validation error
            valid_error[:,i] .= copy(error[2])
        end
        # elseif typeof(P) <: Thread
          # @inbounds   Threads.@threads for i=1:n_folds
          #     if verbose == true
          #         println("Fold "*string(i)*":")
          #     end
          #     idx_train, idx_valid = folds[i]
          #
          #     C, error = greedyfit(Nx, poff, pdiag, X[:,idx_train], X[:,idx_valid], maxfamily,
          #                          λ, δ, γ;
          #                          maxpatience = maxpatience, verbose  = verbose)
          #
          #     # error[2] contains the history of the validation error
          #     valid_error[:,i] .= copy(error[2])
          # end
        # end

        # Find optimal numbers of terms
        mean_valid_error = mean(valid_error, dims  = 2)[:,1]
        _, opt_families = findmin(mean_valid_error)
        # opt_families corresponds to the number of off-diagonal components
        opt_families -= 1

        if verbose == true
            println("Optimization on the entire data set:")
        end
        # Run greedy fit up to opt_nterms on all the data
        C, error = greedyfit(Nx, poff, pdiag, X, opt_families, λ, δ, γ; verbose = verbose)

    elseif maxfamiliesoff ∈ ("split", "Split")
        nvalid = ceil(Int64, floor(0.2*size(X,2)))
        X_train = X[:,nvalid+1:end]
        X_valid = X[:,1:nvalid]

        # Run greedy approximation
        maxfamily =  min(Nx-1, ceil(Int64, sqrt(size(X,2))))

        C, error = greedyfit(Nx, poff, pdiag, X_train, X_valid, maxfamily, λ, δ, γ;
                             maxpatience = maxpatience, verbose  = verbose)
    else
        println("Argument max_terms is not recognized")
        error()
    end
    return C, error
end

function optimize(S::SparseRadialMap, X::AbstractMatrix{Float64}, poff::Int64, pdiag::Union{Int64, Vector{Int64}},
	maxfamilies::Union{Int64, Nothing, String}; start::Int64=1, maxpatience::Int64=10^5, verbose::Bool=true, P::Parallel=serial)
	NxX, Ne = size(X)
	@get S (Nx, p, γ, λ, δ, κ)

	@assert NxX == Nx "Wrong dimension of the ensemble matrix X"

	# Compute centers and widths
	center_std!(S, X)

	# Optimize coefficients
	# Skip the identity components of the map
	if typeof(P)==Serial
		@inbounds for i=start:Nx
			if !allequal(p[i], -1) || !(typeof(maxfamilies) <: Nothing)
				if typeof(pdiag) <: Vector{Int64}
					S.C[i], _ = optimize(S.C[i], X[1:i,:], poff, pdiag[i-start+1], maxfamilies, λ, δ, γ;
					                     maxpatience = maxpatience, verbose = verbose)
					copy!(S.p[i], S.C[i].p)
				else
					S.C[i], _ = optimize(S.C[i], X[1:i,:], poff, pdiag, maxfamilies, λ, δ, γ;
					                     maxpatience = maxpatience, verbose = verbose)
					copy!(S.p[i], S.C[i].p)

				end
			end
		end
	else
		@inbounds Threads.@threads for i=start:Nx
			if !allequal(p[i], -1) || !(typeof(maxfamilies) <: Nothing)
				if typeof(pdiag) <: Vector{Int64}
					S.C[i], _ = optimize(S.C[i], X[1:i,:], poff, pdiag[i-start+1], maxfamilies, λ, δ, γ;
					                     maxpatience = maxpatience, verbose = verbose)
					copy!(S.p[i], S.C[i].p)
				else
					S.C[i], _ = optimize(S.C[i], X[1:i,:], poff, pdiag, maxfamilies, λ, δ, γ;
					                     maxpatience = maxpatience, verbose = verbose)
					copy!(S.p[i], S.C[i].p)
				end
			end
		end
	end
	return S
end
