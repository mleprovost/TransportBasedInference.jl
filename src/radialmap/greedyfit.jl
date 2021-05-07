export greedyfit, gradient_off!, update_component!

function greedyfit(Nx, p::Int64, X, maxfamilies::Int64, λ, δ, γ)

    NxX, Ne = size(X)
    Xsort = deepcopy(sort(X; dims = 2))
    @assert p > -1 "The order p of the features must be > 0"
    @assert λ == 0 "Greedy fit is only implemented for λ = 0"
    @assert NxX == Nx "Wrong dimension of the ensemble matrix `X`"

    # Initialize a sparse radial map component C with only a diagonal term of order p
    order = -1*ones(Int64, Nx)
    order[end] = p
    C = SparseRadialMapComponent(Nx, order)

    center_std!(C, Xsort; γ = γ)
    x_diag = optimize(C, X, λ, δ)
    modify_a!(C, x_diag)

    if Nx>1

        # Create a radial map with order p for all the entries
        Cfull = SparseRadialMapComponent(Nx, p)

        # Compute centers and widths
        center_std!(Cfull, Xsort; γ = γ)

        ### Evaluate the different basis

        # Create weights
        ψ_off, ψ_diag, dψ_diag = compute_weights(Cfull, X)
        n_off = size(ψ_off,1)
        n_diag = size(ψ_diag,1)

        Asqrt = zero(ψ_diag)
        lhd = LHD(zeros(n_diag,n_diag), Matrix(dψ_diag'), λ, δ)

        # Normalize diagtone basis functions
        μψ = mean(ψ_diag, dims=1)[1,:]
        # σψ = std(ψ_diag, dims=2, corrected=false)[:,1]
        σψ = norm.(eachslice(ψ_diag; dims = 1))[1,:]
        ψ_diagscaled = copy(ψ_diag)
        dψ_diagscaled = copy(dψ_diag)

        ψ_diagscaled .-= μψ'
        ψ_diagscaled ./= σψ'
        dψ_diagscaled ./= σψ'

        ψ_offscaled = copy(ψ_off)
        μψ_off = mean(ψ_off, dims = 1)[1,:]
        # σψ_off = std(ψ_off, dims = 2, corrected = false)
        σψ_off = norm.(eachslice(ψ_off; dims = 1))[1,:]
        ψ_offscaled .-= μψ_off'
        ψ_offscaled ./= σψ_off'

        # rhs = -ψ_diag x_diag
        rhs = zeros(Ne)
        ψ_diag'*x_diag[2:n_diag+1]
        mul!(rhs, ψ_diag', x_diag[2:n_diag+1])
        rhs .+= x_diag[1]
        rmul!(rhs, -1.0)

        # Create updatable QR factorization
        # For the greedy optimization, we don't use a L2 regularization λ||x||^2,
        # since we are already making a greedy selection of the features
        # F = qrfactUnblocked(zeros(0,0))
        # Off-diagonal coefficients will be permuted based on the greedy procedure

        candidates = collect(1:Nx-1)
        # Unordered off-diagonal active dimensions
        offdims = Int64[]

        # Compute the gradient of the different basis
        dJ = zeros((p+1)*(Nx-1))

        x_off = zeros((p+1)*(Nx-1))
        tmp_off = Float64[]

        budget = min(maxfamilies, Nx-1)
        # Compute the norm of the different candidate features
        sqnormfeatures = map(i-> norm(view(ψ_off, (i-1)*(p+1)+1:i*(p+1)))^2, candidates)
        cache = zeros(Ne)
        @inbounds for i=1:budget
            # Compute the gradient of the different basis (use the unscaled basis evaluations)
            mul!(rhs, ψ_diag', view(x_diag,2:n_diag+1))
            rhs .+= x_diag[1]
            rmul!(rhs, -1.0)
            gradient_off!(dJ, cache, ψ_off, x_off, rhs, Ne)

            _, max_dim = findmax(map(k-> norm(view(dJ, (k-1)*(p+1)+1:k*(p+1)))^2/sqnormfeatures[k], candidates))
            new_dim = candidates[max_dim]
            push!(offdims, copy(new_dim))
            tmp_off = vcat(tmp_off, zeros(p+1))

            # Update storage in C
            update_component!(C, p, new_dim)

            # Compute center and std for this new family of features
            # The centers and widths have already been computed in Cfull
            copy!(C.ξ[new_dim], Cfull.ξ[new_dim])
            copy!(C.σ[new_dim], Cfull.σ[new_dim])

            # Update qr factorization with the new family of features
            if i == 1
                F = qrfactUnblocked(ψ_offscaled[(new_dim-1)*(p+1)+1:new_dim*(p+1),:])
            else
                updateqrfactUnblocked!(F, view(ψ_offscaled,(new_dim-1)*(p+1)+1:new_dim*(p+1),:))
            end
            @show size(F)

            Asqrt .= ψ_diagscaled - fast_mul(ψ_diagscaled, F.Q, Ne, (p+1)*size(offdims, 1))
            BLAS.gemm!('N', 'T', 1/Ne, Asqrt, Asqrt, 1.0, lhd.A)

            if C.p[Nx] == 0
                @assert size(lhd.A)==(1,1) "Quadratic matrix should be a scalar."
                # This equation is equation (A.9) Couplings for nonlinear ensemble filtering
                # uNx(z) = c + α z so α = 1/√κ*
                x_diag[2] = sqrt(1/lhd.A[1,1])
            else
                # Update A and b of the loss function lhd
                # Add L-2 regularization
                fill!(x_diag, 1.0)
                lhd.A .+= (λ/Ne)*I
                # Update b coefficient
                lhd.b .= δ*sum(lhd.A, dims=2)[:,1]

                tmp_diag,_,_,_ = projected_newton(x_diag[2:end], lhd, "TrueHessian")
                tmp_diag .+= δ
            end
            cache .= ψ_diag*tmp_diag
            
            tmp_off .= -F.R\(F.Q'*cache)

            # Rescale coefficients
            for j=1:n_diag
                x_diag[j+1] = tmp_diag[j]/σψ[j]
            end

            tmp_off ./= σψ_off

            # Compute and store the constant term (expectation of selected terms)
            x_diag[1] = -dot(μψ, x_diag[2:n_diag+1]) -dot(μψ_off, tmp_off)

            # Make sure that active dim are in the right order when we affect coefficient.
            # For the split and kfold compute the training and validation losses.

            for (j, offdimj) in enumerate(offdims)
                copy!(view(x_off, (offdimj-1)*(p+1)+1:offdimj*(p+1)), tmp_off[(j-1)*(p+1)+1:j*(p+1)])
            end

            modify_a!(C, vcat(x_off, x_diag))

            filter!(x-> x!= new_dim, candidates)
        end
    end
    return C
end

function gradient_off!(dJ::AbstractVector{Float64}, cache::AbstractVector{Float64}, ψ_off::AbstractMatrix{Float64}, x_off, rhs, Ne::Int64)
    fill!(dJ, 0.0)
    cache .= ψ_off'*x_off - rhs
    dJ .= (1/Ne)*ψ_off*cache
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
