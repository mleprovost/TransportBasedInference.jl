export greedyfit, gradient_off!, update_component

function greedyfit(Nx, p::Int64, X, maxterms::Int64, λ, δ, γ)

    NxX, Ne = size(X)
    Xsort = deepcopy(sort(X; dims = 2))
    @assert p > -1 "Treat this particular case later"

    @assert NxX == Nx "Wrong dimension of the ensemble matrix `X`"
    @assert Nx > 1 "Treat this particular case later"
    # Initialize a sparse radial map component C with only a diagonal term of order p
    order = -1*ones(Int64, Nx)
    order[end] = p
    C = SparseRadialMapComponent(Nx, order)

    center_std!(C, X; γ = γ)
    xdiag = optimize(C, X, λ, δ)
    modify_a(xdiag, C)

    # Create a radial map with order p for all the entries
    Cfull = SparseRadialMapComponent(Nx, p)

    # Compute centers and widths
    center_std!(Cfull, X)

    ### Evaluate the different basis

    # Create weights
    ψ_off, ψ_mono, dψ_mono = compute_weights(Cfull, X)

    candidate = collect(1:Nx-1)

    # Compute the gradient of the different basis
    dJ = zeros((p+1)*(Nx-1))

    x_off = zeros((p+1)*(Nx-1))

    # maxfmaily is the maximal number of
    if p == 0
        maxfamily = ceil(Int64, (sqrt(Ne)-(p+1))/(p+1))
    elseif p > 0
        maxfamily = ceil(Int64, (sqrt(Ne)-(p+3))/(p+1))
    else
        error("Wrong value for p")
    end

    budget = min(maxfamily, Nx-1)
    count = 0

    for i=1:budget
        # Compute the gradient of the different basis
        gradient_off!(dJ, ψ_off, ψ_diag, x_off, x_diag, λ, δ)

        _, new_dim = findmax(map(i-> norm(view(dJ, (i-1)*(p+1)+1:i*(p+1))), candidate))
        # Update storage in C
        update_component(C, p, new_dim)

        # Compute center and std for this new family of features
        # center_std_off! expects an ensemble matrix X sorted by dimensions
        center_std_off!(C, Xsort, γ, new_dim)

        # Then update qr, then do change of variables
        F = 1.0

    end



end

function gradient_off!(dJ::AbstractVector{Float64}, ψ_off::AbstractMatrix{Float64}, ψ_diag::AbstractMatrix{Float64}, x_off, x_diag, λ::Float64, δ::Float64)





end


function update_component(C::SparseRadialMapComponent, p::Int64, new_dim::Int64)
    @assert C.Nx >= newdim
    if newdim == C.Nx
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
            C.ξ[new_dim] = zeros(p+1)
            C.σ[new_dim] = zeros(p+1)
            C.a[new_dim] = zeros(p+2)
        end
    end
end
