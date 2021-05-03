export greedyfit

function greedyfit(Nx, p::Int64, X, maxterms::Union{Int64, Nothing}, λ, δ)

    NxX, Ne = size(X)
    @assert NxX == Nx "Wrong dimension of the ensemble matrix `X`"
    # Initialize a sparse radial map component C with only a diagonal term of order p
    order = -1*ones(Int64, Nx)
    order[end] = p
    C = SparseRadialMapComponent(Nx, order)

    optimize(C, X, λ, δ)

    # Create a radial map with order p for all the entries

    Cfull = RadialMapComponent(Nx, p)

    # Compute centers and widths
    center_std(Cfull, X)

    ### Evaluate the different basis

    # Create weights
    Wfull = create_weights(Cfull, X)

    # Compute weights
    compute_weights(Cfull, X, Wfull)

    active_dim = Int64[]
    # push!(active_dim, Nx)

    # Compute the gradient of the different basis














end
