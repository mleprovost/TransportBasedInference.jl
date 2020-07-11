export Storage

# Create a structure that will hold evaluation of the basis functions,
# as well as their derivative and second derivative


struct Storage{m, Nψ, Nx, Ne}

    # Expanded function
    f::ExpandedFunction{m, Nψ, Nx}

    # Off-diagonal basis evaluation
    offbasis::Array{Float64, 3}

    # Diagonal basis evaluation
    diagbasis::Array{Float64, 3}

    # Evaluate ∂_xd ψ(x_1,...,x_d)
    Gxdbasis::Array{Float64,1}

    # Evaluate ∂²_xd ψ(x_1,...,x_d)
    Hxdbasis::Array{Float64,1}

    # Evaluate derivative of all variables
    Gxbasis::Array{Float64, 3}
    # Evaluate hessian of all variables
    Hxbasis::Array{Float64, 4}

    ## Tools for integration

    # Evaluate ∂xd f for integration
    fi::Array{Int64,1}
    # Points for integration
    nodes::Array{Float64,1}
    # Weights for integration
    weights::Array{Float64,1}



    # Evaluate basis functions at x = 0
    diagbasis0::Array{Float64,1}
    # Evaluate ψ(x_1,...,x_Nx-1, 0)
    ψ0::Array{Float64,1}
    # Evaluate with derivatives of diagonal basis
    dψxd::Array{Float64,2}
end

# function Storage(f::ExpandedFunction{m, Nψ, Nx, Ne}, ens::EnsembleState{Nx, Ne}) where {m, Nψ, Nx, Ne}
#         # offbasis =
#
#
#
#
#
#     # return Storage{m, Nx, Ne}(B, basis, Gxbasis, Hxbasis, diagbasis0, ψ0, Gxdbasis, Hxdbasis)
#
# end
