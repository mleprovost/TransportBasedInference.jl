export Storage

# Create a structure that will hold evaluation of the basis functions,
# as well as their derivative and second derivative


struct Storage{m, Nx, Ne}
    B::MultiBasis{m, Nx}

    # Evaluate all basis functions for all variables
    basis::Array{Float64, 3}

    # Evaluate derivative of all variables
    Gxbasis::Array{Float64, 3}

    # Evaluate hessian of all variables
    Hxbasis::Array{Float64, 3}

    # Evaluate basis functions at x = 0
    diagbasis0::Array{Float64,1}

    # Evaluate ψ(x_1,...,x_Nx-1, 0)
    ψ0::Array{Float64,1}

    # Evaluate ∂_xd ψ(x_1,...,x_d)
    Gxdbasis::Array{Float64,1}

    # Evaluate ∂²_xd ψ(x_1,...,x_d)
    Hxdbasis::Array{Float64,1}

    # Integration
end

function Storage(B::MultiBasis{m, Nx}, ens::EnsembleState{Nx, Ne}) where {m, Nx, Ne}
        basis = zeros(m, Nx, Ne)

        for i=1:m
            for j=1:Nx
                for k=1:Ne
                    basis[:,j,k] = vander(B.B, 0)
                end
            end
        end






    # return Storage{m, Nx, Ne}(B, basis, Gxbasis, Hxbasis, diagbasis0, ψ0, Gxdbasis, Hxdbasis)

end
