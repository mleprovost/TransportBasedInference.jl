export ParametricFunction, evaluate


struct ParametricFunction{m, Nψ, Nx}
    f::ExpandedFunction{m, Nψ, Nx}
end


function evaluate_offdiagbasis(fp::ParametricFunction{m, Nψ, Nx}, ens::EnsembleState{Nx, Ne}) where {m, Nψ, Nx, Ne}
    return evaluate_basis(fp.f, ens, collect(1:Nx-1))
end

function evaluate_offdiagbasis(fp::ParametricFunction{m, Nψ, Nx}, ens::EnsembleState{Nx, Ne}, idx::Array{Int64,2}) where {m, Nψ, Nx, Ne}
    return evaluate_basis(fp.f, ens, collect(1:Nx-1), idx)
end


# evaluate(fp::ParametricFunction{m, Nψ, Ne}, ens::EnsembleState{Nx, Ne}) where {m, Nψ, Nx, Ne}
# end
