export ParametricFunction, evaluate_offdiagbasis,
       evaluate_diagbasis, grad_xd_diagbasis,
       hess_xd_diagbasis,
       evaluate, grad_xd, hess_xd


struct ParametricFunction{m, Nψ, Nx}
    f::ExpandedFunction{m, Nψ, Nx}
end


function evaluate_offdiagbasis(fp::ParametricFunction{m, Nψ, Nx}, ens::EnsembleState{Nx, Ne}) where {m, Nψ, Nx, Ne}
    return evaluate_basis(fp.f, ens, collect(1:Nx-1))
end

function evaluate_offdiagbasis(fp::ParametricFunction{m, Nψ, Nx}, ens::EnsembleState{Nx, Ne}, idx::Array{Int64,2}) where {m, Nψ, Nx, Ne}
    return evaluate_basis(fp.f, ens, collect(1:Nx-1), idx)
end

function evaluate_diagbasis(fp::ParametricFunction{m, Nψ, Nx}, ens::EnsembleState{Nx, Ne}) where {m, Nψ, Nx, Ne}
    return evaluate_basis(fp.f, ens, [Nx])
end

function evaluate_diagbasis(fp::ParametricFunction{m, Nψ, Nx}, ens::EnsembleState{Nx, Ne}, idx::Array{Int64,2}) where {m, Nψ, Nx, Ne}
    return evaluate_basis(fp.f, ens, [Nx], idx)
end


function grad_xd_diagbasis(fp::ParametricFunction{m, Nψ, Nx}, ens::EnsembleState{Nx, Ne}) where {m, Nψ, Nx, Ne}
    # Evaluate derivatives of basis of x_{d}
    return grad_xk_basis(fp.f, Nx, 1, ens)
end

function grad_xd_diagbasis(fp::ParametricFunction{m, Nψ, Nx}, ens::EnsembleState{Nx, Ne}, idx::Array{Int64,2}) where {m, Nψ, Nx, Ne}
    # Evaluate derivatives of basis of x_{d}
    return grad_xk_basis(fp.f, Nx, 1, ens, idx)
end

function hess_xd_diagbasis(fp::ParametricFunction{m, Nψ, Nx}, ens::EnsembleState{Nx, Ne}) where {m, Nψ, Nx, Ne}
    # Evaluate 2nd derivatives of basis of x_{d}
    return grad_xk_basis(fp.f, Nx, 2, ens)
end

function hess_xd_diagbasis(fp::ParametricFunction{m, Nψ, Nx}, ens::EnsembleState{Nx, Ne}, idx::Array{Int64,2}) where {m, Nψ, Nx, Ne}
    # Evaluate 2nd derivatives of basis of x_{d}
    return grad_xk_basis(fp.f, Nx, 2, ens, idx)
end


function evaluate(fp::ParametricFunction{m, Nψ, Nx}, ens::EnsembleState{Nx, Ne}) where {m, Nψ, Nx, Ne}
    return (evaluate_offdiagbasis(fp, ens) .* evaluate_diagbasis(fp, ens)) * fp.f.coeff
end

function grad_xd(fp::ParametricFunction{m, Nψ, Nx}, ens::EnsembleState{Nx, Ne}) where {m, Nψ, Nx, Ne}
    ψ = evaluate_offdiagbasis(fp, ens)
    dψxd = grad_xd_diagbasis(fp, ens)
    @show ψ
    @show dψxd
    return (ψ .* dψxd) * fp.f.coeff
end

function hess_xd(fp::ParametricFunction{m, Nψ, Nx}, ens::EnsembleState{Nx, Ne}) where {m, Nψ, Nx, Ne}
    ψ = evaluate_offdiagbasis(fp, ens)
    d2ψxd = hess_xd_diagbasis(fp, ens)
    return ψ .* d2ψxd * fp.f.coeff
end


# function grad_coeff
