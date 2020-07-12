export ParametricFunction, evaluate_offdiagbasis,
       evaluate_diagbasis, grad_xd_diagbasis,
       hess_xd_diagbasis,
       evaluate, grad_xd, hess_xd,
       grad_coeff, grad_coeff_grad_xd


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
    return grad_xk_basis(fp.f, Nx, 1, Nx, ens)
end

function grad_xd_diagbasis(fp::ParametricFunction{m, Nψ, Nx}, ens::EnsembleState{Nx, Ne}, idx::Array{Int64,2}) where {m, Nψ, Nx, Ne}
    # Evaluate derivatives of basis of x_{d}
    return grad_xk_basis(fp.f, Nx, 1, Nx, ens, idx)
end

function hess_xd_diagbasis(fp::ParametricFunction{m, Nψ, Nx}, ens::EnsembleState{Nx, Ne}) where {m, Nψ, Nx, Ne}
    # Evaluate 2nd derivatives of basis of x_{d}
    return grad_xk_basis(fp.f, Nx, 2, Nx, ens)
end

function hess_xd_diagbasis(fp::ParametricFunction{m, Nψ, Nx}, ens::EnsembleState{Nx, Ne}, idx::Array{Int64,2}) where {m, Nψ, Nx, Ne}
    # Evaluate 2nd derivatives of basis of x_{d}
    return grad_xk_basis(fp.f, Nx, 2, Nx, ens, idx)
end


function evaluate(fp::ParametricFunction{m, Nψ, Nx}, ens::EnsembleState{Nx, Ne}) where {m, Nψ, Nx, Ne}
    return (evaluate_offdiagbasis(fp, ens) .* evaluate_diagbasis(fp, ens)) * fp.f.coeff
end

function grad_xd(fp::ParametricFunction{m, Nψ, Nx}, ens::EnsembleState{Nx, Ne}) where {m, Nψ, Nx, Ne}
    ψ = evaluate_offdiagbasis(fp, ens)
    dψxd = grad_xd_diagbasis(fp, ens)
    return (ψ .* dψxd) * fp.f.coeff
end

function hess_xd(fp::ParametricFunction{m, Nψ, Nx}, ens::EnsembleState{Nx, Ne}) where {m, Nψ, Nx, Ne}
    ψ = evaluate_offdiagbasis(fp, ens)
    d2ψxd = hess_xd_diagbasis(fp, ens)
    return ψ .* d2ψxd * fp.f.coeff
end


function grad_coeff(fp::ParametricFunction{m, Nψ, Nx}, ens::EnsembleState{Nx, Ne}) where {m, Nψ, Nx, Ne}
    ψoff = evaluate_offdiagbasis(fp, ens)
    ψdiag = evaluate_diagbasis(fp, ens)
    return ψoff .* ψdiag
end

function grad_coeff(fp::ParametricFunction{m, Nψ, Nx}, ens::EnsembleState{Nx, Ne}, idx::Array{Int64,2}) where {m, Nψ, Nx, Ne}
    ψoff = evaluate_offdiagbasis(fp, ens, idx)
    ψdiag = evaluate_diagbasis(fp, ens, idx)
    return ψoff .* ψdiag
end

function grad_coeff_grad_xd(fp::ParametricFunction{m, Nψ, Nx}, ens::EnsembleState{Nx, Ne}) where {m, Nψ, Nx, Ne}
    ψoff = evaluate_offdiagbasis(fp, ens)
    dψxd = grad_xd_diagbasis(fp, ens)
    return ψoff .* dψxd
end

function grad_coeff_grad_xd(fp::ParametricFunction{m, Nψ, Nx}, ens::EnsembleState{Nx, Ne}, idx::Array{Int64, 2}) where {m, Nψ, Nx, Ne}
    ψoff = evaluate_offdiagbasis(fp, ens, idx)
    dψxd = grad_xd_diagbasis(fp, ens, idx)
    return ψoff .* dψxd
end
