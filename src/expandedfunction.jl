

export ExpandedFunction, alleval,
       evaluate_basis, grad_xk_basis,
       grad_x_basis, hess_x_basis,
       evaluate, grad_x, hess_x,
       grad_xd, hess_xd,
       grad_coeff, hess_coeff,
       grad_coeff_grad_xd,
       hess_coeff_grad_xd

# ExpandedFunction decomposes a multi-dimensional function f:Rᴹ → R onto
# a basis of MultiFunctions ψ_α where c_α are scalar coefficients
# for each MultiFunction:
# f(x1, x2, ..., xNx) = ∑_α c_α ψ_α(x1, x2, ..., xNx)
# Nψ is the number of MultiFunctions used,
# Nx is the dimension of the input vector x

struct ExpandedFunction{m, Nψ, Nx}
    B::MultiBasis{m, Nx}
    idx::Array{Int64,2}
    coeff::Array{Float64,1}
    function ExpandedFunction(B::MultiBasis{m, Nx}, idx::Array{Int64,2}, coeff::Array{Float64,1}) where {m, Nx}
            @assert size(idx,1) == size(coeff, 1) "The dimension of the basis functions don't
                                            match the number of coefficients"
            Nψ = size(idx,1)

            @assert size(idx,2) == Nx "Size of the array of multi-indices idx is wrong"
        return new{m, Nψ, Nx}(B, idx, coeff)
    end
end



# This code is not optimized for speed
function (f::ExpandedFunction{m, Nψ, Nx})(x::Array{T,1}) where {m, Nψ, Nx, T<:Real}
    out = 0.0
    for i=1:Nψ
        fi = MultiFunction(f.B, f.idx[i,:])
        out += f.coeff[i]*fi(x)
    end
    return out
end



# alleval computes the evaluation, graidnet of hessian of the function
# use it for validatio since it is slower than the other array-based variants
function alleval(f::ExpandedFunction{m, Nψ, Nx}, ens::EnsembleState{Nx, Ne}) where {m, Nψ, Nx, Ne}
        ψ = zeros(Ne, Nψ)
       dψ = zeros(Ne, Nψ, Nx)
      d2ψ = zeros(Ne, Nψ, Nx, Nx)
   result = DiffResults.HessianResult(zeros(Nx))

    for i=1:Nψ
        fi = MultiFunction(f.B, f.idx[i,:])
        for j=1:Ne
            result = ForwardDiff.hessian!(result, fi, member(ens,j))
            ψ[j,i] = DiffResults.value(result)
            dψ[j,i,:,:] .= DiffResults.gradient(result)
            d2ψ[j,i,:,:] .= DiffResults.hessian(result)
        end
    end
    return ψ, dψ, d2ψ
end



function evaluate_basis(f::ExpandedFunction{m, Nψ, Nx}, ens::EnsembleState{Nx, Ne}) where {m, Nψ, Nx, Ne}
    # Compute products of basis functions of the base polynomials
    # with respect to each dimension
    ψ = ones(Ne, Nψ)
    for j=1:Nx
        midxj = f.idx[:,j]

        maxj = maximum(midxj)
        ψj = vander(f.B.B, maxj, 0, ens.S[j,:])
        ψ .*= ψj[:, midxj .+ 1]
    end
    return ψ
end

function evaluate_basis(f::ExpandedFunction{m, Nψ, Nx}, ens::EnsembleState{Nx, Ne}, dims::Array{Int64,1}) where {m, Nψ, Nx, Ne}
    # Compute products of basis functions of the base polynomials
    # with respect to each dimension

    ψ = ones(Ne, Nψ)
    for j in dims
        midxj = idx[:,j]

        maxj = maximum(midxj)
        ψj = vander(f.B.B, maxj, 0, ens.S[j,:])
        ψ .*= ψj[:, midxj .+ 1]
    end
    return ψ
end

function evaluate_basis(f::ExpandedFunction{m, Nψ, Nx}, ens::EnsembleState{Nx, Ne}, idx::Array{Int64,2}) where {m, Nψ, Nx, Ne}
    # Compute products of basis functions of the base polynomials
    # with respect to each dimension
    Nψreduced = size(idx,1)
    ψ = ones(Ne, Nψreduced)
    for j=1:Nx
        midxj = idx[:,j]

        maxj = maximum(midxj)
        ψj = vander(f.B.B, maxj, 0, ens.S[j,:])
        ψ .*= ψj[:, midxj .+ 1]
    end
    return ψ
end

function evaluate_basis(f::ExpandedFunction{m, Nψ, Nx}, ens::EnsembleState{Nx, Ne}, dims::Array{Int64,1}, idx::Array{Int64,2}) where {m, Nψ, Nx, Ne}
    # Compute products of basis functions of the base polynomials
    # with respect to each dimension
    Nψreduced = size(idx,1)
    ψ = ones(Ne, Nψreduced)
    for j in dims
        midxj = idx[:,j]

        maxj = maximum(midxj)
        ψj = vander(f.B.B, maxj, 0, ens.S[j,:])
        ψ .*= ψj[:, midxj .+ 1]
    end
    return ψ
end

function grad_xk_basis(f::ExpandedFunction{m, Nψ, Nx}, grad_dim::Union{Int64, Array{Int64,1}}, k::Int64, ens::EnsembleState{Nx, Ne}) where {m, Nψ, Nx, Ne}
    # Compute the k=th order deriviative of an expanded function along the direction grad_dim

    # ∂ᵏf/∂x_{grad_dim} = ψ
    @assert k>=0  "The derivative order k must be >=0"

    T = typeof(grad_dim)
    if T <:Array{Int64,1}
        @assert all(1 .<= grad_dim)
        @assert all(grad_dim .<= Nx)
    elseif T <: Int64
        @assert all(1 <= grad_dim)
        @assert all(grad_dim <= Nx)
    end

    dkψ = ones(Ne, Nψ)
    for j=1:Nx
        midxj = f.idx[:,j]
        maxj = maximum(midxj)
        if j in grad_dim # Compute the kth derivative along grad_dim
            dkψj = vander(f.B.B, maxj, k, ens.S[j,:])

        else # Simple evaluation
            dkψj = vander(f.B.B, maxj, 0, ens.S[j,:])
        end
        dkψ .*= dkψj[:, midxj .+ 1]
    end
    return dkψ
end

function grad_xk_basis(f::ExpandedFunction{m, Nψ, Nx}, grad_dim::Union{Int64, Array{Int64,1}}, k::Int64, ens::EnsembleState{Nx, Ne}, idx::Array{Int64,2}) where {m, Nψ, Nx, Ne}
    # Compute the k=th order deriviative of an expanded function along the direction grad_dim

    # ∂ᵏf/∂x_{grad_dim} = ψ
    @assert k>=0  "The derivative order k must be >=0"

    T = typeof(grad_dim)
    if T <:Array{Int64,1}
        @assert all(1 .<= grad_dim)
        @assert all(grad_dim .<= Nx)
    elseif T <: Int64
        @assert all(1 <= grad_dim)
        @assert all(grad_dim <= Nx)
    end
    Nψreduced = size(idx,1)
    dkψ = ones(Ne, Nψreduced)
    for j=1:Nx
        midxj = idx[:,j]
        maxj = maximum(midxj)
        if j in grad_dim # Compute the kth derivative along grad_dim
            dkψj = vander(f.B.B, maxj, k, ens.S[j,:])

        else # Simple evaluation
            dkψj = vander(f.B.B, maxj, 0, ens.S[j,:])
        end
        dkψ .*= dkψj[:, midxj .+ 1]
    end
    return dkψ
end


function grad_x_basis(f::ExpandedFunction{m, Nψ, Nx}, ens::EnsembleState{Nx, Ne}) where {m, Nψ, Nx, Ne}
    # Compute the k=th order deriviative of an expanded function along the direction grad_dim
    dψ = ones(Ne, Nψ, Nx)
    @inbounds for j=1:Nx
            dψ[:,:,j] .= grad_xk_basis(f, j, 1, ens)
        end
    return dψ
end

function grad_x_basis(f::ExpandedFunction{m, Nψ, Nx}, ens::EnsembleState{Nx, Ne}, idx::Array{Int64,2}) where {m, Nψ, Nx, Ne}
    # Compute the k=th order deriviative of an expanded function along the direction grad_dim
    Nψreduced = size(idx,1)
    dψ = ones(Ne, Nψreduced, Nx)
    @inbounds for j=1:Nx
            dψ[:,:,j] .= grad_xk_basis(f, j, 1, ens, idx)
        end
    return dψ
end

function hess_x_basis(f::ExpandedFunction{m, Nψ, Nx}, ens::EnsembleState{Nx, Ne}) where {m, Nψ, Nx, Ne}
    # Compute the k=th order deriviative of an expanded function along the direction grad_dim
    d2ψ = ones(Ne, Nψ, Nx, Nx)

    # Fill diagonal components
    @inbounds for i=1:Nx
        d2ψ[:,:,i,i] .= grad_xk_basis(f, i, 2, ens)
    end

    # Fill off-diagonal and exploit symmetry (Schwartz theorem)
    @inbounds for i=1:Nx
                for j=i+1:Nx
                    d2ψ[:,:,i,j] .= grad_xk_basis(f, [i;j], 1, ens)
                    d2ψ[:,:,j,i] .= d2ψ[:,:,i,j]
                end
    end
    return d2ψ
end

function hess_x_basis(f::ExpandedFunction{m, Nψ, Nx}, ens::EnsembleState{Nx, Ne}, idx::Array{Int64,2}) where {m, Nψ, Nx, Ne}
    # Compute the k=th order deriviative of an expanded function along the direction grad_dim
    Nψreduced = size(idx,1)
    d2ψ = ones(Ne, Nψreduced, Nx, Nx)

    # Fill diagonal components
    @inbounds for i=1:Nx
        d2ψ[:,:,i,i] .= grad_xk_basis(f, i, 2, ens, idx)
    end

    # Fill off-diagonal and exploit symmetry (Schwartz theorem)
    @inbounds for i=1:Nx
                for j=i+1:Nx
                    d2ψ[:,:,i,j] .= grad_xk_basis(f, [i;j], 1, ens, idx)
                    d2ψ[:,:,j,i] .= d2ψ[:,:,i,j]
                end
    end
    return d2ψ
end


function evaluate(f::ExpandedFunction{m, Nψ, Nx}, ens::EnsembleState{Nx, Ne}) where {m, Nψ, Nx, Ne}
    return evaluate_basis(f, ens)*f.coeff
end

function grad_x(f::ExpandedFunction{m, Nψ, Nx}, ens::EnsembleState{Nx, Ne}) where {m, Nψ, Nx, Ne}
    dψ = zeros(Ne, Nx)
    dψ_basis = grad_x_basis(f, ens)
    @tensor dψ[a,b] = dψ_basis[a,c,b] * f.coeff[c]
    return dψ
end

function hess_x(f::ExpandedFunction{m, Nψ, Nx}, ens::EnsembleState{Nx, Ne}) where {m, Nψ, Nx, Ne}
    d2ψ = zeros(Ne, Nx, Nx)
    d2ψ_basis = hess_x_basis(f, ens)

    @tensor d2ψ[a,b,c] = d2ψ_basis[a,d,b,c] * f.coeff[d]
    return d2ψ
end

function grad_xd(f::ExpandedFunction{m, Nψ, Nx}, ens::EnsembleState{Nx, Ne}) where {m, Nψ, Nx, Ne}
    dψxd = grad_xk_basis(f, Nx, 1, ens)
    return dψxd*f.coeff
end

function hess_xd(f::ExpandedFunction{m, Nψ, Nx}, ens::EnsembleState{Nx, Ne}) where {m, Nψ, Nx, Ne}
    d2ψxd = grad_xk_basis(f, Nx, 2, ens)
    return d2ψxd*f.coeff
end


# Derivative with respect to the some coefficients
function grad_coeff(f::ExpandedFunction{m, Nψ, Nx}, ens::EnsembleState{Nx, Ne}, coeff_idx::Array{Int64, 1}) where {m, Nψ, Nx, Ne}
    # Verify that all the index
    @assert all([0 <=idx <= Nψ for idx in coeff_idx]) "idx is larger than Nψ"
    return evaluate_basis(f, ens, f.idx[coeff_idx,:])
end


# Derivative with respect to the coefficients
function grad_coeff(f::ExpandedFunction{m, Nψ, Nx}, ens::EnsembleState{Nx, Ne}) where {m, Nψ, Nx, Ne}
    return evaluate_basis(f, ens)
end


# Hessian with respect to the some coefficients
function hess_coeff(f::ExpandedFunction{m, Nψ, Nx}, ens::EnsembleState{Nx, Ne}, coeff_idx::Array{Int64, 1}) where {m, Nψ, Nx, Ne}
    # Verify that all the index
    Nψreduced = size(coeff_idx,1)
    @assert all([0 <=idx <= Nψ for idx in coeff_idx]) "idx is larger than Nψ"
    return zeros(Ne, Nψreduced, Nψreduced)
end


# Hessian with respect to the coefficients
function hess_coeff(f::ExpandedFunction{m, Nψ, Nx}, ens::EnsembleState{Nx, Ne}) where {m, Nψ, Nx, Ne}
    return zeros(Ne, Nψ, Nψ)
end


# Gradient with respect to the coefficients of the gradient with respect to xd

function grad_coeff_grad_xd(f::ExpandedFunction{m, Nψ, Nx}, ens::EnsembleState{Nx, Ne}) where {m, Nψ, Nx, Ne}
    return grad_xk_basis(f, Nx, 1, ens)
end
# function grad_xk_basis(f::ExpandedFunction{m, Nψ, Nx}, grad_dim::Union{Int64, Array{Int64,1}}, k::Int64, ens::EnsembleState{Nx, Ne}, idx::Array{Int64,2}) where {m, Nψ, Nx, Ne}

function grad_coeff_grad_xd(f::ExpandedFunction{m, Nψ, Nx}, ens::EnsembleState{Nx, Ne},coeff_idx::Array{Int64,1}) where {m, Nψ, Nx, Ne}
    return grad_xk_basis(f, Nx, 1, ens, f.idx[coeff_idx,:])
end

# Hessian with respect to the coefficients of the gradient with respect to xd

function hess_coeff_grad_xd(f::ExpandedFunction{m, Nψ, Nx}, ens::EnsembleState{Nx, Ne}) where {m, Nψ, Nx, Ne}
    return zeros(Ne, Nψ, Nψ)
end

function hess_coeff_grad_xd(f::ExpandedFunction{m, Nψ, Nx}, ens::EnsembleState{Nx, Ne},coeff_idx::Array{Int64,1}) where {m, Nψ, Nx, Ne}
    # Verify that all the index
    Nψreduced = size(coeff_idx,1)
    @assert all([0 <=idx <= Nψ for idx in coeff_idx]) "idx is larger than Nψ"
    return zeros(Ne, Nψreduced, Nψreduced)
end
