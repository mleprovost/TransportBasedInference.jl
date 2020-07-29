using LoopVectorization: @avx

export  ExpandedFunction, alleval,
        evaluate_basis!,
        evaluate_basis,
        repeated_evaluate_basis,
        grad_xk_basis!,
        grad_xk_basis,
        grad_x_basis!,
        grad_x_basis,
        hess_x_basis!,
        hess_x_basis,
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

struct ExpandedFunction
    m::Int64
    Nψ::Int64
    Nx::Int64
    B::MultiBasis
    idx::Array{Int64,2}
    coeff::Array{Float64,1}
    function ExpandedFunction(B::MultiBasis, idx::Array{Int64,2}, coeff::Array{Float64,1})
            Nψ = size(idx,1)
            Nx = B.Nx
            @assert Nψ == size(coeff, 1) "The dimension of the basis functions don't
                                            match the number of coefficients"


            @assert size(idx,2) == Nx "Size of the array of multi-indices idx is wrong"
        return new(B.B.m, Nψ, Nx, B, idx, coeff)
    end
end



# This code is not optimized for speed
function (f::ExpandedFunction)(x::Array{T,1}) where {T<:Real}
    out = 0.0
    @inbounds for i=1:f.Nψ
        fi = MultiFunction(f.B, f.idx[i,:])
        out += f.coeff[i]*fi(x)
    end
    return out
end



# alleval computes the evaluation, graidnet of hessian of the function
# use it for validatio since it is slower than the other array-based variants
function alleval(f::ExpandedFunction, ens::EnsembleState{Nx, Ne}) where {Nx, Ne}
        Nψ = f.Nψ
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


function evaluate_basis!(ψ, f::ExpandedFunction, X, dims::Union{Array{Int64,1},UnitRange{Int64}}, idx::Array{Int64,2})
    Nψreduced = size(idx,1)
    NxX, Ne = size(X)
    @assert NxX == f.Nx "Wrong dimension of the input sample X"
    @assert size(ψ) == (Ne, Nψreduced) "Wrong dimension of the ψ"

    # maxdim = maximum(f.idx)
    # ψvander = zeros(Ne, maxdim)
    fill!(ψ, 1.0)
    ψtmp = zero(ψ)

    # The maximal size of ψtmp assumes that the set of index is downward closed
    # such that Nψreduced is always smaller of equal to maxj+1
    @inbounds for j in dims
        midxj = view(idx,:,j)
        maxj = maximum(midxj)
        Xj = view(X,j,:)
        ψj = ψtmp[:,1:maxj+1]

        vander!(ψj, f.B.B, maxj, 0, Xj)

        @avx for l = 1:Nψreduced
            for k=1:Ne
                ψ[k,l] *= ψj[k, midxj[l] + 1]
            end
        end

        # @avx  ψ .*= view(ψj,:, midxj .+ 1)#view(ψvanderj, :, midxj .+ 1)#
    end
    return ψ
end

# In-place versions
evaluate_basis!(ψ, f::ExpandedFunction, X, dims::Union{Array{Int64,1},UnitRange{Int64}}) =
              evaluate_basis!(ψ, f, X, dims, f.idx)

evaluate_basis!(ψ, f::ExpandedFunction, X, idx::Array{Int64,2}) =
            evaluate_basis!(ψ, f, X, 1:f.Nx, idx)

evaluate_basis!(ψ, f::ExpandedFunction, X) =
            evaluate_basis!(ψ, f, X, 1:f.Nx, f.idx)


# Versions with allocations
evaluate_basis(f::ExpandedFunction, X::Array{Float64,2}, dims::Union{Array{Int64,1},UnitRange{Int64}}, idx::Array{Int64,2}) =
              evaluate_basis!(zeros(size(X,1),size(idx,1)), f, X, dims, idx)

evaluate_basis(f::ExpandedFunction, X::Array{Float64,2}, dims::Union{Array{Int64,1},UnitRange{Int64}}) =
              evaluate_basis!(zeros(size(X,1),size(f.idx,1)), f, X, dims, f.idx)

evaluate_basis(f::ExpandedFunction, X::Array{Float64,2}, idx::Array{Int64,2}) =
            evaluate_basis!(zeros(size(X,2),size(idx,1)), f, X, 1:f.Nx, idx)

evaluate_basis(f::ExpandedFunction, X::Array{Float64,2}) =
            evaluate_basis!(zeros(size(X,2),size(f.idx,1)), f, X, 1:f.Nx, f.idx)

function repeated_evaluate_basis(f::ExpandedFunction, x, idx::Array{Int64,2})
    # Compute the last component
    midxj = idx[:,f.Nx]
    maxj = maximum(midxj)
    ψj = vander(f.B.B, maxj, 0, x)
    return ψj[:, midxj .+ 1]
end

repeated_evaluate_basis(f::ExpandedFunction, x) = repeated_evaluate_basis(f, x, f.idx)


function grad_xk_basis!(dkψ, f::ExpandedFunction, X::Array{Float64,2}, k::Int64, grad_dim::Union{Int64, Array{Int64,1}}, dims::Union{Int64, UnitRange{Int64}, Array{Int64,1}}, idx::Array{Int64,2})
    m = f.m
    Nx = f.Nx

    NxX, Ne = size(X)
    Nψreduced = size(idx,1)

    @assert NxX == Nx "Wrong dimension of the input sample X"
    @assert size(dkψ) == (Ne, Nψreduced) "Wrong dimension of ψ"
    # ∂ᵏf/∂x_{grad_dim} = ψ
    @assert k>=0  "The derivative order k must be >=0"

    T = typeof(grad_dim)
    if T <:Array{Int64,1}
        @assert all(1 .<= grad_dim)
        @assert all(grad_dim .<= Nx)
    elseif T <: Int64
        @assert 1 <= grad_dim <= Nx
    end

    fill!(dkψ, 1.0)

    for j in dims
        midxj = idx[:,j]
        maxj = maximum(midxj)
        Xj = view(X,j,:)
        if j in grad_dim # Compute the kth derivative along grad_dim
            dkψj = vander(f.B.B, maxj, k, Xj)

        else # Simple evaluation
            dkψj = vander(f.B.B, maxj, 0, Xj)
        end
        dkψ .*= dkψj[:, midxj .+ 1]
    end
    return dkψ
end

# In-place versions of grad_xk_basis!

grad_xk_basis!(dkψ, f::ExpandedFunction, X::Array{Float64,2}, k::Int64, grad_dim::Union{Int64, Array{Int64,1}}, dims::Union{Int64, UnitRange{Int64}, Array{Int64,1}}) =
              grad_xk_basis!(dkψ, f, X, k, grad_dim, dims, f.idx)

grad_xk_basis!(dkψ, f::ExpandedFunction, X::Array{Float64,2}, k::Int64, grad_dim::Union{Int64, Array{Int64,1}}, idx::Array{Int64,2}) =
              grad_xk_basis!(dkψ, f, X, k, grad_dim, 1:f.Nx, idx)

grad_xk_basis!(dkψ, f::ExpandedFunction, X::Array{Float64,2}, k::Int64, grad_dim::Union{Int64, Array{Int64,1}}) =
              grad_xk_basis!(dkψ, f, X, k, grad_dim, 1:f.Nx, f.idx)


# Versions with allocations
grad_xk_basis(f::ExpandedFunction, X::Array{Float64,2}, k::Int64, grad_dim::Union{Int64, Array{Int64,1}}, dims::Union{Int64, UnitRange{Int64}, Array{Int64,1}}, idx::Array{Int64,2}) =
             grad_xk_basis!(zeros(size(X,2), size(idx,1)), f, X, k, grad_dim, dims, f.idx)

grad_xk_basis(f::ExpandedFunction, X::Array{Float64,2}, k::Int64, grad_dim::Union{Int64, Array{Int64,1}}, dims::Union{Int64, UnitRange{Int64}, Array{Int64,1}}) =
             grad_xk_basis!(zeros(size(X,2), size(f.idx,1)), f, X, k, grad_dim, dims, f.idx)

grad_xk_basis(f::ExpandedFunction, X::Array{Float64,2}, k::Int64, grad_dim::Union{Int64, Array{Int64,1}}, idx::Array{Int64,2}) =
             grad_xk_basis!(zeros(size(X,2), size(idx,1)), f, X, k, grad_dim, 1:f.Nx, idx)

grad_xk_basis(f::ExpandedFunction, X::Array{Float64,2}, k::Int64, grad_dim::Union{Int64, Array{Int64,1}}) =
             grad_xk_basis!(zeros(size(X,2), size(f.idx,1)), f, X, k, grad_dim, 1:f.Nx, f.idx)




function grad_x_basis!(dψ::Array{Float64,3}, f::ExpandedFunction, X::Array{Float64,2}, idx::Array{Int64,2})
    m = f.m
    Nx = f.Nx
    # Compute the k-th order deriviative of an expanded function along the direction grad_dim
    Nψreduced = size(idx,1)
    fill!(dψ, 1)
    Ne, Nψr1, Nxψ = size(dψ)
    @assert Nψr1 == size(idx,1) "Wrong dimension of dψ"

    @inbounds for j=1:Nx
        dψj = view(dψ,:,:,j)
        grad_xk_basis!(dψj, f, X, 1, j, idx)
    end
    return dψ
end


# In place version
grad_x_basis!(dψ::Array{Float64,3}, f::ExpandedFunction, X::Array{Float64,2}) = grad_x_basis!(dψ, f, X, f.idx)


# Version with allocations
grad_x_basis(f::ExpandedFunction, X::Array{Float64,2}, idx::Array{Int64,2}) = grad_x_basis!(zeros(size(X,2), size(idx,1), f.Nx), f, X, idx)

grad_x_basis(f::ExpandedFunction, X::Array{Float64,2}) = grad_x_basis!(zeros(size(X,2), size(f.idx,1), f.Nx), f, X, f.idx)


function hess_x_basis!(d2ψ::Array{Float64,4}, f::ExpandedFunction, X::Array{Float64,2}, idx::Array{Int64,2})
    m = f.m
    Nx = f.Nx
    # Compute the k=th order deriviative of an expanded function along the direction grad_dim
    Nψreduced = size(idx,1)
    fill!(d2ψ, 1)
    Ne, Nψr1, Nxψ1, Nxψ2 = size(d2ψ)
    @assert Ne == size(X,2)
    @assert Nxψ1 == Nxψ2 "Wrong dimension of d2ψ"
    @assert Nψr1 == size(idx,1) "Wrong dimension of d2ψ"


    # Fill diagonal components
    @inbounds for j=1:Nx
        d2ψj = view(d2ψ,:,:,j,j)
        grad_xk_basis!(d2ψj, f, X, 2, j, idx)
    end

    # Fill off-diagonal and exploit symmetry (Schwartz theorem)
    @inbounds for i=1:Nx
                for j=i+1:Nx
                    d2ψij = view(d2ψ,:,:,i,j)
                    grad_xk_basis!(d2ψij, f, X, 1, [i;j], idx)
                    d2ψ[:,:,j,i] .= d2ψ[:,:,i,j]
                end
    end
    return d2ψ
end

# In place version
hess_x_basis!(d2ψ::Array{Float64,4}, f::ExpandedFunction, X::Array{Float64,2}) = hess_x_basis!(d2ψ, f, X, f.idx)


# Version with allocations
hess_x_basis(f::ExpandedFunction, X::Array{Float64,2}, idx::Array{Int64,2}) = hess_x_basis!(zeros(size(X,2), size(idx,1), f.Nx, f.Nx), f, X, idx)

hess_x_basis(f::ExpandedFunction, X::Array{Float64,2})  = hess_x_basis!(zeros(size(X,2), size(f.idx,1), f.Nx, f.Nx), f, X, f.idx)



function evaluate!(ψ, f::ExpandedFunction, X::Array{Float64,2})
    ψ .= evaluate_basis(f, X)*f.coeff
    return ψ
end

evaluate(f::ExpandedFunction, X::Array{Float64,2}) = evaluate!(zeros(size(X,2)), f, X)

function grad_x(f::ExpandedFunction, X::Array{Float64,2})
    NxX, Ne = size(X)
    @assert NxX == f.Nx "Wrong dimension of the input"
    dψ = zeros(Ne, f.Nx)
    dψ_basis = grad_x_basis(f, X)
    @tensor dψ[a,b] = dψ_basis[a,c,b] * f.coeff[c]
    return dψ
end

function hess_x(f::ExpandedFunction, X::Array{Float64,2})
    Nx = f.Nx
    NxX, Ne = size(X)
    @assert NxX == Nx "Wrong dimension of the input"
    d2ψ = zeros(Ne, Nx, Nx)
    d2ψ_basis = hess_x_basis(f, X)

    @tensor d2ψ[a,b,c] = d2ψ_basis[a,d,b,c] * f.coeff[d]
    return d2ψ
end

function grad_xd(f::ExpandedFunction, X::Array{Float64,2})
    dψxd = grad_xk_basis(f, X, 1, f.Nx)
    return dψxd*f.coeff
end

function hess_xd(f::ExpandedFunction, X::Array{Float64,2})
    d2ψxd = grad_xk_basis(f, X, 2, f.Nx)
    return d2ψxd*f.coeff
end


# Derivative with respect to the some coefficients
function grad_coeff(f::ExpandedFunction, X::Array{Float64,2}, coeff_idx::Array{Int64, 1})
    Nψ = f.Nψ
    # Verify that all the index
    @assert all([0 <=idx <= Nψ for idx in coeff_idx]) "idx is larger than Nψ"
    return evaluate_basis(f, X, f.idx[coeff_idx,:])
end


# Derivative with respect to the coefficients
function grad_coeff(f::ExpandedFunction, X::Array{Float64,2})
    return evaluate_basis(f, X)
end


# Hessian with respect to the some coefficients
function hess_coeff(f::ExpandedFunction, X::Array{Float64,2}, coeff_idx::Array{Int64, 1})
    # Verify that all the index
    Nψ = f.Nψ
    Ne = size(X,2)
    Nψreduced = size(coeff_idx,1)
    @assert all([0 <=idx <= Nψ for idx in coeff_idx]) "idx is larger than Nψ"
    return zeros(Ne, Nψreduced, Nψreduced)
end


# Hessian with respect to the coefficients
function hess_coeff(f::ExpandedFunction, X::Array{Float64,2})
    Nψ = f.Nψ
    return zeros(size(X,2), Nψ, Nψ)
end


# Gradient with respect to the coefficients of the gradient with respect to xd

function grad_coeff_grad_xd(f::ExpandedFunction, X::Array{Float64,2})
    return grad_xk_basis(f, X, 1, f.Nx)
end
# function grad_xk_basis(f::ExpandedFunction{m, Nψ, Nx}, grad_dim::Union{Int64, Array{Int64,1}}, k::Int64, X::EnsembleState{Nx, Ne}, idx::Array{Int64,2}) where {m, Nψ, Nx, Ne}

function grad_coeff_grad_xd(f::ExpandedFunction, X::Array{Float64,2}, coeff_idx::Array{Int64,1})
    return grad_xk_basis(f, X, 1, f.Nx, f.idx[coeff_idx,:])
end

# Hessian with respect to the coefficients of the gradient with respect to xd

function hess_coeff_grad_xd(f::ExpandedFunction, X::Array{Float64,2})
    Ne = size(X,2)
    Nψ = f.Nψ
    return zeros(Ne, Nψ, Nψ)
end

function hess_coeff_grad_xd(f::ExpandedFunction, X::Array{Float64,2}, coeff_idx::Array{Int64,1})
    # Verify that all the index
    Ne = size(X,2)
    Nψ = f.Nψ
    Nψreduced = size(coeff_idx,1)
    @assert all([0 <=idx <= Nψ for idx in coeff_idx]) "idx is larger than Nψ"
    return zeros(Ne, Nψreduced, Nψreduced)
end
