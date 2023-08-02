using LoopVectorization: @avx

export  ExpandedFunction,
        active_dim,
        alleval,
        getbasis,
        evaluate_basis!,
        evaluate_basis,
        repeated_evaluate_basis,
        grad_xk_basis!,
        grad_xk_basis,
        repeated_grad_xk_basis!,
        repeated_grad_xk_basis,
        repeated_hess_xk_basis!,
        repeated_hess_xk_basis,
        grad_x_basis!,
        grad_x_basis,
        hess_x_basis!,
        hess_x_basis,
        evaluate, grad_x, hess_x,
        grad_xd, hess_xd,
        grad_x_grad_xd,
        hess_x_grad_xd,
        reduced_grad_x_grad_xd!,
        reduced_grad_x_grad_xd,
        reduced_hess_x_grad_xd!,
        reduced_hess_x_grad_xd,
        grad_coeff, hess_coeff,
        grad_coeff_grad_xd,
        hess_coeff_grad_xd

"""
$(TYPEDEF)

`ExpandedFunction` decomposes a multi-dimensional function f:R^{Nx} → R onto
a basis of `MultiFunction` ψ_α where c_α are scalar coefficients
for each MultiFunction:
f(x1, x2, ..., xNx) = ∑_α c_α ψ_α(x1, x2, ..., xNx), where
`Nψ` is the number of `MultiFunction`s used, and
`Nx` is the dimension of the input vector `x`.

## Fields
$(TYPEDFIELDS)

"""
struct ExpandedFunction
    m::Int64
    Nψ::Int64
    Nx::Int64
    B::MultiBasis
    idx::Array{Int64,2}
    dim::Array{Int64, 1} # contains the active dimensions, i.e. columns of idx not equal to zeros
    coeff::Array{Float64,1}
    function ExpandedFunction(B::MultiBasis, idx::Array{Int64,2}, coeff::Array{Float64,1})
            Nψ = size(idx,1)
            Nx = B.Nx
            @assert Nψ == size(coeff, 1) "The dimension of the basis functions don't
                                            match the number of coefficients"


            @assert size(idx,2) == Nx "Size of the array of multi-indices idx is wrong"
        return new(B.B.m, Nψ, Nx, B, idx, active_dim(idx, B), coeff)
    end
end

"""
$(TYPEDSIGNATURES)

Returns the kind of basis of the `ExpandedFunction` `f`.
"""
getbasis(f::ExpandedFunction) = getbasis(f.B)


"""
$(TYPEDSIGNATURES)

Returns the number of features of the `ExpandedFunction` `f` to `coeff`.
"""
ncoeff(f::ExpandedFunction) = f.Nψ

"""
$(TYPEDSIGNATURES)

Returns the coefficients of the `ExpandedFunction` `f` to `coeff`.
"""
getcoeff(f::ExpandedFunction)= f.coeff

"""
$(TYPEDSIGNATURES)

Set the coefficients of the `ExpandedFunction` `f` to `coeff`.
"""

function setcoeff!(f::ExpandedFunction, coeff::Array{Float64,1})
        @assert size(coeff,1) == f.Nψ "Wrong dimension of coeff"
        f.coeff .= coeff
end

"""
$(TYPEDSIGNATURES)

Set all the coefficients of the `ExpandedFunction` `f` to zero.
"""
clearcoeff!(f::ExpandedFunction) = fill!(f.coeff, 0.0)

"""
$(TYPEDSIGNATURES)

Returns the multi-indices of the features of the `ExpandedFunction` `f`.
"""
getidx(f::ExpandedFunction) = f.idx


"""
$(TYPEDSIGNATURES)

Returns the active dimensions of the `ExpandedFunction` `f`.
"""
active_dim(f::ExpandedFunction) = f.dim


"""
$(TYPEDSIGNATURES)

Evaluates the `ExpandedFunction` `f` at `x`.
"""
function (f::ExpandedFunction)(x::Array{T,1}) where {T<:Real}
    out = 0.0
    @inbounds for i=1:f.Nψ
        fi = MultiFunction(f.B, f.idx[i,:])
        out += f.coeff[i]*fi(x)
    end
    return out
end

"""
$(TYPEDSIGNATURES)

Returns the active dimensions of the  set of multi-indices `idx` for the `Basis` `B`.
"""
function active_dim(idx::Array{Int64,2}, B::T) where {T<:Basis}
    # Nx should always be an active dimension (we need to ensures
    # that we have a strictly increasing function in the last component)
    dim = Int64[]
    Nx = size(idx,2)
    if iszerofeatureactive(B) == false
        @inbounds for i=1:Nx-1
            if !all(view(idx,:,i) .== 0)
                push!(dim, i)
            end
        end
        push!(dim, Nx)
    else
        dim = collect(1:size(idx,2))
    end

    return dim
end

# alleval computes the evaluation, gradient and hessian of the function
# use it for validatio since it is slower than the other array-based variants
"""
$(TYPEDSIGNATURES)

Returns the evaluation, gradient and hessian of the `ExpandedFunction` `f` for the ensemble matrix `X`.
Note that this routine is not designed for speed, but for validation only,
as all the derivatives are computedto machine precision with `ForwardDiff.jl`
"""
function alleval(f::ExpandedFunction, X)
        Nx, Ne = size(X)
        Nψ = f.Nψ
        ψ = zeros(Ne, Nψ)
       dψ = zeros(Ne, Nψ, Nx)
      d2ψ = zeros(Ne, Nψ, Nx, Nx)
   result = DiffResults.HessianResult(zeros(Nx))

    for i=1:Nψ
        fi = MultiFunction(f.B, f.idx[i,:])
        for j=1:Ne
            result = ForwardDiff.hessian!(result, fi, X[:,j])
            ψ[j,i] = DiffResults.value(result)
            dψ[j,i,:,:] .= DiffResults.gradient(result)
            d2ψ[j,i,:,:] .= DiffResults.hessian(result)
        end
    end
    return ψ, dψ, d2ψ
end

"""
$(TYPEDSIGNATURES)

Evaluates in-place the basis of `ExpandedFunction` `f` for the ensemble matrix `X` along the dimensions `dims` for the set of multi-indices of features `idx`
"""
function evaluate_basis!(ψ, f::ExpandedFunction, X, dims::Union{Array{Int64,1},UnitRange{Int64}}, idx::Array{Int64,2})
    Nψreduced = size(idx,1)
    NxX, Ne = size(X)
    @assert NxX == f.Nx "Wrong dimension of the input sample X"
    @assert size(ψ) == (Ne, Nψreduced) "Wrong dimension of the ψ"

    maxdim = maximum(idx)
    # ψvander = zeros(Ne, maxdim)
    fill!(ψ, 1.0)
    if maxdim+1<= Nψreduced
        ψtmp = zero(ψ)
    else
        ψtmp = zeros(Ne, maxdim+1)
    end

    # The maximal size of ψtmp assumes that the set of index is downward closed
    # such that Nψreduced is always smaller of equal to maxj+1
    @inbounds for j in intersect(dims, f.dim)
        idxj = view(idx,:,j)
        maxj = maximum(idxj)
        Xj = view(X,j,:)
        ψj = ψtmp[:,1:maxj+1]

        vander!(ψj, f.B.B, maxj, 0, Xj)

        @avx for l = 1:Nψreduced
            for k=1:Ne
                ψ[k,l] *= ψj[k, idxj[l] + 1]
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
            evaluate_basis!(ψ, f, X, f.dim, idx)
            # evaluate_basis!(ψ, f, X, 1:f.Nx, idx)

evaluate_basis!(ψ, f::ExpandedFunction, X) =
            evaluate_basis!(ψ, f, X, f.dim, f.idx)
            # evaluate_basis!(ψ, f, X, 1:f.Nx, f.idx)


# Versions with allocations
"""
$(TYPEDSIGNATURES)

Evaluates the basis of `ExpandedFunction` `f` for the ensemble matrix `X` along the dimensions `dims` for the set of multi-indices of features `idx`
"""
evaluate_basis(f::ExpandedFunction, X, dims::Union{Array{Int64,1},UnitRange{Int64}}, idx::Array{Int64,2}) =
              evaluate_basis!(zeros(size(X,2),size(idx,1)), f, X, dims, idx)

evaluate_basis(f::ExpandedFunction, X, dims::Union{Array{Int64,1},UnitRange{Int64}}) =
              evaluate_basis!(zeros(size(X,2),size(f.idx,1)), f, X, dims, f.idx)

evaluate_basis(f::ExpandedFunction, X, idx::Array{Int64,2}) =
            evaluate_basis!(zeros(size(X,2),size(idx,1)), f, X, f.dim, idx)
            # evaluate_basis!(zeros(size(X,2),size(idx,1)), f, X, 1:f.Nx, idx)

evaluate_basis(f::ExpandedFunction, X) =
            evaluate_basis!(zeros(size(X,2),size(f.idx,1)), f, X, f.dim, f.idx)
            # evaluate_basis!(zeros(size(X,2),size(f.idx,1)), f, X, 1:f.Nx, f.idx)

"""
$(TYPEDSIGNATURES)

Evaluates the basis of `ExpandedFunction` `f` for the last component
"""
function repeated_evaluate_basis(f::ExpandedFunction, x, idx::Array{Int64,2})
    # Compute the last component
    midxj = idx[:,f.Nx]
    maxj = maximum(midxj)
    ψj = vander(f.B.B, maxj, 0, x)
    return ψj[:, midxj .+ 1]
end

repeated_evaluate_basis(f::ExpandedFunction, x) = repeated_evaluate_basis(f, x, f.idx)

# function grad_xk_basis!(dkψ, f::ExpandedFunction, X::Array{Float64,2}, k::Int64, grad_dim::Union{Int64, Array{Int64,1}}, dims::Union{Int64, UnitRange{Int64}, Array{Int64,1}}, idx::Array{Int64,2})
"""
$(TYPEDSIGNATURES)

Evaluates in-place the k-th (k>0) derivative of the basis features of `f` with respect to the `grad_dim` components of the states.
The i-th column of the output contains ∂^k...∂^k ∏_{j ∈ dims} ψ^i_{j}(x_1:n) ∂x_{grad_dim[1]}^k ... ∂x_{grad_dim[end]}^k
for the different columns of the ensemble matrix `X`, where ψ^i is the i-th feature.
"""
function grad_xk_basis!(dkψ, f::ExpandedFunction, X, k::Int64, grad_dim::Union{Int64, Array{Int64,1}}, dims::Union{Int64, UnitRange{Int64}, Array{Int64,1}}, idx::Array{Int64,2})
    m = f.m
    Nx = f.Nx

    NxX, Ne = size(X)
    Nψreduced = size(idx,1)

    @assert NxX == Nx "Wrong dimension of the input sample X"
    @assert size(dkψ) == (Ne, Nψreduced) "Wrong dimension of ψ"
    # ∂ᵏf/∂x_{grad_dim} = ψ
    @assert k>0  "The derivative order k must be >0, if k=0, you better use evaluate_basis!"

    T = typeof(grad_dim)
    if T <:Array{Int64,1}
        @assert all(1 .<= grad_dim)
        @assert all(grad_dim .<= Nx)
    elseif T <: Int64
        @assert 1 <= grad_dim <= Nx
    end

    # Check if we are taking derivative with respect
    # to non active dimension
    if any([!(gdim ∈ f.dim) for gdim in grad_dim])
        fill!(dkψ, 0.0)
    else
        fill!(dkψ, 1.0)

        for j in intersect(dims, f.dim)
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
    end
    return dkψ
end

# In-place versions of grad_xk_basis!

grad_xk_basis!(dkψ, f::ExpandedFunction, X, k::Int64, grad_dim::Union{Int64, Array{Int64,1}}, dims::Union{Int64, UnitRange{Int64}, Array{Int64,1}}) =
              grad_xk_basis!(dkψ, f, X, k, grad_dim, dims, f.idx)

grad_xk_basis!(dkψ, f::ExpandedFunction, X, k::Int64, grad_dim::Union{Int64, Array{Int64,1}}, idx::Array{Int64,2}) =
              grad_xk_basis!(dkψ, f, X, k, grad_dim, 1:f.Nx, idx)

grad_xk_basis!(dkψ, f::ExpandedFunction, X, k::Int64, grad_dim::Union{Int64, Array{Int64,1}}) =
              grad_xk_basis!(dkψ, f, X, k, grad_dim, 1:f.Nx, f.idx)


# Versions with allocations
"""
$(TYPEDSIGNATURES)

Evaluates in-place the k-th (k>0) derivative of the basis features of `f` with respect to the `grad_dim` components of the states.
The i-th column of the output contains ∂^k...∂^k ∏_{j ∈ dims} ψ^i_{j}(x_{1:n}) ∂x_{grad_dim[1]}^k ... ∂x_{grad_dim[end]}^k
for the different columns of the ensemble matrix `X`, where ψ^i is the i-th feature.
"""
grad_xk_basis(f::ExpandedFunction, X, k::Int64, grad_dim::Union{Int64, Array{Int64,1}}, dims::Union{Int64, UnitRange{Int64}, Array{Int64,1}}, idx::Array{Int64,2}) =
             grad_xk_basis!(zeros(size(X,2), size(idx,1)), f, X, k, grad_dim, dims, f.idx)

grad_xk_basis(f::ExpandedFunction, X, k::Int64, grad_dim::Union{Int64, Array{Int64,1}}, dims::Union{Int64, UnitRange{Int64}, Array{Int64,1}}) =
             grad_xk_basis!(zeros(size(X,2), size(f.idx,1)), f, X, k, grad_dim, dims, f.idx)

grad_xk_basis(f::ExpandedFunction, X, k::Int64, grad_dim::Union{Int64, Array{Int64,1}}, idx::Array{Int64,2}) =
             grad_xk_basis!(zeros(size(X,2), size(idx,1)), f, X, k, grad_dim, 1:f.Nx, idx)

grad_xk_basis(f::ExpandedFunction, X, k::Int64, grad_dim::Union{Int64, Array{Int64,1}}) =
             grad_xk_basis!(zeros(size(X,2), size(f.idx,1)), f, X, k, grad_dim, 1:f.Nx, f.idx)



 """
 $(TYPEDSIGNATURES)

 Evaluates in-place the gradient of the basis features of `f` with respect to the different state components
 The i-th column of the output contains ∂ψ^{i}(x_1:n)∂x for the different columns of the ensemble matrix `X`,
 where ψ^{i} is the i-th feature.
 """
function grad_x_basis!(dψ::Array{Float64,3}, f::ExpandedFunction, X, idx::Array{Int64,2})
    m = f.m
    Nx = f.Nx
    # Compute the k-th order deriviative of an expanded function along the direction grad_dim
    Nψreduced = size(idx,1)
    fill!(dψ, 1.0)
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
"""
$(TYPEDSIGNATURES)

Evaluates in-place the gradient of the basis features of `f` with respect to the different state components
The i-th column of the output contains ∂ψ^{i}(x_1:n)∂x for the different columns of the ensemble matrix `X`,
where ψ^{i} is the i-th feature.
"""
grad_x_basis(f::ExpandedFunction, X::Array{Float64,2}, idx::Array{Int64,2}) = grad_x_basis!(zeros(size(X,2), size(idx,1), f.Nx), f, X, idx)

grad_x_basis(f::ExpandedFunction, X::Array{Float64,2}) = grad_x_basis!(zeros(size(X,2), size(f.idx,1), f.Nx), f, X, f.idx)


"""
$(TYPEDSIGNATURES)

Evaluates in-place the hessian of the basis features of `f` with respect to the different state components
The i-th column of the output contains ∂^2ψ^{i}(x_1:n)∂x^2 for the different columns of the ensemble matrix `X`,
where ψ^{i} is the i-th feature.
"""
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
"""
$(TYPEDSIGNATURES)

Evaluates the hessian of the basis features of `f` with respect to the different state components
The i-th column of the output contains ∂^2ψ^{i}(x_1:n)∂x^2 for the different columns of the ensemble matrix `X`,
where ψ^{i} is the i-th feature.
"""
hess_x_basis(f::ExpandedFunction, X::Array{Float64,2}, idx::Array{Int64,2}) = hess_x_basis!(zeros(size(X,2), size(idx,1), f.Nx, f.Nx), f, X, idx)

hess_x_basis(f::ExpandedFunction, X::Array{Float64,2})  = hess_x_basis!(zeros(size(X,2), size(f.idx,1), f.Nx, f.Nx), f, X, f.idx)


"""
$(TYPEDSIGNATURES)

Evaluates in-place the function `f` for the ensemble matrix `X`.
"""
function evaluate!(ψ, f::ExpandedFunction, X::Array{Float64,2})
    ψ .= evaluate_basis(f, X)*f.coeff
    return ψ
end

"""
$(TYPEDSIGNATURES)

Evaluates the function `f` for the ensemble matrix `X`.
"""
evaluate(f::ExpandedFunction, X::Array{Float64,2}) = evaluate!(zeros(size(X,2)), f, X)

"""
$(TYPEDSIGNATURES)

Evaluates the gradient of `f` with respect to the different state components
The i-th column of the output contains ∂f/∂x_i for the different columns of the ensemble matrix `X`.
"""
function grad_x(f::ExpandedFunction, X::Array{Float64,2})
    NxX, Ne = size(X)
    @assert NxX == f.Nx "Wrong dimension of the input"
    dψ = zeros(Ne, f.Nx)
    dψ_basis = grad_x_basis(f, X)
    @tensor dψ[a,b] = dψ_basis[a,c,b] * f.coeff[c]
    return dψ
end

"""
$(TYPEDSIGNATURES)

Evaluates the hessian of `f` with respect to the different state components
The (i,j,k) entry of the output contains ∂^2f(X[:,i])/∂x_j∂x_k.
"""
function hess_x(f::ExpandedFunction, X::Array{Float64,2})
    Nx = f.Nx
    NxX, Ne = size(X)
    @assert NxX == Nx "Wrong dimension of the input"
    d2ψ = zeros(Ne, Nx, Nx)
    d2ψ_basis = hess_x_basis(f, X)

    @tensor d2ψ[a,b,c] = d2ψ_basis[a,d,b,c] * f.coeff[d]
    return d2ψ
end


"""
$(TYPEDSIGNATURES)

Evaluates the gradient of `f` with respect to the last state component.
"""
function grad_xd(f::ExpandedFunction, X::Array{Float64,2})
    dψxd = grad_xk_basis(f, X, 1, f.Nx)
    return dψxd*f.coeff
end

"""
$(TYPEDSIGNATURES)

Evaluates the hessian of `f` with respect to the last state component.
"""
function hess_xd(f::ExpandedFunction, X::Array{Float64,2})
    d2ψxd = grad_xk_basis(f, X, 2, f.Nx)
    return d2ψxd*f.coeff
end


"""
$(TYPEDSIGNATURES)

Computes in-place the gradient with respect to the last state component of the basis of the last univariate function of each feature with multi-indices `idx` at `x`.
"""
function repeated_grad_xk_basis!(out, cache, f::ExpandedFunction, x, idx::Array{Int64,2})
    # Compute the k-th order deriviative of an expanded function along the last state component
    Ne = size(x,1)
    Nx = f.Nx

    # @assert size(out,1) = (N, size(idx, 1)) "Wrong dimension of the output vector"
    # ∂ᵏf/∂x_{grad_dim} = ψ
    k = 1
    grad_dim = Nx
    dims = Nx

    midxj = idx[:, Nx]
    maxj = maximum(midxj)
    # dkψj = zeros(Ne, maxj+1)
    vander!(cache, f.B.B, maxj, k, x)
    Nψreduced = size(idx, 1)
    @avx for l = 1:Nψreduced
        for k=1:Ne
            out[k,l] = cache[k, midxj[l] + 1]
        end
    end

    return out#dkψj[:, midxj .+ 1]
end

repeated_grad_xk_basis!(out, cache, f::ExpandedFunction, x) = repeated_grad_xk_basis!(out, cache, f, x, f.idx)

# repeated_grad_xk_basis(f::ExpandedFunction{m, Nψ, Nx}, x::Array{Float64,1}) where {m, Nψ, Nx} =
"""
$(TYPEDSIGNATURES)

Computes the gradient with respect to the last state component of the basis of the last univariate function of each feature with multi-indices `idx` at `x`.
"""
repeated_grad_xk_basis(f::ExpandedFunction, x, idx::Array{Int64,2}) =
    repeated_grad_xk_basis!(zeros(size(x,1),size(idx,1)), zeros(size(x,1), maximum(idx[:,f.Nx])+1), f, x, idx)
repeated_grad_xk_basis(f::ExpandedFunction, x) = repeated_grad_xk_basis(f, x, f.idx)


"""
$(TYPEDSIGNATURES)

Computes in-place the hessian with respect to the last state component of the basis of the last univariate function of each feature with multi-indices `idx` at `x`.
"""
function repeated_hess_xk_basis!(out, cache, f::ExpandedFunction, x, idx::Array{Int64,2})
    # Compute the Hessian of the basis functions with respect to the last component
    Ne = size(x,1)
    Nx = f.Nx

    # @assert size(out,1) = (N, size(idx, 1)) "Wrong dimension of the output vector"
    # ∂ᵏf/∂x_{grad_dim} = ψ
    k = 2
    grad_dim = Nx
    dims = Nx

    midxj = idx[:, Nx]
    maxj = maximum(midxj)
    #   Compute the kth derivative along grad_dim
    # dkψj = zeros(Ne, maxj+1)
    vander!(cache, f.B.B, maxj, k, x)
    Nψreduced = size(idx, 1)
    @avx for l = 1:Nψreduced
        for k=1:Ne
            out[k,l] = cache[k, midxj[l] + 1]
        end
    end

    return out#dkψj[:, midxj .+ 1]
end

repeated_hess_xk_basis!(out, cache, f::ExpandedFunction, x) = repeated_hess_xk_basis!(out, cache, f, x, f.idx)

repeated_hess_xk_basis(f::ExpandedFunction, x, idx::Array{Int64,2}) =
    repeated_hess_xk_basis!(zeros(size(x,1),size(idx,1)), zeros(size(x,1), maximum(idx[:,f.Nx])+1), f, x, idx)

"""
$(TYPEDSIGNATURES)

Computes the hessian (with respect to the last state component) of the basis of the last univariate function of each feature with multi-indices `idx` at `x`.
"""
repeated_hess_xk_basis(f::ExpandedFunction, x) = repeated_hess_xk_basis(f, x, f.idx)


## Compute ∂_i (∂_k f(x_{1:k}))

# function active_dimik(idx::Array{Int64,2})
#     dim = Int64[]
#     Nx = size(idx,2)
#     boolik = sum(idx .* idx[:,end]; dims = 1)[1,:] .> 0
#     return (1:Nx)[boolik]
# end
#
# active_dimik(f::ExpandedFunction) = active_dimik(f.idx)


# function grad_x_grad_xd(f::ExpandedFunction, X::Array{Float64,2}, idx::Array{Int64,2})

function grad_x_grad_xd(f::ExpandedFunction, X, idx::Array{Int64,2})
    NxX, Ne = size(X)
    m = f.m
    Nx = f.Nx
    Nψ = f.Nψ
    @assert NxX == Nx "Wrong dimension of the input"
    dxdxkψ = zeros(Ne, Nx)

    if f.dim == Int64[]
        return dxdxkψ
    end

    if f.Nx == f.dim[end]
        dxdxkψ_basis = zeros(Ne, Nψ)
        # dxdxkψ_basis = zeros(Ne, Nψ, Nx)
        @inbounds for i ∈ f.dim[f.dim .< f.Nx]
            # Reduce further the computation, we have a non-zero output only if
            # there is a feature such that idx[:,i]*idx[:,Nx]>0
            if any([line[i]*line[f.Nx] for line in eachslice(idx; dims = 1)] .> 0) || iszerofeatureactive(f.B.B)
                fill!(dxdxkψ_basis, 0.0)
                grad_xk_basis!(dxdxkψ_basis, f, X, 1, [i;Nx], idx)
                dxidxkψ = view(dxdxkψ,:,i)
                mul!(dxidxkψ, dxdxkψ_basis, f.coeff)
            end
        end

        d2xkψ = view(dxdxkψ,:,f.Nx)
        fill!(dxdxkψ_basis, 0.0)
        grad_xk_basis!(dxdxkψ_basis, f, X, 2, f.Nx)
        mul!(d2xkψ, dxdxkψ_basis, f.coeff)
        # dxdxkψ = zeros(Ne, Nx)
        # @tensor dxdxkψ[a,b] = dxdxkψ_basis[a,c,b] * f.coeff[c]
    end

    return dxdxkψ
end

"""
$(TYPEDSIGNATURES)

Computes ∂_i (∂_k f(x_{1:k}))
"""
grad_x_grad_xd(f::ExpandedFunction, X) = grad_x_grad_xd(f, X, f.idx)


# This version outputs an object of dimension (Ne, f.dim)
"""
$(TYPEDSIGNATURES)

Computes in-place ∂_i (∂_k f(x_{1:k})). In this routine, gradients are only computed with respect to the active dimensions of `f`.
"""
function reduced_grad_x_grad_xd!(dxdxkψ, f::ExpandedFunction, X, idx::Array{Int64,2})
    NxX, Ne = size(X)
    m = f.m
    Nx = f.Nx
    Nψ = f.Nψ

    dim = f.dim
    dimoff = dim[ dim .< Nx]
    @assert NxX == Nx "Wrong dimension of the input"
    @assert size(dxdxkψ) == (Ne, length(f.dim))

    if f.dim == Int64[]
        fill!(dxdxkψ, 0.0)
        return dxdxkψ
    end

    if f.Nx == f.dim[end]
        dxdxkψ_basis = zeros(Ne, Nψ)
        # dxdxkψ_basis = zeros(Ne, Nψ, Nx)
        @inbounds for i=1:length(dimoff)
            # Reduce further the computation, we have a non-zero output only if
            # there is a feature such that idx[:,i]*idx[:,Nx]>0
            if any([line[dim[i]]*line[f.Nx] for line in eachslice(idx; dims = 1)] .> 0) || iszerofeatureactive(f.B.B)
                fill!(dxdxkψ_basis, 0.0)
                grad_xk_basis!(dxdxkψ_basis, f, X, 1, [dim[i]; Nx], idx)
                dxidxkψ = view(dxdxkψ,:,i)
                mul!(dxidxkψ, dxdxkψ_basis, f.coeff)
            end
        end

        d2xkψ = view(dxdxkψ,:,length(f.dim))
        fill!(dxdxkψ_basis, 0.0)
        grad_xk_basis!(dxdxkψ_basis, f, X, 2, f.Nx)
        mul!(d2xkψ, dxdxkψ_basis, f.coeff)
        # dxdxkψ = zeros(Ne, Nx)
        # @tensor dxdxkψ[a,b] = dxdxkψ_basis[a,c,b] * f.coeff[c]
    end
    return dxdxkψ
end

reduced_grad_x_grad_xd!(dxdxkψ, f::ExpandedFunction, X) = reduced_grad_x_grad_xd!(dxdxkψ, f, X, f.idx)

# This version outputs an object of dimension (Ne, f.dim)
"""
$(TYPEDSIGNATURES)

Computes ∂_i (∂_k f(x_{1:k})). In this routine, gradients are only computed with respect to the active dimensions of `f`.
"""
reduced_grad_x_grad_xd(f::ExpandedFunction, X) = reduced_grad_x_grad_xd!(zeros(size(X,2), length(f.dim)), f, X, f.idx)

# Compute ∂_i ∂_j (∂_k f(x_{1:k}))

# function hess_x_grad_xd(f::ExpandedFunction, X::Array{Float64,2}, idx::Array{Int64,2})
# This version outputs an object of dimension (Ne, f.dim)
"""
$(TYPEDSIGNATURES)

Computes ∂_i ∂_j (∂_k f(x_{1:k}))
"""
function hess_x_grad_xd(f::ExpandedFunction, X, idx::Array{Int64,2})
    NxX, Ne = size(X)
    m = f.m
    Nx = f.Nx
    Nψ = f.Nψ
    @assert NxX == Nx "Wrong dimension of the input"
    # dxidxjdxkψ_basis = zeros(Ne, Nψ, Nx, Nx)
    d2xdxkψ = zeros(Ne, Nx, Nx)

    d2xdxkψ_basis = zeros(Ne, Nψ)

    # Store the derivative of the basis with respect to the last component
    dxkψ  = repeated_grad_xk_basis(f, X[Nx,:])
    d2xkψ = repeated_hess_xk_basis(f, X[Nx,:])

    @inbounds for i ∈ f.dim
        for j ∈ f.dim[f.dim .>= i]
            # Reduce further the computation, we have a non-zero output only if
            # there is a feature such that idx[:,i]*idx[:,j]*idx[:,Nx]>0 or if the first feature is active
            if any([line[i]*line[j]*line[f.Nx] for line in eachslice(f.idx; dims = 1)] .> 0) || iszerofeatureactive(f.B.B)
                fill!(d2xdxkψ_basis, 0.0)
                dxidxjdxkψ = view(d2xdxkψ,:,i,j)
                #  Case i = j = k
                if i==Nx && j==Nx
                    grad_xk_basis!(d2xdxkψ_basis, f, X, 3, Nx)
                    dxidxjdxkψ = view(d2xdxkψ,:,i,j)
                    mul!(dxidxjdxkψ, d2xdxkψ_basis, f.coeff)
                elseif i==j && j!=Nx
                    grad_xk_basis!(d2xdxkψ_basis, f, X, 2, i, f.dim[f.dim .< f.Nx], idx)
                    d2xdxkψ_basis .*= dxkψ
                    dxidxjdxkψ = view(d2xdxkψ,:,i,j)
                    mul!(dxidxjdxkψ, d2xdxkψ_basis, f.coeff)
                elseif i!=Nx && j==Nx #(use symmetry as well)
                    grad_xk_basis!(d2xdxkψ_basis, f, X, 1, i, f.dim[f.dim .< f.Nx], idx)
                    d2xdxkψ_basis .*= d2xkψ
                    dxidxjdxkψ = view(d2xdxkψ,:,i,j)
                    dxjdxidxkψ = view(d2xdxkψ,:,j,i)
                    mul!(dxidxjdxkψ, d2xdxkψ_basis, f.coeff)
                    dxjdxidxkψ .= dxidxjdxkψ
                else # the rest of the cases (use symmetry as well)
                    grad_xk_basis!(d2xdxkψ_basis, f, X, 1, [i;j], f.dim[f.dim .< f.Nx], idx)
                    d2xdxkψ_basis .*= dxkψ
                    dxidxjdxkψ = view(d2xdxkψ,:,i,j)
                    dxjdxidxkψ = view(d2xdxkψ,:,j,i)
                    mul!(dxidxjdxkψ, d2xdxkψ_basis, f.coeff)
                    dxjdxidxkψ .= dxidxjdxkψ
                end
            end
        end
    end

    # dxidxjdxkψ = zeros(Ne, Nx, Nx)
    # @tensor dxidxjdxkψ[a,b,c] = dxidxjdxkψ_basis[a,d,b, c] * f.coeff[d]

    return d2xdxkψ
end

# hess_x_grad_xd(f::ExpandedFunction, X::Array{Float64,2}) = hess_x_grad_xd(f, X, f.idx)
"""
$(TYPEDSIGNATURES)

Computes ∂_i ∂_j (∂_k f(x_{1:k}))
"""
hess_x_grad_xd(f::ExpandedFunction, X) = hess_x_grad_xd(f, X, f.idx)


# function hess_x_grad_xd(f::ExpandedFunction, X::Array{Float64,2}, idx::Array{Int64,2})
"""
$(TYPEDSIGNATURES)

Computes in-place ∂_i ∂_j (∂_k f(x_{1:k})). In this routine, gradients are only computed with respect to the active dimensions of `f`.
"""
function reduced_hess_x_grad_xd!(d2xdxkψ, f::ExpandedFunction, X, idx::Array{Int64,2})
    NxX, Ne = size(X)
    m = f.m
    Nx = f.Nx
    Nψ = f.Nψ
    dim = f.dim
    @assert NxX == Nx "Wrong dimension of the input"
    # dxidxjdxkψ_basis = zeros(Ne, Nψ, Nx, Nx)
    @assert size(d2xdxkψ) == (Ne, length(dim), length(dim))

    d2xdxkψ_basis = zeros(Ne, Nψ)

    # Store the derivative of the basis with respect to the last component
    dxkψ  = repeated_grad_xk_basis(f, view(X, Nx,:))
    d2xkψ = repeated_hess_xk_basis(f, view(X, Nx,:))

    @inbounds for i=1:length(dim)
        for j = i:length(dim)
            # Reduce further the computation, we have a non-zero output only if
            # there is a feature such that idx[:,i]*idx[:,j]*idx[:,Nx]>0
            if any([line[dim[i]]*line[dim[j]]*line[f.Nx] for line in eachslice(f.idx; dims = 1)] .> 0) || iszerofeatureactive(f.B.B)
                fill!(d2xdxkψ_basis, 0.0)
                dxidxjdxkψ = view(d2xdxkψ,:,i,j)
                #  Case i = j = k
                if dim[i]==Nx && dim[j]==Nx
                    grad_xk_basis!(d2xdxkψ_basis, f, X, 3, Nx)
                    dxidxjdxkψ = view(d2xdxkψ,:,i,j)
                    mul!(dxidxjdxkψ, d2xdxkψ_basis, f.coeff)
                elseif dim[i]==dim[j] && dim[j]!=Nx
                    grad_xk_basis!(d2xdxkψ_basis, f, X, 2, dim[i], f.dim[f.dim .< f.Nx], idx)
                    d2xdxkψ_basis .*= dxkψ
                    dxidxjdxkψ = view(d2xdxkψ,:,i,j)
                    mul!(dxidxjdxkψ, d2xdxkψ_basis, f.coeff)
                elseif dim[i]!=Nx && dim[j]==Nx #(use symmetry as well)
                    grad_xk_basis!(d2xdxkψ_basis, f, X, 1, dim[i], f.dim[f.dim .< f.Nx], idx)
                    d2xdxkψ_basis .*= d2xkψ
                    dxidxjdxkψ = view(d2xdxkψ,:,i,j)
                    dxjdxidxkψ = view(d2xdxkψ,:,j,i)
                    mul!(dxidxjdxkψ, d2xdxkψ_basis, f.coeff)
                    dxjdxidxkψ .= dxidxjdxkψ
                else # the rest of the cases (use symmetry as well)
                    grad_xk_basis!(d2xdxkψ_basis, f, X, 1, [dim[i]; dim[j]], f.dim[f.dim .< f.Nx], idx)
                    d2xdxkψ_basis .*= dxkψ
                    dxidxjdxkψ = view(d2xdxkψ,:,i,j)
                    dxjdxidxkψ = view(d2xdxkψ,:,j,i)
                    mul!(dxidxjdxkψ, d2xdxkψ_basis, f.coeff)
                    dxjdxidxkψ .= dxidxjdxkψ
                end
            end
        end
    end

    return d2xdxkψ
end

"""
$(TYPEDSIGNATURES)

Computes ∂_i ∂_j (∂_k f(x_{1:k})). In this routine, gradients are only computed with respect to the active dimensions of `f`.
"""
reduced_hess_x_grad_xd(f::ExpandedFunction, X) = reduced_hess_x_grad_xd!(zeros(size(X, 2), length(f.dim), length(f.dim)), f, X, f.idx)

# Derivative with respect to the some coefficients
"""
$(TYPEDSIGNATURES)

Computes the gradient f(x_{1:k})) with respect to the components `coeff_idx` of `getcoeff(f)`.
"""
function grad_coeff(f::ExpandedFunction, X::Array{Float64,2}, coeff_idx::Array{Int64, 1})
    Nψ = f.Nψ
    # Verify that all the index
    @assert all([0 <=idx <= Nψ for idx in coeff_idx]) "idx is larger than Nψ"
    return evaluate_basis(f, X, f.idx[coeff_idx,:])
end


# Derivative with respect to the coefficients
"""
$(TYPEDSIGNATURES)

Computes the gradient f(x_{1:k})) with respect to the entire vector of coefficient `getcoeff(f)`.
"""
function grad_coeff(f::ExpandedFunction, X::Array{Float64,2})
    return evaluate_basis(f, X)
end


# Hessian with respect to the some coefficients
"""
$(TYPEDSIGNATURES)

Computes the hessian f(x_{1:k})) with respect to the components `coeff_idx` of `getcoeff(f)`.
"""
function hess_coeff(f::ExpandedFunction, X::Array{Float64,2}, coeff_idx::Array{Int64, 1})
    # Verify that all the index
    Nψ = f.Nψ
    Ne = size(X,2)
    Nψreduced = size(coeff_idx,1)
    @assert all([0 <=idx <= Nψ for idx in coeff_idx]) "idx is larger than Nψ"
    return zeros(Ne, Nψreduced, Nψreduced)
end


# Hessian with respect to the coefficients
"""
$(TYPEDSIGNATURES)

Computes the hessian f(x_{1:k})) with respect to the entire vector of coefficient `getcoeff(f)`.
"""
function hess_coeff(f::ExpandedFunction, X::Array{Float64,2})
    Nψ = f.Nψ
    return zeros(size(X,2), Nψ, Nψ)
end


# Gradient with respect to the coefficients of the gradient with respect to xd
"""
$(TYPEDSIGNATURES)

Computes ∂_c ∂_xₖ f.
"""
function grad_coeff_grad_xd(f::ExpandedFunction, X::Array{Float64,2})
    return grad_xk_basis(f, X, 1, f.Nx)
end
# function grad_xk_basis(f::ExpandedFunction{m, Nψ, Nx}, grad_dim::Union{Int64, Array{Int64,1}}, k::Int64, X::Array{Float64,2}, idx::Array{Int64,2}) where {m, Nψ, Nx, Ne}

"""
$(TYPEDSIGNATURES)

Computes ∂_c ∂_xₖ f for the components `coeff_idx` of the coefficient vector of `f`.
"""
function grad_coeff_grad_xd(f::ExpandedFunction, X::Array{Float64,2}, coeff_idx::Array{Int64,1})
    return grad_xk_basis(f, X, 1, f.Nx, f.idx[coeff_idx,:])
end

# Hessian with respect to the coefficients of the gradient with respect to xd
"""
$(TYPEDSIGNATURES)

Computes ∂^2_c ∂_xₖ f.
"""
function hess_coeff_grad_xd(f::ExpandedFunction, X::Array{Float64,2})
    Ne = size(X,2)
    Nψ = f.Nψ
    return zeros(Ne, Nψ, Nψ)
end

"""
$(TYPEDSIGNATURES)

Computes ∂_c ∂_xₖ f for the components `coeff_idx` of the coefficient vector of `f`.
"""
function hess_coeff_grad_xd(f::ExpandedFunction, X::Array{Float64,2}, coeff_idx::Array{Int64,1})
    # Verify that all the index
    Ne = size(X,2)
    Nψ = f.Nψ
    Nψreduced = size(coeff_idx,1)
    @assert all([0 <=idx <= Nψ for idx in coeff_idx]) "idx is larger than Nψ"
    return zeros(Ne, Nψreduced, Nψreduced)
end
