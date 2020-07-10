

export ExpandedFunction, alleval,
       evaluate_basis, grad_xk_basis,
       grad_x_basis, hess_x_basis

# ExpandedFunction decomposes a multi-dimensional function f:Rᴹ → R onto
# a basis of MultiFunctions ψ_α where c_α are scalar coefficients
# for each MultiFunction:
# f(x1, x2, ..., xNx) = ∑_α c_α ψ_α(x1, x2, ..., xNx)
# Nψ is the number of MultiFunctions used,
# Nx is the dimension of the input vector x

struct ExpandedFunction{m, Nψ, Nx}
    B::MultiBasis{m, Nx}
    idx::Array{Int64,2}
    c::Array{Float64,1}
    function ExpandedFunction(B::MultiBasis{m, Nx}, idx::Array{Int64,2}, c::Array{Float64,1}) where {m, Nx}
            @assert size(idx,1) == size(c,1) "The dimension of the basis functions don't
                                            match the number of coefficients"
            Nψ = size(idx,1)

            @assert size(idx,2) == Nx "Size of the array of multi-indices idx is wrong"
        return new{m, Nψ, Nx}(B, idx, c)
    end
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


function grad_x_basis(f::ExpandedFunction{m, Nψ, Nx}, ens::EnsembleState{Nx, Ne}) where {m, Nψ, Nx, Ne}
    # Compute the k=th order deriviative of an expanded function along the direction grad_dim
    dψ = ones(Ne, Nψ, Nx)
    @inbounds for j=1:Nx
            dψ[:,:,j] .= deepcopy(grad_xk_basis(f, j, 1, ens))
        end
    return dψ
end

function hess_x_basis(f::ExpandedFunction{m, Nψ, Nx}, ens::EnsembleState{Nx, Ne}) where {m, Nψ, Nx, Ne}
    # Compute the k=th order deriviative of an expanded function along the direction grad_dim
    d2ψ = ones(Ne, Nψ, Nx, Nx)

    @inbounds for i=1:Nx
        d2ψ[:,:,i,i] .= deepcopy(grad_xk_basis(f, i, 2, ens))
    end

    @inbounds for i=1:Nx
                for j=i+1:Nx
                    d2ψ[:,:,i,j] .= deepcopy(grad_xk_basis(f, [i;j], 1, ens))
                    d2ψ[:,:,j,i] .= deepcopy(d2ψ[:,:,i,j])
                end
    end
    return d2ψ
end
