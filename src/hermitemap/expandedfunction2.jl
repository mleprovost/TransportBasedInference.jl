export  evaluate_offdiagbasis!,
        evaluate_offdiagbasis,
        evaluate_diagbasis!,
        evaluate_diagbasis,
        grad_xd_diagbasis!,
        grad_xd_diagbasis,
        hess_xd_diagbasis!,
        hess_xd_diagbasis,
        evaluate!,
        evaluate,
        grad_xd!,
        grad_xd,
        hess_xd!,
        hess_xd,
        grad_coeff!,
        grad_coeff,
        grad_coeff_grad_xd!,
        grad_coeff_grad_xd


## evaluate_offdiagbasis
# X::Array{Float64,2}

"""
$(TYPEDSIGNATURES)

Evaluates in-place the basis of `ExpandedFunction` `f` for the ensemble matrix `X` along the off-diagonal state dimensions for the set of multi-indices of features `idx`
"""
evaluate_offdiagbasis!(ψoff, f::ExpandedFunction, X, idx::Array{Int64,2}) =
    evaluate_basis!(ψoff, f, X, 1:f.Nx-1, idx)

evaluate_offdiagbasis!(ψoff, f::ExpandedFunction, X) =
    evaluate_offdiagbasis!(ψoff, f, X, f.idx)

evaluate_offdiagbasis(f::ExpandedFunction, X, idx::Array{Int64,2}) =
    evaluate_offdiagbasis!(zeros(size(X,2), size(idx,1)), f, X, idx)

evaluate_offdiagbasis(f::ExpandedFunction, X) =
    evaluate_offdiagbasis!(zeros(size(X,2), size(f.idx,1)), f, X)

## evaluate_diagbasis

# evaluate_diagbasis!(ψoff, f::ExpandedFunction{m, Nψ, Nx}, X::Array{Float64,2}, idx::Array{Int64,2}) where {m, Nψ, Nx} =
"""
$(TYPEDSIGNATURES)

Evaluates in-place the basis of `ExpandedFunction` `f` for the ensemble matrix `X` along the last state dimension for the set of multi-indices of features `idx`
"""
evaluate_diagbasis!(ψoff, f::ExpandedFunction, X, idx::Array{Int64,2}) =
    evaluate_basis!(ψoff, f, X, [f.Nx], idx)

evaluate_diagbasis!(ψoff, f::ExpandedFunction, X) =
    evaluate_diagbasis!(ψoff, f, X, f.idx)

evaluate_diagbasis(f::ExpandedFunction, X, idx::Array{Int64,2}) =
        evaluate_diagbasis!(zeros(size(X,2), size(idx,1)), f, X, idx)

evaluate_diagbasis(f::ExpandedFunction, X) =
        evaluate_diagbasis!(zeros(size(X,2), size(f.idx,1)), f, X)

## grad_xd_diagbasis

# grad_xk_basis!(dkψ, f, X, k, grad_dim, dims, idx) where {m, Nψ, Nx}

 # Evaluate derivatives of basis of x_{d}

# grad_xd_diagbasis!(dψxd::Array{Float64,2}, f::ExpandedFunction, X::Array{Float64,2}) =
"""
$(TYPEDSIGNATURES)

Evaluates in-place gradient of the basis of the last component, with respect to the last component for the set of multi-indices of features `idx`
"""
grad_xd_diagbasis!(dψxd::Array{Float64,2}, f::ExpandedFunction, X, idx::Array{Int64,2}) =
                  grad_xk_basis!(dψxd, f, X, 1, f.Nx, f.Nx, idx)

grad_xd_diagbasis!(dψxd::Array{Float64,2}, f::ExpandedFunction, X) =
                  grad_xd_diagbasis(dψxd, f, X, 1, f.Nx, f.Nx, f.idx)

grad_xd_diagbasis(f::ExpandedFunction, X, idx::Array{Int64,2}) =
                 grad_xd_diagbasis!(zeros(size(X,2), size(idx,1)), f, X, idx)

grad_xd_diagbasis(f::ExpandedFunction, X) =
                 grad_xd_diagbasis!(zeros(size(X,2), size(f.idx,1)), f, X, f.idx)


## hess_xd_diagbasis

# Evaluate second derivatives of basis of x_{d}
"""
$(TYPEDSIGNATURES)

Evaluates in-place the hessian of the basis of the last component, with respect to the last component for the set of multi-indices of features `idx`
"""
hess_xd_diagbasis!(d2ψxd::Array{Float64,2}, f::ExpandedFunction, X::Array{Float64,2}, idx::Array{Int64,2}) =
                  grad_xk_basis!(d2ψxd, f, X, 2, f.Nx, f.Nx, idx)

hess_xd_diagbasis!(d2ψxd::Array{Float64,2}, f::ExpandedFunction, X::Array{Float64,2}) =
                  hess_xd_diagbasis(d2ψxd, f, X, 2, f.Nx, f.Nx, f.idx)

hess_xd_diagbasis(f::ExpandedFunction, X::Array{Float64,2}, idx::Array{Int64,2}) =
                 hess_xd_diagbasis!(zeros(size(X,2), size(idx,1)), f, X, idx)

hess_xd_diagbasis(f::ExpandedFunction, X::Array{Float64,2}) =
                 hess_xd_diagbasis!(zeros(size(X,2), size(f.idx,1)), f, X, f.idx)


## evaluate!
"""
$(TYPEDSIGNATURES)

Evaluates in-place the features of multi-indices `idx` for the ensemble matrix `X`.
"""
function evaluate!(ψ::Array{Float64,1}, ψoff::Array{Float64,2}, ψd::Array{Float64,2}, f::ExpandedFunction, X::Array{Float64, 2}, idx::Array{Int64,2})
    @assert size(ψ,1) == size(X,2) "Wrong dimension of the output vector ψ"
    evaluate_offdiagbasis!(ψoff, f, X, idx)
    evaluate_diagbasis!(ψd, f, X, idx)
    elementproductmatmul!(ψ, ψoff, ψd, f.coeff)
    return ψ
end

evaluate!(ψ::Array{Float64,1}, ψoff::Array{Float64,2}, ψd::Array{Float64,2}, f::ExpandedFunction, X::Array{Float64, 2}) =
         evaluate!(ψ, ψoff, ψd, f, X, f.idx)

"""
$(TYPEDSIGNATURES)

Evaluates the features of multi-indices `idx` for the ensemble matrix `X`.
"""
function evaluate(f::ExpandedFunction, X::Array{Float64, 2}, idx::Array{Int64,2})
    Ne = size(X,2)
    evaluate!(zeros(Ne), zeros(Ne, size(idx,1)), zeros(Ne, size(idx,1)), f, X, idx)
end
#
# function evaluate(f::ExpandedFunction, X::Array{Float64, 2})
#     Ne = size(X,2)
#     evaluate!(zeros(Ne), zeros(Ne, size(f.idx,1)), zeros(Ne, size(f.idx,1)), f, X, f.idx)
# end

## grad_xd!
"""
$(TYPEDSIGNATURES)

Computes in-place the gradient with respect to the last component of the features with multi-indices `idx` for the ensemble matrix `X`.
"""
function grad_xd!(dψ::Array{Float64,1}, ψoff::Array{Float64,2}, dψxd::Array{Float64,2}, f::ExpandedFunction, X, idx::Array{Int64,2})
    @assert size(dψ,1) == size(X,2) "Wrong dimension of the output vector ψ"
    evaluate_offdiagbasis!(ψoff, f, X, idx)
    grad_xd_diagbasis!(dψxd, f, X, idx)
    elementproductmatmul!(dψ, ψoff, dψxd, f.coeff)
    return dψ
end

grad_xd!(dψ::Array{Float64,1}, ψoff::Array{Float64,2}, dψxd::Array{Float64,2}, f::ExpandedFunction, X) =
        grad_xd!(dψ, ψoff, dψxd, f, X, f.idx)

# function grad_xd(f::ExpandedFunction, X::Array{Float64, 2}, idx::Array{Int64,2})
"""
$(TYPEDSIGNATURES)

Computes the gradient with respect to the last component of the features with multi-indices `idx` for the ensemble matrix `X`.
"""
function grad_xd(f::ExpandedFunction, X, idx::Array{Int64,2})
    Ne = size(X,2)
    grad_xd!(zeros(Ne), zeros(Ne, size(idx,1)), zeros(Ne, size(idx,1)), f, X, idx)
end

"""
$(TYPEDSIGNATURES)

Computes the gradient with respect to the last component of `ExpandedFunction` `f` for the ensemble matrix `X`.
"""
function grad_xd(f::ExpandedFunction, X)
    Ne = size(X,2)
    grad_xd!(zeros(Ne), zeros(Ne, size(f.idx,1)), zeros(Ne, size(f.idx,1)), f, X, f.idx)
end


## hess_xd!
#     ψ = evaluate_offdiagbasis(f, X)
#     d2ψxd = hess_xd_diagbasis(f, X)
#     return ψ .* d2ψxd * f.coeff
"""
$(TYPEDSIGNATURES)

Computes in-place the hessian with respect to the last component of the features with multi-indices `idx` for the ensemble matrix `X`.
"""
function hess_xd!(d2ψ::Array{Float64,1}, ψoff::Array{Float64,2}, d2ψxd::Array{Float64,2}, f::ExpandedFunction, X::Array{Float64, 2}, idx::Array{Int64,2})
    @assert size(d2ψ,1) == size(X,2) "Wrong dimension of the output vector ψ"
    evaluate_offdiagbasis!(ψoff, f, X, idx)
    hess_xd_diagbasis!(d2ψxd, f, X, idx)
    elementproductmatmul!(d2ψ, ψoff, d2ψxd, f.coeff)
    return d2ψ
end

hess_xd!(d2ψ::Array{Float64,1}, ψoff::Array{Float64,2}, d2ψxd::Array{Float64,2}, f::ExpandedFunction, X::Array{Float64, 2}) =
        hess_xd!(d2ψ, ψoff, d2ψxd, f, X, f.idx)

"""
$(TYPEDSIGNATURES)

Computes the hessian with respect to the last component of the features with multi-indices `idx` for the ensemble matrix `X`.
"""
function hess_xd(f::ExpandedFunction, X::Array{Float64, 2}, idx::Array{Int64,2})
    Ne = size(X,2)
    hess_xd!(zeros(Ne), zeros(Ne, size(idx,1)), zeros(Ne, size(idx,1)), f, X, idx)
end
#
# function hess_xd(f::ExpandedFunction, X::Array{Float64, 2})
#     Ne = size(X,2)
#     hess_xd!(zeros(Ne), zeros(Ne, size(f.idx,1)), zeros(Ne, size(f.idx,1)), f, X, f.idx)
# end

## grad_coeff!

"""
$(TYPEDSIGNATURES)

Computes in-place the gradient with respect to the coefficients of the features with multi-indices `idx` for the ensemble matrix `X`.
"""
function grad_coeff!(dcψ::Array{Float64,2}, ψoff::Array{Float64,2}, ψd::Array{Float64,2}, f::ExpandedFunction, X::Array{Float64, 2}, idx::Array{Int64,2})
    @assert size(dcψ,1) == size(X,2) "Wrong dimension of the output vector ψ"
    # evaluate_basis!(dcψ, f, X, 1:Nx, idx)
    evaluate_offdiagbasis!(ψoff, f, X, idx)
    evaluate_diagbasis!(ψd, f, X)
    dcψ .= ψoff .* ψd
    return dcψ
end

grad_coeff!(dcψ::Array{Float64,2}, ψoff::Array{Float64,2}, ψd::Array{Float64,2}, f::ExpandedFunction, X::Array{Float64, 2}) =
         grad_coeff!(dcψ, ψoff, ψd, f, X, f.idx)

 """
 $(TYPEDSIGNATURES)

 Computes the gradient with respect to the coefficients of the features with multi-indices `idx` for the ensemble matrix `X`.
 """
function grad_coeff(f::ExpandedFunction, X::Array{Float64, 2}, idx::Array{Int64,2})
         grad_coeff!(zeros(size(X,2), size(idx,1)), zeros(size(X,2), size(idx,1)), zeros(size(X,2), size(idx,1)), f, X, idx)
end

# function grad_coeff(f::ExpandedFunction, X::Array{Float64, 2})
#          grad_coeff!(zeros(size(X,2), size(f.idx,1)), zeros(size(X,2), size(f.idx,1)), zeros(size(X,2), size(f.idx,1)), f, X, f.idx)
# end

## grad_coeff_grad_xd!
"""
$(TYPEDSIGNATURES)

Computes in-place ∂_c ∂_x{k} (∑_{α ∈ idx} c_{α} ψ_{α}(x_{1:k})).
"""
function grad_coeff_grad_xd!(dcdψxd::Array{Float64,2}, ψoff::Array{Float64,2}, dψd::Array{Float64,2},f::ExpandedFunction, X::Array{Float64, 2}, idx::Array{Int64,2})
    @assert size(dcdψxd,1) == size(X,2) "Wrong dimension of the output vector ψ"
    # evaluate_basis!(dcψ, f, X, 1:Nx, idx)
    evaluate_offdiagbasis!(ψoff, f, X, idx)
    grad_xd_diagbasis!(dψd, f, X, idx)
    dcdψxd .= ψoff .* dψd
    return dcdψxd
end

grad_coeff_grad_xd!(dcdψxd::Array{Float64,2}, ψoff::Array{Float64,2}, dψd::Array{Float64,2},f::ExpandedFunction, X::Array{Float64, 2}) =
    grad_coeff_grad_xd!(dcdψxd, ψoff, dψd, f, X, f.idx)

# function grad_coeff_grad_xd(f::ExpandedFunction, X::Array{Float64, 2}, idx::Array{Int64,2})
#          grad_coeff_grad_xd!(zeros(size(X,2), size(idx,1)), zeros(size(X,2), size(idx,1)), zeros(size(X,2), size(idx,1)), f, X, idx)
# end
#
# function grad_coeff_grad_xd(f::ExpandedFunction, X::Array{Float64, 2})
#          grad_coeff_grad_xd!(zeros(size(X,2), size(f.idx,1)), zeros(size(X,2), size(f.idx,1)), zeros(size(X,2), size(f.idx,1)), f, X, f.idx)
# end
