export  ParametricFunction,
        evaluate_offdiagbasis!,
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


struct ParametricFunction
    f::ExpandedFunction
end

## evaluate_offdiagbasis
# X::Array{Float64,2}
evaluate_offdiagbasis!(ψoff, fp::ParametricFunction, X, idx::Array{Int64,2}) =
    evaluate_basis!(ψoff, fp.f, X, 1:fp.f.Nx-1, idx)

evaluate_offdiagbasis!(ψoff, fp::ParametricFunction, X) =
    evaluate_offdiagbasis!(ψoff, fp, X, fp.f.idx)

evaluate_offdiagbasis(fp::ParametricFunction, X, idx::Array{Int64,2}) =
        evaluate_offdiagbasis!(zeros(size(X,2), size(idx,1)), fp, X, idx)

evaluate_offdiagbasis(fp::ParametricFunction, X) =
        evaluate_offdiagbasis!(zeros(size(X,2), size(fp.f.idx,1)), fp, X)

## evaluate_diagbasis

# evaluate_diagbasis!(ψoff, fp::ParametricFunction{m, Nψ, Nx}, X::Array{Float64,2}, idx::Array{Int64,2}) where {m, Nψ, Nx} =

evaluate_diagbasis!(ψoff, fp::ParametricFunction, X, idx::Array{Int64,2}) =
    evaluate_basis!(ψoff, fp.f, X, [fp.f.Nx], idx)

evaluate_diagbasis!(ψoff, fp::ParametricFunction, X) =
    evaluate_diagbasis!(ψoff, fp, X, fp.f.idx)

evaluate_diagbasis(fp::ParametricFunction, X, idx::Array{Int64,2}) =
        evaluate_diagbasis!(zeros(size(X,2), size(idx,1)), fp, X, idx)

evaluate_diagbasis(fp::ParametricFunction, X) =
        evaluate_diagbasis!(zeros(size(X,2), size(fp.f.idx,1)), fp, X)

## grad_xd_diagbasis

# grad_xk_basis!(dkψ, f, X, k, grad_dim, dims, idx) where {m, Nψ, Nx}

 # Evaluate derivatives of basis of x_{d}

# grad_xd_diagbasis!(dψxd::Array{Float64,2}, fp::ParametricFunction, X::Array{Float64,2}) =
grad_xd_diagbasis!(dψxd::Array{Float64,2}, fp::ParametricFunction, X, idx::Array{Int64,2}) =
                  grad_xk_basis!(dψxd, fp.f, X, 1, fp.f.Nx, fp.f.Nx, idx)

grad_xd_diagbasis!(dψxd::Array{Float64,2}, fp::ParametricFunction, X) =
                  grad_xd_diagbasis(dψxd, fp, X, 1, fp.f.Nx, fp.f.Nx, fp.f.idx)

grad_xd_diagbasis(fp::ParametricFunction, X, idx::Array{Int64,2}) =
                 grad_xd_diagbasis!(zeros(size(X,2), size(idx,1)), fp, X, idx)

grad_xd_diagbasis(fp::ParametricFunction, X) =
                 grad_xd_diagbasis!(zeros(size(X,2), size(fp.f.idx,1)), fp, X, fp.f.idx)


## hess_xd_diagbasis

# Evaluate second derivatives of basis of x_{d}
hess_xd_diagbasis!(d2ψxd::Array{Float64,2}, fp::ParametricFunction, X::Array{Float64,2}, idx::Array{Int64,2}) =
                  grad_xk_basis!(d2ψxd, fp.f, X, 2, fp.f.Nx, fp.f.Nx, idx)

hess_xd_diagbasis!(d2ψxd::Array{Float64,2}, fp::ParametricFunction, X::Array{Float64,2}) =
                  hess_xd_diagbasis(d2ψxd, fp, X, 2, fp.f.Nx, fp.f.Nx, fp.f.idx)

hess_xd_diagbasis(fp::ParametricFunction, X::Array{Float64,2}, idx::Array{Int64,2}) =
                 hess_xd_diagbasis!(zeros(size(X,2), size(idx,1)), fp, X, idx)

hess_xd_diagbasis(fp::ParametricFunction, X::Array{Float64,2}) =
                 hess_xd_diagbasis!(zeros(size(X,2), size(fp.f.idx,1)), fp, X, fp.f.idx)


## evaluate!

function evaluate!(ψ::Array{Float64,1}, ψoff::Array{Float64,2}, ψd::Array{Float64,2}, fp::ParametricFunction, X::Array{Float64, 2}, idx::Array{Int64,2})
    @assert size(ψ,1) == size(X,2) "Wrong dimension of the output vector ψ"
    evaluate_offdiagbasis!(ψoff, fp, X, idx)
    evaluate_diagbasis!(ψd, fp, X, idx)
    elementproductmatmul!(ψ, ψoff, ψd, fp.f.coeff)
    return ψ
end

evaluate!(ψ::Array{Float64,1}, ψoff::Array{Float64,2}, ψd::Array{Float64,2}, fp::ParametricFunction, X::Array{Float64, 2}) =
         evaluate!(ψ, ψoff, ψd, fp, X, fp.f.idx)

function evaluate(fp::ParametricFunction, X::Array{Float64, 2}, idx::Array{Int64,2})
    Ne = size(X,2)
    evaluate!(zeros(Ne), zeros(Ne, size(idx,1)), zeros(Ne, size(idx,1)), fp, X, idx)
end

function evaluate(fp::ParametricFunction, X::Array{Float64, 2})
    Ne = size(X,2)
    evaluate!(zeros(Ne), zeros(Ne, size(fp.f.idx,1)), zeros(Ne, size(fp.f.idx,1)), fp, X, fp.f.idx)
end

## grad_xd!

function grad_xd!(dψ::Array{Float64,1}, ψoff::Array{Float64,2}, dψxd::Array{Float64,2}, fp::ParametricFunction, X, idx::Array{Int64,2})
    @assert size(dψ,1) == size(X,2) "Wrong dimension of the output vector ψ"
    evaluate_offdiagbasis!(ψoff, fp, X, idx)
    grad_xd_diagbasis!(dψxd, fp, X, idx)
    elementproductmatmul!(dψ, ψoff, dψxd, fp.f.coeff)
    return dψ
end

grad_xd!(dψ::Array{Float64,1}, ψoff::Array{Float64,2}, dψxd::Array{Float64,2}, fp::ParametricFunction, X) =
        grad_xd!(dψ, ψoff, dψxd, fp, X, fp.f.idx)

# function grad_xd(fp::ParametricFunction, X::Array{Float64, 2}, idx::Array{Int64,2})
function grad_xd(fp::ParametricFunction, X, idx::Array{Int64,2})
    Ne = size(X,2)
    grad_xd!(zeros(Ne), zeros(Ne, size(idx,1)), zeros(Ne, size(idx,1)), fp, X, idx)
end

function grad_xd(fp::ParametricFunction, X)
    Ne = size(X,2)
    grad_xd!(zeros(Ne), zeros(Ne, size(fp.f.idx,1)), zeros(Ne, size(fp.f.idx,1)), fp, X, fp.f.idx)
end


## hess_xd!
#     ψ = evaluate_offdiagbasis(fp, X)
#     d2ψxd = hess_xd_diagbasis(fp, X)
#     return ψ .* d2ψxd * fp.f.coeff

function hess_xd!(d2ψ::Array{Float64,1}, ψoff::Array{Float64,2}, d2ψxd::Array{Float64,2}, fp::ParametricFunction, X::Array{Float64, 2}, idx::Array{Int64,2})
    @assert size(d2ψ,1) == size(X,2) "Wrong dimension of the output vector ψ"
    evaluate_offdiagbasis!(ψoff, fp, X, idx)
    hess_xd_diagbasis!(d2ψxd, fp, X, idx)
    elementproductmatmul!(d2ψ, ψoff, d2ψxd, fp.f.coeff)
    return d2ψ
end

hess_xd!(d2ψ::Array{Float64,1}, ψoff::Array{Float64,2}, d2ψxd::Array{Float64,2}, fp::ParametricFunction, X::Array{Float64, 2}) =
        hess_xd!(d2ψ, ψoff, d2ψxd, fp, X, fp.f.idx)

function hess_xd(fp::ParametricFunction, X::Array{Float64, 2}, idx::Array{Int64,2})
    Ne = size(X,2)
    hess_xd!(zeros(Ne), zeros(Ne, size(idx,1)), zeros(Ne, size(idx,1)), fp, X, idx)
end

function hess_xd(fp::ParametricFunction, X::Array{Float64, 2})
    Ne = size(X,2)
    hess_xd!(zeros(Ne), zeros(Ne, size(fp.f.idx,1)), zeros(Ne, size(fp.f.idx,1)), fp, X, fp.f.idx)
end

## grad_coeff!

function grad_coeff!(dcψ::Array{Float64,2}, ψoff::Array{Float64,2}, ψd::Array{Float64,2}, fp::ParametricFunction, X::Array{Float64, 2}, idx::Array{Int64,2})
    @assert size(dcψ,1) == size(X,2) "Wrong dimension of the output vector ψ"
    # evaluate_basis!(dcψ, fp.f, X, 1:Nx, idx)
    evaluate_offdiagbasis!(ψoff, fp, X, idx)
    evaluate_diagbasis!(ψd, fp, X)
    dcψ .= ψoff .* ψd
    return dcψ
end

grad_coeff!(dcψ::Array{Float64,2}, ψoff::Array{Float64,2}, ψd::Array{Float64,2}, fp::ParametricFunction, X::Array{Float64, 2}) =
         grad_coeff!(dcψ, ψoff, ψd, fp, X, fp.f.idx)

function grad_coeff(fp::ParametricFunction, X::Array{Float64, 2}, idx::Array{Int64,2})
         grad_coeff!(zeros(size(X,2), size(idx,1)), zeros(size(X,2), size(idx,1)), zeros(size(X,2), size(idx,1)), fp, X, idx)
end

function grad_coeff(fp::ParametricFunction, X::Array{Float64, 2})
         grad_coeff!(zeros(size(X,2), size(fp.f.idx,1)), zeros(size(X,2), size(fp.f.idx,1)), zeros(size(X,2), size(fp.f.idx,1)), fp, X, fp.f.idx)
end

## grad_coeff_grad_xd!

function grad_coeff_grad_xd!(dcdψxd::Array{Float64,2}, ψoff::Array{Float64,2}, dψd::Array{Float64,2},fp::ParametricFunction, X::Array{Float64, 2}, idx::Array{Int64,2})
    @assert size(dcdψxd,1) == size(X,2) "Wrong dimension of the output vector ψ"
    # evaluate_basis!(dcψ, fp.f, X, 1:Nx, idx)
    evaluate_offdiagbasis!(ψoff, fp, X, idx)
    grad_xd_diagbasis!(dψd, fp, X, idx)
    dcdψxd .= ψoff .* dψd
    return dcdψxd
end

grad_coeff_grad_xd!(dcdψxd::Array{Float64,2}, ψoff::Array{Float64,2}, dψd::Array{Float64,2},fp::ParametricFunction, X::Array{Float64, 2}) =
    grad_coeff_grad_xd!(dcdψxd, ψoff, dψd, fp, X, fp.f.idx)

function grad_coeff_grad_xd(fp::ParametricFunction, X::Array{Float64, 2}, idx::Array{Int64,2})
         grad_coeff_grad_xd!(zeros(size(X,2), size(idx,1)), zeros(size(X,2), size(idx,1)), zeros(size(X,2), size(idx,1)), fp, X, idx)
end

function grad_coeff_grad_xd(fp::ParametricFunction, X::Array{Float64, 2})
         grad_coeff_grad_xd!(zeros(size(X,2), size(fp.f.idx,1)), zeros(size(X,2), size(fp.f.idx,1)), zeros(size(X,2), size(fp.f.idx,1)), fp, X, fp.f.idx)
end
