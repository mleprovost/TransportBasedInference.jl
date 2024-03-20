export sample_banana, log_pdf_banana

"""
    sample_banana(N; μ = 0.0, σ = 2.0, bananicity = 0.2)

Generate `N` samples [x₁; x₂] of the Banana distribution defined as:
    x₁ ∼ N(μ, σ²),
    ϵ  ∼ N(0, 1),
    x₂ ∼ bananicity × (x₁^2 - σ²) + ϵ.
"""
function sample_banana(N; μ = 0.0, σ = 2.0, bananicity = 0.2)
    X = zeros(2,N)
    for i=1:N
        X[1,i] = μ + σ*randn()
        X[2,i] = bananicity*(X[1,i]^2 - σ^2) + randn()
    end
    return X
end

# log_pdf_banana(X) = log_pdf(X[1]) + log_pdf(X[2] - X[1]^2)
# log_pdf_banana(X; a = 0.0, b = 1.0, c = 1.0) = log_pdf((X[1] -a)/b) + log_pdf(X[2] - c*X[1]^2)

"""
    log_pdf_banana(X; μ = 0.0, σ = 2.0, bananicity = 0.2)

Compute the logarithm of the porbability density function (pdf) of the Banana distribution at `X`.
See documentation of sample_banana for details on the Banana distribution.
"""
log_pdf_banana(X::AbstractVector; μ = 0.0, σ = 2.0, bananicity = 0.2) = log_pdf((X[1]-μ)/σ) + log_pdf(X[2] - bananicity*(X[1]^2 - σ^2))
