export log_pdf, gradlog_pdf, hesslog_pdf

"""
    log_pdf(x)

Compute the log of the density of the univariate standard Gaussian distribution (zero mean and unitary standard deviation).
"""
log_pdf(x; σ = 1.0) = -0.5*(log(2*π*σ^2) + (x/σ)^2)

"""
    gradlog_pdf(x)

Compute the gradient of the log of the density of the univariate standard Gaussian distribution (zero mean and unitary standard deviation).
"""
gradlog_pdf(x; σ  = 1.0) = -x*inv(σ^2)

"""
    hesslog_pdf(x)

Compute the hessian of the log of the density of the univariate standard Gaussian distribution (zero mean and unitary standard deviation).
"""
hesslog_pdf(x; σ = 1.0) = -inv(σ^2)
