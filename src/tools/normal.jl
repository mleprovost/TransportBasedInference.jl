# Define the Hessian of the logpdf of a Normal distribution
# The logpdf and gradient of the logpdf are already defined in Distributions.jl

export log_pdf, gradlog_pdf, hesslog_pdf

# gradlogpdf(Normal(), 2.0)
# logpdf(Normal(1.0, 2.0),randn(10))

# Define only for the standard Normal distribution
log_pdf(x) = -0.5*x^2
gradlog_pdf(x) = -x
hesslog_pdf(x) = -1.0
