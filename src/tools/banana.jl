export sample_banana, log_pdf_banana

function sample_banana(N)
    X = zeros(2,N)
    for i=1:N
        X[1,i] = randn()
        X[2,i] = X[1,i]^2 + randn()
    end
    return X
end

log_pdf_banana(X) = log_pdf(X[1]) + log_pdf(X[2] - X[1]^2)
