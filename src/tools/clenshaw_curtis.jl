export clenshaw_curtis

"""
    clenshaw_curtis(N)

Compute the nodes `x` and weights `w` for integrating a
continuous functions from [-1,1] using the Clenshaw-Curtis
integration rule with order `N`.
"""
function clenshaw_curtis(N::Int64)


    @assert N>0 "Order must be greater than 0"

    x = zeros(N)
    w = ones(N)


    if N==1
        #x[1]=0.0
        w[1] = 2.0
        return x, w
    end

    # Compute x values
    @inbounds for i=1:N
        x[i] = cos((N-i)*π/(N-1))
    end

    n = N
    x[1] = -1.0

    if mod(n,2)==1
        x[ceil(Int64, (n+1)/2)] = 0.0
    end

    x[n] = 1.0

    @inbounds for i=1:n
        θ = (i-1)*π/(n-1)
        for j=1:ceil(Int64,(n-1)/2)
            if 2*j == n-1
                b = 1.0
            else
                b = 2.0
            end

            w[i] -= b*cos(2*j*θ)/(4*j*j-1)
        end
    end

    w[1] *= 1/(n-1)
    w[2:n] .*= 2.0/(n-1)
    w[n] *= 1/(n-1)

    return x, w
end
