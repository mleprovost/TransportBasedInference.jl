export fact2, member, allequal

"""
    fact2(n)

Compute the double factorial of `n`, i.e. the product of the odd (if `n` is odd) or even (if `n` is even) numbers up to `n`.
"""
function fact2(n::Int64)
    if n == 1 || n==0
        return 1;
    else
        if mod(n,2) == 0 # the number is even
            return prod(2:2:n);
        else # the number is odd
            return prod(1:2:n);
        end
    end
end

"""
    member(X, idx)

Extract the `idx` ensemble member of the ensemble matrix `X`
"""
member(X, idx) = X[:,idx]

"""
    allequal(x, r)

A function to verify that all the entries of the vector `x` are equal to `r`
"""
allequal(x, r) = all(y->y==r, x)
