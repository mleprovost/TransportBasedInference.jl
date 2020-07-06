

export fact2
# Compute the product of the odd or even numbers up to n
# For
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
