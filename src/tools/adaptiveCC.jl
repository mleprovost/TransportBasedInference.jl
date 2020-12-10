

# https://discourse.julialang.org/t/julia-integral-calculation-community-module-or-own-module/24278/14
# Finally note that, If you are calculating F(x) = quadgk(f, 0, x) for a whole bunch of different values of x ∈ [0,a],
# then there are much more efficient things that you can do than separate quadgk integrals for each x,
# which wastes a lot of calculations that could be shared between different x values, especially if f(x) is a smooth function.
# For example, using ApproxFun.jl, you can do F = cumsum(Fun(f, 0..a)), and then evaluate F(x) very quickly —
# this works by first constructing a polynomial approximation of f(x) on [0,a] and
# then forming the polynomial F(x) that is the indefinite integral.
