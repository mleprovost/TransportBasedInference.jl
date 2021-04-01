export bisection, hybridsolver, hybridinverse!

function bisectionbook(f′, a, b, ϵ)
if a > b; a,b = b,a; end # ensure a < b
ya, yb = f′(a), f′(b)
if ya == 0; b = a; end
if yb == 0; a = b; end
while b - a > ϵ
x = (a+b)/2
y = f′(x)
if y == 0
a, b = x, x
elseif sign(y) == sign(ya)
a = x
else
b = x
end
end
return (a,b)
end

function bisection(x, fx, xm, fm, xp, fp)
@assert xm < xp "Error in the order of the argument"
    if sign(fx) == sign(fp)
    xp = x
    fp = fx
    else
    xm = x
    fm = fx
    end
    # More stable than 0.5*(xp +xm)
    x = xm + 0.5*(xp - xm)
    return x, fx, xm, fm, xp, fp
end

# Combine Bisection and Newton method, this method has guaranteed convergence.
function hybridsolver(f, g, out, a, b; ϵx = 1e-4, ϵf = 1e-4, niter = 100)
    dxold = abs(b-a)
    dx = dxold
    fout = f(out)
    gout = g(out)


    fa = f(a)
    fb = f(b)

    @inbounds for j=1:niter
        # Bisect if Newton out of range, or not decreasing fast enough.
        if ((out - b)*gout - fout)*((out - a)*gout - fout) > 0.0 || abs(2.0*fout) >  abs(dxold * gout)
            dxold = dx
            dx = 0.5*(b-a)
            out = a + dx
            if isapprox(a, out, atol = ϵx)
                return out
            end
        else #Newton step is acceptable
            dxold = dx
            dx    = fout/gout
            tmp   = out
            out = out - dx
            if isapprox(tmp, out, atol = ϵx)
                return out
            end
        end
        # Convergence criterion
        if abs(dx) < ϵx || abs(fout) < ϵf
            return out
        end
        # The one new function evaluation per iteration
        fout = f(out)
        gout = g(out)
        # Maintain the bracket on the root
        if fout<0.0
            a = out
        else
            b = out
        end
    end

    return out
end

function hybridinverse!(X, F, R::IntegratedFunction, S::Storage)
    Nψ = R.Nψ
    Nx = R.Nx
    NxX, Ne = size(X)

    @assert NxX == R.Nx "Wrong dimension of the sample X"

    cache  = zeros(Ne, Nψ)
    cache_vander = zeros(Ne, maximum(R.f.f.idx[:,Nx])+1)
    f0 = zeros(Ne)

    # Remove f(x_{1:k-1},0) from the output F
    @avx for i=1:Ne
        f0i = zero(Float64)
        for j=1:Nψ
            f0i += (S.ψoffψd0[i,j])*R.f.f.coeff[j]
        end
        F[i] -= f0i
    end


    # lower and upper brackets of the ensemble members
    xk = view(X, Nx, :)
    xm = copy(xk)
    xp = copy(xk)
    σ = std(xm)
    xm .-= 1.0*σ
    xp .+= 1.0*σ
    fm = zeros(Ne)
    fp = zeros(Ne)
    # Find a bracket for the different samples
    niter = 100
    factor = 1.6
    counter = 0
    bracketed = false
    while bracketed == false
        functionalf!(fm, xm, cache, cache_vander, S.ψoff, F, R)
        functionalf!(fp, xp, cache, cache_vander, S.ψoff, F, R)
        # We know that the function is strictly increasing
        @show all(fm .< 0.0)
        @show all(fp .> 0.0)
        if all(fm .< 0.0) && all(fp .> 0.0)
            bracketed = true
            break
        end
        @inbounds for i=1:Ne
            if fm[i]*fp[i] > 0.0
                counter += 1
                center, width = 0.5*(xp[i] + xm[i]), factor*(xp[i] - xm[i])
                xm[i] = center - width
                xp[i] = center + width
            end
        end
        counter += 1
    end

    @assert counter < niter "Maximal number of iterations reached"

    @show all(fm .< 0.0) && all(fp .> 0.0)


end
function bracket_sign_change(f, a, b; k=2, niter = 100)
    if a > b; a,b = b,a; end # ensure a < b
    center, half_width = (b+a)/2, (b-a)/2
    counter = 0
    while f(a)*f(b) > 0
    counter += 1
    half_width *= k
    a = center - half_width
    b = center + half_width
    end
    @assert counter < niter "Maximal number of iterations reached"
    return a, b
end
