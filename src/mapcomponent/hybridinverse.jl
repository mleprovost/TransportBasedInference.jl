export bisection, hybridsolver, hybridinverse!


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
# Convergence order should be between linear and quadratic
# http://www.m-hikari.com/ams/ams-2017/ams-53-56-2017/p/hahmAMS53-56-2017.pdf
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

function hybridinverse!(X, F, R::IntegratedFunction, S::Storage; niter= 100, ϵx = 1e-4, ϵf = 1e-4, P::Parallel = serial)
    Nψ = R.Nψ
    Nx = R.Nx
    NxX, Ne = size(X)

    @assert NxX == R.Nx "Wrong dimension of the sample X"

    cache  = zeros(Ne, Nψ)
    cache_vander = zeros(Ne, maximum(R.f.f.idx[:,Nx])+1)
    fout = zeros(Ne)

    # Remove f(x_{1:k-1},0) from the output F
    @avx for i=1:Ne
        f0i = zero(Float64)
        for j=1:Nψ
            f0i += (S.ψoffψd0[i,j])*R.f.f.coeff[j]
        end
        F[i] -= f0i
    end

    # We can tighter the brackets, as we know that g(dxf)>0 for every dxf value
    # so either the upper or the lower bound is zero for the different smaples

    # lower and upper brackets of the ensemble members
    xk = view(X, Nx, :)
    xbound = copy(xk)
    fbound = zeros(Ne)

    σ = std(xk)
    # We can reduce by two the number of evaluations, if we use the sign of F
    xbound = xk +2σ*sign.(F)
    # @inbounds for i=1:Ne
    #     if F[i]>0.0
    #         # implies the root is >0
    #         # xa[i] = 0.0
    #         # xb[i] = xk[i] + 2σ
    #         xbound[i] += 2σ
    #     else
    #         # xa[i] = xk[i] - 2σ
    #         # xb[i] = 0.0
    #         xbound[i] -= 2σ
    #     end
    # end
    # Evaluate at 0.0 f
    # We can rewrite the bracketing with a single function evaluation for xbound
    factor = 1.6
    bracketed = false
    for j=1:niter
        functionalf!(fbound, xbound, cache, cache_vander, S.ψoff, F, R)
        # We know that the function is strictly increasing
        if all(sign.(fbound) == sign.(xbound))
            bracketed = true
            break
        end
        @inbounds for i=1:Ne
            if sign(fbound[i]) != sign(xbound[i])
                Δ = factor*(xbound[i] - 0.0)
                if xbound[i]<0.0
                    xbound[i] -= Δ
                else
                    xbound[i] += Δ
                end
            end
        end
    end

    @assert bracketed == true "Maximal number of iterations reached, without bracket"

    # Let create xa, xb, fa, fb based on xbound
    xa = zeros(Ne)
    xb = zeros(Ne)
    fa = zeros(Ne)
    fb = zeros(Ne)

    @inbounds for i=1:Ne
        if xbound[i]<0.0
            xa[i] = xbound[i]
            fa[i] = fbound[i]
            fb[i] = F[i]
        else
            xb[i] = xbound[i]
            fa[i] = F[i]
            fb[i] = fbound[i]
        end
    end

    ##  Find a bracket for the different samples
    # factor = 1.6
    # bracketed = false
    # @inbounds for j=1:niter
    #     functionalf!(fa, xa, cache, cache_vander, S.ψoff, F, R)
    #     functionalf!(fb, xb, cache, cache_vander, S.ψoff, F, R)
    #     # We know that the function is strictly increasing
    #     if all(fa .< 0.0) && all(fb .> 0.0)
    #         bracketed = true
    #         break
    #     end
    #     @inbounds for i=1:Ne
    #         if fa[i]*fb[i] > 0.0
    #             Δ = factor*(xb[i] - xa[i])
    #             if abs(fa[i]) < abs(fb[i])
    #                 xa[i] -= Δ
    #             else
    #                 xb[i] += Δ
    #             end
    #         end
    #     end
    # end
    # @assert bracketed == true "Maximal number of iterations reached, without bracket"

    dx = zeros(Ne)
    gout = zeros(Ne)
    # Initial guess: mid point of the bracket
    @avx for i=1:Ne
        xai = xa[i]
        xbi = xb[i]
        dx[i] = xbi - xai
        xk[i] = xai + 0.5*(dx[i])
    end

    dxold = copy(dx)

    functionalf!(fout, xk, cache, cache_vander, S.ψoff, F, R)
    functionalg1D!(gout, xk, cache, cache_vander, S.ψoff, F, R)
    convergence = false
    @inbounds for j=1:niter
        @inbounds for i=1:Ne
            # Bisect if Newton out of range, or not decreasing fast enough.
            if ((xk[i] - xb[i])*gout[i] - fout[i])*((xk[i] - xa[i])*gout[i] - fout[i]) > 0.0 ||
                abs(2.0*fout[i]) >  abs(dxold[i] * gout[i])
                dxold[i] = dx[i]
                dx[i] = 0.5*(xb[i]-xa[i])
                xk[i] = xa[i] + dx[i]
                # if isapprox(xa[i], xk[i], atol = ϵx)
                # end
            else #Newton step is acceptable
                dxold[i] = dx[i]
                dx[i]    = fout[i]/gout[i]
                xk[i] -=  dx[i]
                # if isapprox(dx[i], 0.0, atol = ϵx)
                # # if isapprox(tmp, xk[i], atol = ϵx)
                #     continue
                # end
            end

            # if abs(dx[i]) < ϵx || abs(fout[i]) < ϵf
            #     continue
            # end
        end
        # Convergence criterion
        if norm(xa - xk, Inf) < ϵx || norm(dx, Inf) < ϵx || norm(fout, Inf) < ϵf
            convergence = true
            break
        end

        # Evaluate the function and its gradient for the different samples
        functionalf!(fout, xk, cache, cache_vander, S.ψoff, F, R)
        functionalg1D!(gout, xk, cache, cache_vander, S.ψoff, F, R)

        # Maintain the bracket on the root
        @inbounds for i=1:Ne
            if fout[i]<0.0
                xa[i] = xk[i]
            else
                xb[i] = xk[i]
            end
        end
    end
    @assert convergence == true "Inversion did not converge"
end

hybridinverse!(X, F, C::MapComponent, S::Storage; P::Parallel = serial) = hybridinverse!(X, F, C.I, S; P = P)

function hybridinverse!(X::Array{Float64,2}, F, L::LinMapComponent, S::Storage; P::Parallel = serial)
    # Pay attention that S is computed in the renormalized space for improve performance !!!
    transform!(L.L, X)
    hybridinverse!(X, F, L.C, S; P = P)
    itransform!(L.L, X)
end
