export projectGradient, line_search, projected_newton

# Write projected Newton method, according to
# Bertsekas, Dimitri P. "Projected Newton methods for optimization problems with simple constraints." SIAM Journal on control and Optimization 20.2 (1982): 221-246.


# Code inspired from Ricardo Baptista, Youssef Marzouk MIT
function projectGradient(x::Array{Float64,1}, G::Array{Float64,1})
    PG = copy(G);
    for i=1:size(PG,1)
        @inbounds begin
        if x[i]==0.0 && G[i] >=0
        PG[i] = 0.0
        end
        end
    end
    return PG
end

function linesearch(x::Array{Float64,1}, G::Array{Float64,1},
         p::Array{Float64,1}, J::Float64,Idx::Array{Int64,1}, Lhd::LHD)

    # Fixed parameters
    itmax = 15
    σ = 1e-4
    β = 2

    it = 0
    Jx_α_lin = copy(J)+1 # To enter the while loop
    α = copy(β)# The first iteration inside the while loop will be for α = 1
    α_p = copy(p)
    x_α = zero(x)
    while (J<Jx_α_lin) && (it<itmax)
        α /= β
        α_p .= copy(p)
        rmul!(α_p, copy(α))
        x_α .= max.(0.0, x - α_p)
        Jx_α = copy(Lhd(x_α, false, noutput = 1))
        α_p[Idx] .= x[Idx] - x_α[Idx]
        Jx_α_lin = Jx_α + σ*dot(G, α_p)
        it += 1
    end

    if (J<Jx_α_lin)
        print("Line search method reached maximum number of iterations")
        xh = copy(x)
        xh[Idx] .= 0.0
        index = Int64[]

        for i=1:size(x,1)
            @inbounds begin
            if (xh[i]>0.0) && (p[i]>0.0)
                push!(index, copy(i))
            end
            end
        end
        if isempty(index)
            α = 1.0
        else
            α = min(x[index]./p[index])
        end
    end
return α
end

function projected_newton(x0::Array{Float64,1}, Lhd::LHD, type::String)
# Default parameters
rtol_J = 1e-6;
rtol_G = 1e-6;
itmax = 30;
ϵ = 0.01;

n = size(x0,1)

for i=1:n
    @assert sign(x0[i])==1.0 "Initial conditions are not feasible"
end

x = copy(x0)
Lhd(x, true)
@get Lhd (G,H)

Jold =copy(Lhd.J[1])
J = copy(Lhd.J[1])

norm_PG0 = norm(projectGradient(x, G))
tol_G = norm_PG0*rtol_G

norm_PG = copy((norm_PG0))
rδJ = rtol_J + 1
it = 0

# Pre-allocate space
p =zero(x)
# h = zero(x)
z = zero(x)

## Iteration
while (rδJ > rtol_J) && (norm_PG > tol_G) && (it<itmax)
    # Define search direction
    wk = norm(x - max.(0.0, x-G))
    ϵk = min(ϵ, wk)
    Idx= Int64[]
    @inbounds for i=1:n
        if x[i] <= ϵk && G[i] >0.0
        push!(Idx, copy(i))
        end
    end

    if isempty(Idx)==false
        z .= zeros(n)
        @inbounds for i in Idx
            z[i] = copy(H[i,i])
        end
        # h .= deepcopy(diag(H))
        # z .= zeros(n)
        # z[Idx] .= h[Idx]
        H[Idx,:] .= 0.0
        H[:,Idx] .= 0.0
        H .+= diagm(z)
    end

    # Switch according to search direction
    if type =="TrueHessian"
    p .= copy(G)
    # Perform in-place Cholesky decomposition and store H\G into p
    LAPACK.posv!('U', H, p)
    elseif type =="ModifHessian"
        itmax = 100
        # Try if the TrueHessian technique works
        if isposdef(H)
            p .= copy(G)
            # Perform in-place Cholesky decomposition and store H\G into p
            LAPACK.posv!('U', H, p)
        else
        # else Strategy 1- flip negative eigenvalues
            if !issymmetric(H)
                H .+= H'
                rmul!(H, 0.5)
            end

            print("Hessian H is not positive-definite")
            # EH = eigen(Symmetric(H))
            EH = eigen(H)
            for i=1:size(EH.values,1)
                @inbounds EH.values[i] = copy( 1/(max(1e-8, abs(EH.values[i]))) )
            end

            # p .= deepcopy(G)
            mul!(p, EH.vectors', G)
            p .*= EH.values
            mul!(p, EH.vectors, copy(p))

            # p .= deepcopy(EH.vectors*( EH.values .* (EH.vectors'*G)))
        end
    elseif type =="Gradient"
        itmax = 1000
        p = copy(G)

    else
        error("Method not implemented yet")
    end

    # Perform line search
    α = linesearch(x,G,p,J,Idx,Lhd)
    # @show α

    # Update the estimate
    x .= copy(max.(0.0, x - α*p))
    # @show x
    Lhd(x, true)
    @get Lhd (G,H)
    J = copy(Lhd.J[1])

    # Convergence criteria
    rδJ = abs(J-Jold)/abs(Jold)
    # @show rδJ
    Jold = copy(J)
    norm_PG = copy(norm(projectGradient(x,G)))
    it +=1
end


if it>=itmax
    print("Max number of iterations is reached during the optimization")
end

return x, rδJ, norm_PG, it

end
