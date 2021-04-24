export bracket, inverse_uk, inverse, inverse!

#Expands the range of valued searched geometrically until a root is bracketed
# Julia's adaptation of zbrac Numerical Recipes in Fortran 77 p.345
function bracket(f, a, b)
    F = 1.6
    Niter = 100
    outa = deepcopy(a)
    outb = deepcopy(b)
    counter = 0
    # Root is not bracketed
    while f(outa)*f(outb)>0 && counter<Niter
        Δ = F*deepcopy(outa-outb)
        if abs(f(outa))<abs(f(outb))
        outa += Δ
        else
        outb += Δ
        end
        counter += 1
    end
    return outa, outb
    @assert counter<=(Niter-1) "Maximal number of iterations reached"
end

function inverse_uk(u::uk, x, κ; z0::Real=0.0)
    if u.p==-1
    # Identity transformation
        return x
    elseif u.p==0
    # Solve u(z)= az + b = x for z
        return (x-u.ak[1])/u.ak[2]
    else
        zlim = (u.ξk[1]-κ*u.σk[1], u.ξk[end]+κ*u.σk[end])
        #Ensure that the zero is bracketed
        zlim = bracket(z->u(z)-x, zlim[1], zlim[2])
        # Roots.Brent()
        return find_zero(z->u(z)-x, zlim, Roots.Brent())
    end
end

# y is the observation of size ny
function inverse(x::AbstractVector{Float64}, F, S::RadialMap, ystar)
    @get S (Nx, p, κ)
    ny = size(ystar,1)
    x[1:ny] .= ystar
    # Recursive 1D root-finding
    # Start optimization from the a priori component
    @inbounds for i=ny+1:Nx
        Ui = S.C[i]
        uk_i = component(Ui, i)
        x[i] = inverse_uk(uk_i, F[i] - off_diagonal(Ui, view(x,1:i-1)), κ)
    end
end

# y is the observation of size ny
function inverse(x::AbstractVector{Float64}, F, S::SparseRadialMap, ystar)
    @get S (Nx, p, κ)
    ny = size(ystar,1)
    x[1:ny] .= ystar

    # Recursive 1D root-finding
    # Start optimization from the a priori component
    @inbounds for i=ny+1:Nx
        Ui = S.C[i]
        uk_i = component(Ui, i)
        x[i] = inverse_uk(uk_i, F[i] - off_diagonal(Ui, view(x,1:i-1)), κ)
    end
end

function inverse!(X::Array{Float64,2}, F, S::SparseRadialMap, ystar::AbstractVector{Float64}; start::Int64=1)
    @get S (Nx, p, κ)
    Nx = S.Nx
    NxX, Ne = size(X)

    Ny = size(ystar,1)
    @assert NxX == Nx
    @assert 1 <= Ny < Nx
    @assert size(F) == (Nx, Ne)

    @view(X[1:Ny,:]) .= ystar

    # Recursive 1D root-finding
    # Start optimization from the a priori component
    @inbounds for i=Ny+1:Nx
        Ui = S.C[i]
        uk_i = component(Ui, i)
        for j=1:Ne
            X[i,j] = inverse_uk(uk_i, F[i,j] - off_diagonal(Ui, view(X,1:i-1,j)), κ)
        end
    end
end
