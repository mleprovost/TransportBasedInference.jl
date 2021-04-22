export bracket, invert_uk, invert_S

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

function invert_uk(u::uk, x, κ; z0::Real=0.0)
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
function invert(S::RadialMap, Sval, ystar, zplus)
    @get S (Nx, p, κ)
    ny = size(ystar,1)
    zplus[1:ny] = ystar
    # Recursive 1D root-finding
    # Start optimization from the a priori component
    @inbounds for i=ny+1:Nx
    Ui = S.U[i]
    uk_i = component(Ui, i)
    zplus[i] = invert_uk(uk_i, Sval[i] - off_diagonal(Ui, view(zplus,1:i-1)), κ)
    end
end

# y is the observation of size ny
function invert(S::SparseRadialMap, Sval, ystar, zplus)
    @get S (Nx, p, κ)
    ny = size(ystar,1)
    zplus[1:ny] .= ystar

    # Recursive 1D root-finding
    # Start optimization from the a priori component
    @inbounds for i=ny+1:Nx
    Ui = S.U[i]
    uk_i = component(Ui, i)
    zplus[i] = invert_uk(uk_i, Sval[i] - off_diagonal(Ui, view(zplus,1:i-1)), κ)
    end
end
