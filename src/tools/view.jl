export meas, viewmeas, state, viewstate

"""
    meas(X, Ny, Nx)

A function to extract the first `Ny` lines of the ensemble matrix `X`, typically storing the observations.
"""
meas(X, Ny, Nx) = X[1:Ny,:]

"""
    viewmeas(X, Ny, Nx)

Create a view of the first `Ny` lines of the ensemble matrix `X`, typically storing the observations.
"""
viewmeas(X, Ny, Nx) = view(X, 1:Ny,:)

"""
    state(X, Ny, Nx)

A function to extract the lines `Ny+1` to `Ny+Nx` of the ensemble matrix `X`, typically storing the state.
"""
state(X, Ny, Nx) = X[Ny+1:Nx+Ny,:]

"""
    viewstate(X, Ny, Nx)

Create a view of the lines `Ny+1` to `Ny+Nx` of the ensemble matrix `X`, typically storing the state.
"""
viewstate(X, Ny, Nx) = view(X, Ny+1:Nx+Ny,:)
