export meas, viewmeas, state, viewstate


meas(X, Ny, Nx) = X[1:Ny,:]
viewmeas(X, Ny, Nx) = view(X, 1:Ny,:)

state(X, Ny, Nx) = X[Ny+1:Nx+Ny,:]
viewstate(X, Ny, Nx) = view(X, Ny+1:Nx+Ny,:)
