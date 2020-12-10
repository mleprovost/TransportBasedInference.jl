export viewmeas, viewstate

viewmeas(X, Ny, Nx)= view(X, 1:Ny, :)
viewstate(X, Ny, Nx)= view(X, Ny+1:Ny+Nx, :)
