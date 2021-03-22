export omega, omega_unscaled


# Tools to compute the sparsity pattern

function omega(h, M, Y, X, Ny, Nx, ϵY, L::LinearTransform)
    Ne = size(X, 2)
    Omega = zeros(Nx+Ny, Nx+Ny)

    precisionY = (1/ϵY^2)
    sqprecisionY = precisionY^2

    # Fill block
    @showprogress for i=1:Ne
        Xi  = X[:, i]

        # Compute Jacobian of the forward model
        J = Jacobian(h, Xi, Ny, Nx)

        # Compute Hessian of the forward model
        H = Hessian(h, Xi, Ny, Nx)

        # Compute Hessian of the log pdf of the transport map
        Hlog = hess_x_log_pdf(M, X[:,i:i])

        # Compute Hessian of the log pdf of the transport map
        # Fill Ωxy block
        @inbounds for k=1:Ny
            for j=1:Nx
                Omega[Ny+j, k] += (J[k,j]*L.L.diag[k]*L.L.diag[Ny+j])^2
            end
        end

        # Fill Ωxx
        @inbounds for k=1:Nx
            for j=1:Nx
                Jj = view(J,:,j)
                Jk = view(J,:,k)
                Hjk = view(H,:,j,k)
                # @show L.L.diag[Ny+k]*L.L.diag[Ny+j]*precisionY*dot(Jj, Jk)
                # @show Hlog[1,j,k]
                # @show (L.L.diag[Ny+k]*L.L.diag[Ny+j])^2*dot(Hjk, Hjk)*precisionY
                Omega[Ny+j, Ny+k] += (-L.L.diag[Ny+k]*L.L.diag[Ny+j]*precisionY*dot(Jj, Jk) + Hlog[1,j,k])^2 +
                                  (L.L.diag[Ny+k]*L.L.diag[Ny+j])^2*dot(Hjk, Hjk)*precisionY
            end
        end
    end

    view(Omega, Ny+1:Ny+Nx, 1:Ny) .*= sqprecisionY
    view(Omega, 1:Ny, Ny+1:Ny+Nx) .= view(Omega, Ny+1:Ny+Nx, 1:Ny)'

    rmul!(Omega, 1.0/Ne)

    # Fill Ωyy with the identity
    @inbounds for i=1:Ny
        Omega[i,i] = 1.0
    end

    return Omega
end

function omega_unscaled(h, M, Y, X, Ny, Nx, ϵY, L::LinearTransform)
    Ne = size(X, 2)
    Ω = zeros(Nx+Ny, Nx+Ny)

    precisionY = (1/ϵY^2)
    sqprecisionY = precisionY^2

    # Fill block
    @showprogress for i=1:Ne
        Xi  = X[:, i]

        # Compute Jacobian of the forward model
        J = Jacobian(h, Xi, Ny, Nx)

        # Compute Hessian of the forward model
        H = Hessian(h, Xi, Ny, Nx)

        # Compute Hessian of the log pdf of the transport map
        Hlog = hess_x_log_pdf(M, X[:,i:i])

        # Compute Hessian of the log pdf of the transport map
        # Fill Ωxy block
        @inbounds for k=1:Ny
            for j=1:Nx
                Ω[Ny+j, k] += J[k,j]^2
            end
        end

        # Fill Ωxx
        @inbounds for k=1:Nx
            for j=1:Nx
                Jj = view(J,:,j)
                Jk = view(J,:,k)
                Hjk = view(H,j,k)
                Ω[Ny+j, Ny+k] += (-precisionY*dot(Jj, Jk) + 1/(L.L.diag[Ny+k]*L.L.diag[Ny+j])*Hlog[1,j,k])^2 +
                                 dot(Hjk, Hjk)*precisionY
            end
        end
    end

    view(Ω, Ny+1:Ny+Nx, 1:Ny) .*= sqprecisionY
    view(Ω, 1:Ny, Ny+1:Ny+Nx) .= view(Ω, Ny+1:Ny+Nx, 1:Ny)'

    rmul!(Ω, 1/Ne)

    # Fill Ωyy with the identity
    @inbounds for i=1:Ny
        Ω[i,i] = 1.0
    end

    return Ω
end
