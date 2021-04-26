
@testset "Test integrated function Nx = 2" begin

Nx = 2
Ne = 100

X = randn(Nx, Ne)

Blist = [CstProHermite(8); CstPhyHermite(8); CstLinProHermite(8); CstLinPhyHermite(8)]
for b in Blist
    B = MultiBasis(b, Nx)

    idx = [0 0; 0 1; 1 0; 1 1; 1 2]
    truncidx = idx[1:2:end,:]
    Nψ = 5

    coeff = randn(Nψ)

    f = ExpandedFunction(B, idx, coeff)
    R = IntegratedFunction(f)

    # Test evaluate
    ψt = zeros(Ne)

    for i=1:Ne
        x = X[:,i]
        ψt[i] = R.f(vcat(x[1:end-1], 0.0)) +
                quadgk(t->R.g(ForwardDiff.gradient(y->R.f(y), vcat(x[1:end-1],t))[end]), 0, x[end])[1]
    end

    ψ = evaluate(R, X)

    @test norm(ψ - ψt)<5e-9

    # Test grad_x
    dψ = grad_x(R, X)

    function Gradient(x)
        return ForwardDiff.gradient(y->begin
                               y[end] = 0.0
                               return  R.f(y)
            end, x) + quadgk(t->ForwardDiff.gradient(
                      z->R.g(ForwardDiff.gradient(y->R.f(y), vcat(z[1:end-1],t))[end]),
                x), 0.0, x[end])[1] + vcat(zeros(size(x,1)-1), R.g(ForwardDiff.gradient(y->R.f(y), x)[end]))
    end

    for i=1:Ne
        xi = X[:,i]
        dψt = Gradient(xi)
        @test norm(dψ[i,:] - dψt)<1e-8
    end

    # Test hess_x
    H = hess_x(R, X)
    Ht = zeros(Nx, Nx)
    function Hessian!(H, x)
        fill!(H, 0.0)

        H[1:Nx-1,1:Nx-1] .= ForwardDiff.hessian(y->begin
                            y[Nx] = 0.0
                            return R.f(y) end, x)[1:Nx-1,1:Nx-1]

        # Hessian of the integral term
        H[1:Nx-1,1:Nx-1] .+= (quadgk(t -> ForwardDiff.hessian(
        z-> R.g(ForwardDiff.gradient(y->R.f(y), vcat(z[1:end-1], t))[end]), x),
                0.0, x[end], rtol = 1e-3)[1])[1:Nx-1,1:Nx-1]

        # H[Nx,:] and H[:,Nx]
        H[:, Nx] .= ForwardDiff.gradient(z->R.g(ForwardDiff.gradient(y->R.f(y), z)[end]), x)
        H[Nx, :] .= H[:, Nx]
        return H
    end

    @inbounds for i=1:Ne
        @test norm(H[i,:,:] - Hessian!(Ht, X[:,i]))<1e-8
    end

    # Test grad_xd
    gdψ =  grad_xd(R, X)

    gdψt = zeros(Ne)

    for i=1:Ne
        gdψt[i] = R.g(ForwardDiff.gradient(R.f, X[:,i])[end])
    end

    @test norm(gdψ - gdψt) <5e-9

    # Test grad_coeff_grad_xd
    dψ_xd_dc = grad_coeff_grad_xd(R, X)
    dψ_xd_dct = zeros(Ne, Nψ)

    for j=1:Nψ
        fj = MultiFunction(R.f.B, f.idx[j,:])
        ∂kfj(y) = ForwardDiff.gradient(fj, y)[end]
        ∂kf(y) = ForwardDiff.gradient(R.f, y)[end]
        for i=1:Ne
            dψ_xd_dct[i,j] = ∂kfj(X[:,i])*grad_x(R.g, ∂kf(X[:,i]))
        end
    end

    @test norm(dψ_xd_dc - dψ_xd_dct)<5e-9

    # Test hess_coeff_grad_xd
    d2ψ_xd_dc = hess_coeff_grad_xd(R, X)
    d2ψ_xd_dct = zeros(Ne, Nψ, Nψ)

    for j=1:Nψ
        fj = MultiFunction(R.f.B, f.idx[j,:])
        ∂kfj(y) = ForwardDiff.gradient(fj, y)[end]
        ∂kf(y) = ForwardDiff.gradient(R.f, y)[end]
        for l=1:Nψ
        fl = MultiFunction(R.f.B, f.idx[l,:])
        ∂kfl(y) = ForwardDiff.gradient(fl, y)[end]

            for i=1:Ne
                d2ψ_xd_dct[i,j,l] = 0.0*grad_x(R.g, ∂kf(X[:,i])) +
                    hess_x(R.g, ∂kf(X[:,i]))*∂kfj(X[:,i])*∂kfl(X[:,i])
            end
        end
    end

    @test norm(d2ψ_xd_dc - d2ψ_xd_dct)<5e-9



    # Test integrate_xd
    intψt = zeros(Ne)

    for i=1:Ne
        xi = X[:,i]
        intψt[i] = quadgk(t->R.g(ForwardDiff.gradient(y->R.f(y), vcat(xi[1:end-1],t))[end]), 0, xi[end],rtol = 1e-4)[1]
    end

    intψ = integrate_xd(R, X)

    @test norm(intψt - intψ)<5e-9


    # Test grad_coeff_integrate_xd

    dcintψt = zeros(Ne, Nψ)
    xi = zeros(Nx)
    for j=1:Nψ
        fj = MultiFunction(R.f.B, f.idx[j,:])
        ∂kfj(y) = ForwardDiff.gradient(fj, y)[end]
        ∂kf(y) = ForwardDiff.gradient(R.f, y)[end]
        for i=1:Ne
            xi .= X[:,i]
            dcintψt[i,j] = quadgk(t->∂kfj(vcat(xi[1:end-1],t))*grad_x(R.g, ∂kf(vcat(xi[1:end-1],t))),0,xi[end])[1]
        end
    end


    dcintψ = grad_coeff_integrate_xd(R, X)
    @test norm(dcintψ - dcintψt)<1e-6


    # Test hess_coeff_integrate_xd
    d2cintψt = zeros(Ne, Nψ, Nψ)
    xi = zeros(Nx)
    ∂kf(y) = ForwardDiff.gradient(R.f, y)[end]
    for j=1:Nψ
        fj = MultiFunction(R.f.B, f.idx[j,:])
        ∂kfj(y) = ForwardDiff.gradient(fj, y)[end]

        for l=1:Nψ

            fl = MultiFunction(R.f.B, f.idx[l,:])
            ∂kfl(y) = ForwardDiff.gradient(fl, y)[end]

            for i=1:Ne
                xi .= X[:,i]
    #                 @show quadgk(t->∂kfj(vcat(xi[1:end-1],t))*∂kfl(vcat(xi[1:end-1],t))*hess_x(R.g, ∂kf(vcat(xi[1:end-1],t))),0,xi[end])[1]
                    d2cintψt[i,j,l] = quadgk(t->∂kfj(vcat(xi[1:end-1],t))*∂kfl(vcat(xi[1:end-1],t))*hess_x(R.g, ∂kf(vcat(xi[1:end-1],t))),0,xi[end])[1]
            end
        end
    end

    d2cintψ = hess_coeff_integrate_xd(R, X)

    @test norm(d2cintψt - d2cintψ)<5e-9

    # Test grad_coeff
    dcRt = zeros(Ne, Nψ)

    X0 = deepcopy(X)
    X0[end,:] .= zeros(Ne)
    # ∂_c f(x_{1:k-1},0)
    dcRt += evaluate_basis(R.f, X0)
    xi = zeros(Nx)

    for j=1:Nψ
        fj = MultiFunction(R.f.B, f.idx[j,:])
        ∂kfj(y) = ForwardDiff.gradient(fj, y)[end]
        ∂kf(y) = ForwardDiff.gradient(R.f, y)[end]
        for i=1:Ne
            xi .= X[:,i]
            dcRt[i,j] += quadgk(t->∂kfj(vcat(xi[1:end-1],t))*grad_x(R.g, ∂kf(vcat(xi[1:end-1],t))),0,xi[end])[1]

        end
    end

    dcR = grad_coeff(R, X)

    @test norm(dcRt - dcR)<1e-6


    # Test hess_coeff
    d2cRt = zeros(Ne, Nψ, Nψ)

    xi = zeros(Nx)

    for j=1:Nψ
        fj = MultiFunction(R.f.B, f.idx[j,:])
        ∂kfj(y) = ForwardDiff.gradient(fj, y)[end]

        for l=1:Nψ

            fl = MultiFunction(R.f.B, f.idx[l,:])
            ∂kfl(y) = ForwardDiff.gradient(fl, y)[end]

            for i=1:Ne
                xi .= X[:,i]
    #                 @show quadgk(t->∂kfj(vcat(xi[1:end-1],t))*∂kfl(vcat(xi[1:end-1],t))*hess_x(R.g, ∂kf(vcat(xi[1:end-1],t))),0,xi[end])[1]
                    d2cRt[i,j,l] = quadgk(t->∂kfj(vcat(xi[1:end-1],t))*∂kfl(vcat(xi[1:end-1],t))*hess_x(R.g, ∂kf(vcat(xi[1:end-1],t))),0,xi[end])[1]
            end
        end
    end

    d2cR = hess_coeff(R, X)

    @test norm(d2cRt - d2cR)<1e-8

    end


end



@testset "Test integrated function Nx = 3" begin

    Nx = 3
    Ne = 100

    X = randn(Nx, Ne)


    Blist = [CstProHermite(8); CstPhyHermite(8); CstLinProHermite(8); CstLinPhyHermite(8)]
    for b in Blist
        B = MultiBasis(b, Nx)

        # idx = [0 0; 0 1; 1 0; 1 1; 1 2]
        idx = [0 0 0 ;0  0 1; 0 1 0; 1 0 0;1 1 0; 0 1 1; 1 0 1; 1 1 1; 1 2 0; 2 1 0]

        truncidx = idx[1:2:end,:]
        Nψ = 10

        coeff = randn(Nψ)

        f = ExpandedFunction(B, idx, coeff)
        R = IntegratedFunction(f)

        # Test grad_x
        dψ = grad_x(R, X)

        function Gradient(x)
            return ForwardDiff.gradient(y->begin
                                   y[end] = 0.0
                                   return  R.f(y)
                end, x) + quadgk(t->ForwardDiff.gradient(
                          z->R.g(ForwardDiff.gradient(y->R.f(y), vcat(z[1:end-1],t))[end]),
                    x), 0.0, x[end])[1] + vcat(zeros(size(x,1)-1), R.g(ForwardDiff.gradient(y->R.f(y), x)[end]))
        end

        for i=1:Ne
            xi = X[:,i]
            dψt = Gradient(xi)
            @test norm(dψ[i,:] - dψt)<1e-8
        end


        # Test grad_xd
        gdψ =  grad_xd(R, X)

        gdψt = zeros(Ne)

        for i=1:Ne
            gdψt[i] = R.g(ForwardDiff.gradient(R.f, X[:,i])[end])
        end

        @test norm(gdψ - gdψt) <5e-9

        # Test grad_coeff_grad_xd
        dψ_xd_dc = grad_coeff_grad_xd(R, X)
        dψ_xd_dct = zeros(Ne, Nψ)

        for j=1:Nψ
            fj = MultiFunction(R.f.B, f.idx[j,:])
            ∂kfj(y) = ForwardDiff.gradient(fj, y)[end]
            ∂kf(y) = ForwardDiff.gradient(R.f, y)[end]
            for i=1:Ne
                dψ_xd_dct[i,j] = ∂kfj(X[:,i])*grad_x(R.g, ∂kf(X[:,i]))
            end
        end

        @test norm(dψ_xd_dc - dψ_xd_dct)<5e-9

        # Test hess_coeff_grad_xd
        d2ψ_xd_dc = hess_coeff_grad_xd(R, X)
        d2ψ_xd_dct = zeros(Ne, Nψ, Nψ)

        for j=1:Nψ
            fj = MultiFunction(R.f.B, f.idx[j,:])
            ∂kfj(y) = ForwardDiff.gradient(fj, y)[end]
            ∂kf(y) = ForwardDiff.gradient(R.f, y)[end]
            for l=1:Nψ
            fl = MultiFunction(R.f.B, f.idx[l,:])
            ∂kfl(y) = ForwardDiff.gradient(fl, y)[end]

                for i=1:Ne
                    d2ψ_xd_dct[i,j,l] = 0.0*grad_x(R.g, ∂kf(X[:,i])) +
                        hess_x(R.g, ∂kf(X[:,i]))*∂kfj(X[:,i])*∂kfl(X[:,i])
                end
            end
        end

        @test norm(d2ψ_xd_dc - d2ψ_xd_dct)<5e-9



        # Test integrate_xd
        intψt = zeros(Ne)

        for i=1:Ne
            xi = X[:,i]
            intψt[i] = quadgk(t->R.g(ForwardDiff.gradient(y->R.f(y), vcat(xi[1:end-1],t))[end]), 0, xi[end],rtol = 1e-4)[1]
        end

        intψ = integrate_xd(R, X)

        @test norm(intψt - intψ)<5e-9


        # Test grad_coeff_integrate_xd

        dcintψt = zeros(Ne, Nψ)
        xi = zeros(Nx)
        for j=1:Nψ
            fj = MultiFunction(R.f.B, f.idx[j,:])
            ∂kfj(y) = ForwardDiff.gradient(fj, y)[end]
            ∂kf(y) = ForwardDiff.gradient(R.f, y)[end]
            for i=1:Ne
                xi .= X[:,i]
                dcintψt[i,j] = quadgk(t->∂kfj(vcat(xi[1:end-1],t))*grad_x(R.g, ∂kf(vcat(xi[1:end-1],t))),0,xi[end])[1]
            end
        end


        dcintψ = grad_coeff_integrate_xd(R, X)
        @test norm(dcintψ - dcintψt)<1e-6


        # Test hess_coeff_integrate_xd
        d2cintψt = zeros(Ne, Nψ, Nψ)
        xi = zeros(Nx)
        ∂kf(y) = ForwardDiff.gradient(R.f, y)[end]
        for j=1:Nψ
            fj = MultiFunction(R.f.B, f.idx[j,:])
            ∂kfj(y) = ForwardDiff.gradient(fj, y)[end]

            for l=1:Nψ

                fl = MultiFunction(R.f.B, f.idx[l,:])
                ∂kfl(y) = ForwardDiff.gradient(fl, y)[end]

                for i=1:Ne
                    xi .= X[:,i]
        #                 @show quadgk(t->∂kfj(vcat(xi[1:end-1],t))*∂kfl(vcat(xi[1:end-1],t))*hess_x(R.g, ∂kf(vcat(xi[1:end-1],t))),0,xi[end])[1]
                        d2cintψt[i,j,l] = quadgk(t->∂kfj(vcat(xi[1:end-1],t))*∂kfl(vcat(xi[1:end-1],t))*hess_x(R.g, ∂kf(vcat(xi[1:end-1],t))),0,xi[end])[1]
                end
            end
        end

        d2cintψ = hess_coeff_integrate_xd(R, X)

        @test norm(d2cintψt - d2cintψ)<5e-9

        # Test grad_coeff
        dcRt = zeros(Ne, Nψ)

        X0 = deepcopy(X)
        X0[end,:] .= zeros(Ne)
        # ∂_c f(x_{1:k-1},0)
        dcRt += evaluate_basis(R.f, X0)
        xi = zeros(Nx)

        for j=1:Nψ
            fj = MultiFunction(R.f.B, f.idx[j,:])
            ∂kfj(y) = ForwardDiff.gradient(fj, y)[end]
            ∂kf(y) = ForwardDiff.gradient(R.f, y)[end]
            for i=1:Ne
                xi .= X[:,i]
                dcRt[i,j] += quadgk(t->∂kfj(vcat(xi[1:end-1],t))*grad_x(R.g, ∂kf(vcat(xi[1:end-1],t))),0,xi[end])[1]

            end
        end

        dcR = grad_coeff(R, X)

        @test norm(dcRt - dcR)<1e-6


        # Test hess_coeff
        d2cRt = zeros(Ne, Nψ, Nψ)

        xi = zeros(Nx)

        for j=1:Nψ
            fj = MultiFunction(R.f.B, f.idx[j,:])
            ∂kfj(y) = ForwardDiff.gradient(fj, y)[end]

            for l=1:Nψ

                fl = MultiFunction(R.f.B, f.idx[l,:])
                ∂kfl(y) = ForwardDiff.gradient(fl, y)[end]

                for i=1:Ne
                    xi .= X[:,i]
        #                 @show quadgk(t->∂kfj(vcat(xi[1:end-1],t))*∂kfl(vcat(xi[1:end-1],t))*hess_x(R.g, ∂kf(vcat(xi[1:end-1],t))),0,xi[end])[1]
                        d2cRt[i,j,l] = quadgk(t->∂kfj(vcat(xi[1:end-1],t))*∂kfl(vcat(xi[1:end-1],t))*hess_x(R.g, ∂kf(vcat(xi[1:end-1],t))),0,xi[end])[1]
                end
            end
        end

        d2cR = hess_coeff(R, X)

        @test norm(d2cRt - d2cR)<1e-8
    end
end



@testset "Test integrated function Nx = 4" begin

    Nx = 4
    Ne = 100

    X = randn(Nx, Ne)

    Blist = [CstProHermite(8); CstPhyHermite(8); CstLinProHermite(8); CstLinPhyHermite(8)]
    for b in Blist
        B = MultiBasis(b, Nx)

        # idx = [0 0; 0 1; 1 0; 1 1; 1 2]
        idx = [0 0 0 0; 0 2 0 1; 0 2 3 0; 2 0 2 1; 0 0 1 2; 1 2 0 2;3 2 2 2]

        truncidx = idx[1:2:end,:]
        Nψ = 7

        coeff = randn(Nψ)
        f = ExpandedFunction(B, idx, coeff)
        R = IntegratedFunction(f)

        # Test grad_x
        dψ = grad_x(R, X)

        function Gradient(x)
            return ForwardDiff.gradient(y->begin
                                   y[end] = 0.0
                                   return  R.f(y)
                end, x) + quadgk(t->ForwardDiff.gradient(
                          z->R.g(ForwardDiff.gradient(y->R.f(y), vcat(z[1:end-1],t))[end]),
                    x), 0.0, x[end])[1] + vcat(zeros(size(x,1)-1), R.g(ForwardDiff.gradient(y->R.f(y), x)[end]))
        end

        for i=1:Ne
            xi = X[:,i]
            dψt = Gradient(xi)
            @test norm(dψ[i,:] - dψt)<1e-8
        end

        # Test hess_x
        H = hess_x(R, X)
        Ht = zeros(Nx, Nx)
        function Hessian!(H, x)
            fill!(H, 0.0)

            H[1:Nx-1,1:Nx-1] .= ForwardDiff.hessian(y->begin
                                y[Nx] = 0.0
                                return R.f(y) end, x)[1:Nx-1,1:Nx-1]

            # Hessian of the integral term
            H[1:Nx-1,1:Nx-1] .+= (quadgk(t -> ForwardDiff.hessian(
            z-> R.g(ForwardDiff.gradient(y->R.f(y), vcat(z[1:end-1], t))[end]), x),
                    0.0, x[end], rtol = 1e-3)[1])[1:Nx-1,1:Nx-1]

            # H[Nx,:] and H[:,Nx]
            H[:, Nx] .= ForwardDiff.gradient(z->R.g(ForwardDiff.gradient(y->R.f(y), z)[end]), x)
            H[Nx, :] .= H[:, Nx]
            return H
        end

        @inbounds for i=1:Ne
            @test norm(H[i,:,:] - Hessian!(Ht, X[:,i]))<1e-8
        end

        # Test grad_xd
        gdψ =  grad_xd(R, X)

        gdψt = zeros(Ne)

        for i=1:Ne
            gdψt[i] = R.g(ForwardDiff.gradient(R.f, X[:,i])[end])
        end

        @test norm(gdψ - gdψt) <5e-9

        # Test grad_coeff_grad_xd
        dψ_xd_dc = grad_coeff_grad_xd(R, X)
        dψ_xd_dct = zeros(Ne, Nψ)

        for j=1:Nψ
            fj = MultiFunction(R.f.B, f.idx[j,:])
            ∂kfj(y) = ForwardDiff.gradient(fj, y)[end]
            ∂kf(y) = ForwardDiff.gradient(R.f, y)[end]
            for i=1:Ne
                dψ_xd_dct[i,j] = ∂kfj(X[:,i])*grad_x(R.g, ∂kf(X[:,i]))
            end
        end

        @test norm(dψ_xd_dc - dψ_xd_dct)<5e-9

        # Test hess_coeff_grad_xd
        d2ψ_xd_dc = hess_coeff_grad_xd(R, X)
        d2ψ_xd_dct = zeros(Ne, Nψ, Nψ)

        for j=1:Nψ
            fj = MultiFunction(R.f.B, f.idx[j,:])
            ∂kfj(y) = ForwardDiff.gradient(fj, y)[end]
            ∂kf(y) = ForwardDiff.gradient(R.f, y)[end]
            for l=1:Nψ
            fl = MultiFunction(R.f.B, f.idx[l,:])
            ∂kfl(y) = ForwardDiff.gradient(fl, y)[end]

                for i=1:Ne
                    d2ψ_xd_dct[i,j,l] = 0.0*grad_x(R.g, ∂kf(X[:,i])) +
                        hess_x(R.g, ∂kf(X[:,i]))*∂kfj(X[:,i])*∂kfl(X[:,i])
                end
            end
        end

        @test norm(d2ψ_xd_dc - d2ψ_xd_dct)<5e-9

        # Test integrate_xd
        intψt = zeros(Ne)

        for i=1:Ne
            xi = X[:,i]
            intψt[i] = quadgk(t->R.g(ForwardDiff.gradient(y->R.f(y), vcat(xi[1:end-1],t))[end]), 0, xi[end],rtol = 1e-4)[1]
        end

        intψ = integrate_xd(R, X)

        @test norm(intψt - intψ)<5e-9


        # Test grad_coeff_integrate_xd

        dcintψt = zeros(Ne, Nψ)
        xi = zeros(Nx)
        for j=1:Nψ
            fj = MultiFunction(R.f.B, f.idx[j,:])
            ∂kfj(y) = ForwardDiff.gradient(fj, y)[end]
            ∂kf(y) = ForwardDiff.gradient(R.f, y)[end]
            for i=1:Ne
                xi .= X[:,i]
                dcintψt[i,j] = quadgk(t->∂kfj(vcat(xi[1:end-1],t))*grad_x(R.g, ∂kf(vcat(xi[1:end-1],t))),0,xi[end])[1]
            end
        end


        dcintψ = grad_coeff_integrate_xd(R, X)
        @test norm(dcintψ - dcintψt)<1e-6


        # Test hess_coeff_integrate_xd
        d2cintψt = zeros(Ne, Nψ, Nψ)
        xi = zeros(Nx)
        ∂kf(y) = ForwardDiff.gradient(R.f, y)[end]
        for j=1:Nψ
            fj = MultiFunction(R.f.B, f.idx[j,:])
            ∂kfj(y) = ForwardDiff.gradient(fj, y)[end]

            for l=1:Nψ

                fl = MultiFunction(R.f.B, f.idx[l,:])
                ∂kfl(y) = ForwardDiff.gradient(fl, y)[end]

                for i=1:Ne
                    xi .= X[:,i]
        #                 @show quadgk(t->∂kfj(vcat(xi[1:end-1],t))*∂kfl(vcat(xi[1:end-1],t))*hess_x(R.g, ∂kf(vcat(xi[1:end-1],t))),0,xi[end])[1]
                        d2cintψt[i,j,l] = quadgk(t->∂kfj(vcat(xi[1:end-1],t))*∂kfl(vcat(xi[1:end-1],t))*hess_x(R.g, ∂kf(vcat(xi[1:end-1],t))),0,xi[end])[1]
                end
            end
        end

        d2cintψ = hess_coeff_integrate_xd(R, X)

        @test norm(d2cintψt - d2cintψ)<5e-9

        # Test grad_coeff
        dcRt = zeros(Ne, Nψ)

        X0 = deepcopy(X)
        X0[end,:] .= zeros(Ne)
        # ∂_c f(x_{1:k-1},0)
        dcRt += evaluate_basis(R.f, X0)
        xi = zeros(Nx)

        for j=1:Nψ
            fj = MultiFunction(R.f.B, f.idx[j,:])
            ∂kfj(y) = ForwardDiff.gradient(fj, y)[end]
            ∂kf(y) = ForwardDiff.gradient(R.f, y)[end]
            for i=1:Ne
                xi .= X[:,i]
                dcRt[i,j] += quadgk(t->∂kfj(vcat(xi[1:end-1],t))*grad_x(R.g, ∂kf(vcat(xi[1:end-1],t))),0,xi[end])[1]

            end
        end

        dcR = grad_coeff(R, X)

        @test norm(dcRt - dcR)<1e-6


        # Test hess_coeff
        d2cRt = zeros(Ne, Nψ, Nψ)

        xi = zeros(Nx)

        for j=1:Nψ
            fj = MultiFunction(R.f.B, f.idx[j,:])
            ∂kfj(y) = ForwardDiff.gradient(fj, y)[end]

            for l=1:Nψ

                fl = MultiFunction(R.f.B, f.idx[l,:])
                ∂kfl(y) = ForwardDiff.gradient(fl, y)[end]

                for i=1:Ne
                    xi .= X[:,i]
        #                 @show quadgk(t->∂kfj(vcat(xi[1:end-1],t))*∂kfl(vcat(xi[1:end-1],t))*hess_x(R.g, ∂kf(vcat(xi[1:end-1],t))),0,xi[end])[1]
                        d2cRt[i,j,l] = quadgk(t->∂kfj(vcat(xi[1:end-1],t))*∂kfl(vcat(xi[1:end-1],t))*hess_x(R.g, ∂kf(vcat(xi[1:end-1],t))),0,xi[end])[1]
                end
            end
        end

        d2cR = hess_coeff(R, X)

        @test norm(d2cRt - d2cR)<1e-8
    end
end
