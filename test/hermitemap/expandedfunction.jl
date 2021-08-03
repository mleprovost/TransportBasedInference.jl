using TransportBasedInference: evaluate

@testset "Test active_dim" begin

idx = reshape([0], (1, 1))
@test active_dim(idx) == [1]

idx = reshape([0; 1; 2; 3], (4, 1))
@test active_dim(idx) == [1]

idx = [0 0; 0 1; 1 0; 1 1; 1 2]
@test active_dim(idx) == [1; 2]


idx = [0 0 0]
@test active_dim(idx) == [3]

idx = [0 0 0 0; 0 2 0 1; 0 2 3 0; 0 2 2 1; 0 0 1 2; 0 2 0 2;0 2 2 2]
@test active_dim(idx) == [2;3;4]


idx = [5 0 0 0; 4 2 0 1; 3 2 3 0; 2 2 2 1; 2 0 1 2; 2 2 0 2;2 2 2 2]
@test active_dim(idx) == [1; 2;3;4]

end

@testset "Test evaluation, gradient and hessian of expanded function Nx = 1 I" begin

    Nx = 1
    Ne = 500
    X = randn(Nx, Ne)

    Blist = [CstProHermite(8); CstPhyHermite(8); CstLinProHermite(8); CstLinPhyHermite(8)]
    for b in Blist
        B = MultiBasis(b, Nx)

        idx = reshape([0], (1, 1))
        truncidx = idx[1:2:end,:]

        coeff =  randn(size(idx,1))

        f = ExpandedFunction(B, idx, coeff)

        ψt_basis, dψt_basis, d2ψt_basis = alleval(f, X)

        ψ_basis = evaluate_basis(f, X)
        dψ_basis = grad_x_basis(f, X)
        d2ψ_basis = hess_x_basis(f, X);
        @test norm(ψt_basis - ψ_basis)<1e-8
        @test norm(dψt_basis - dψ_basis)<1e-8
        @test norm(d2ψt_basis - d2ψ_basis)<1e-8


        # For truncated basis
        ψtrunc_basis = evaluate_basis(f, X, truncidx)
        dψtrunc_basis = grad_x_basis(f, X, truncidx)
        d2ψtrunc_basis = hess_x_basis(f, X, truncidx);

        @test norm(ψ_basis[:,1:2:end] - ψtrunc_basis)<1e-8
        @test norm(dψ_basis[:,1:2:end,:] - dψtrunc_basis)<1e-8
        @test norm(d2ψt_basis[:,1:2:end,:,:] - d2ψtrunc_basis)<1e-8


        # Verify function evaluation
        @test norm(map(i->f(X[:,i]),1:Ne) - evaluate(f,X))<1e-8

        #  Verify gradient
        dψ = grad_x(f, X)
        @test norm(hcat(map(i->ForwardDiff.gradient(f, X[:,i]), 1:Ne)...)' - dψ)<1e-8

        # Verify hessian
        d2ψ = hess_x(f, X)

        for i=1:Ne
            d2ψt = ForwardDiff.hessian(f, X[:,i])
            @test norm(d2ψt - d2ψ[i,:,:])<1e-8
        end


        # Verify grad_xd
        dψxd = grad_xd(f, X)

        for i=1:Ne
            dψxd_t = ForwardDiff.gradient(f, X[:,i])[end]
            @test abs(dψxd[i] - dψxd_t)<1e-10
        end

        # Verify hess_xd
        d2ψxd = hess_xd(f, X)

        for i=1:Ne
            d2ψxd_t = ForwardDiff.hessian(f, X[:,i])[end,end]
            @test abs(d2ψxd[i] - d2ψxd_t)<1e-10
        end

        # Verify grad_x_grad_xd
        dxdxkψ = grad_x_grad_xd(f, X)

        for i=1:Ne
            dxdxkψ_t = ForwardDiff.hessian(f, X[:,i])
            for j=1:Nx
                @test abs(dxdxkψ[i,j] - dxdxkψ_t[j,end])<1e-10
            end
        end

        # Verify hess_x_grad_xd
        dxidxjdxkψ = hess_x_grad_xd(f, X, f.idx)

        for i=1:Ne
            dxidxjdxkψ_t = ForwardDiff.hessian(xi ->ForwardDiff.gradient(f, xi)[Nx], X[:,i])
            @test norm(dxidxjdxkψ[i,:,:] - dxidxjdxkψ_t)<1e-10
        end

        # Verify grad_coeff
        dψcoeff  = grad_coeff(f, X)
        dψcoefftrunc  = grad_coeff(f, X, collect(1:2:size(idx,1)))
        @test norm(dψcoeff[:,1:2:end] - dψcoefftrunc)<1e-10

        @test norm(dψcoeff - ψ_basis)<1e-10
        @test norm(dψcoefftrunc - ψtrunc_basis)<1e-10

        # Verify hess_coeff
        d2ψcoeff  = hess_coeff(f, X)
        d2ψcoefftrunc  = hess_coeff(f, X, collect(1:2:size(idx,1)))

        @test norm(d2ψcoeff)<1e-10

        # Verify grad_coeff_grad_xd
        dψcoeffxd = grad_coeff_grad_xd(f, X)
        dψcoeffxdtrunc = grad_coeff_grad_xd(f, X, collect(1:2:size(idx,1)))

        @test norm(dψcoeffxd[:,1:2:end] - dψcoeffxdtrunc)<1e-10
        @test norm(dψcoeffxd - grad_xk_basis(f, X, 1, Nx))<1e-10
        @test norm(dψcoeffxdtrunc - grad_xk_basis(f, X, 1, Nx, f.idx[collect(1:2:size(idx,1)),:]))<1e-10

        # Verify hess_coeff_grad_xd
        d2ψcoeffxd = hess_coeff_grad_xd(f, X)
        d2ψcoeffxdtrunc = hess_coeff_grad_xd(f, X, collect(1:2:size(idx,1)))

        @test norm(d2ψcoeffxd)<1e-10
        @test norm(d2ψcoeffxdtrunc)<1e-10
    end
end

@testset "Test evaluation, gradient and hessian of expanded function Nx = 1 II" begin

    Nx = 1
    Ne = 500
    X = randn(Nx, Ne)

    Blist = [CstProHermite(8); CstPhyHermite(8); CstLinProHermite(8); CstLinPhyHermite(8)]
    for b in Blist
        B = MultiBasis(b, Nx)

        idx = reshape([0; 1; 2; 3], (4, 1))
        truncidx = idx[1:2:end,:]

        coeff =  randn(size(idx,1))

        f = ExpandedFunction(B, idx, coeff)



        ψt_basis, dψt_basis, d2ψt_basis = alleval(f, X)

        ψ_basis = evaluate_basis(f, X)
        dψ_basis = grad_x_basis(f, X)
        d2ψ_basis = hess_x_basis(f, X);
        @test norm(ψt_basis - ψ_basis)<1e-8
        @test norm(dψt_basis - dψ_basis)<1e-8
        @test norm(d2ψt_basis - d2ψ_basis)<1e-8


        # For truncated basis
        ψtrunc_basis = evaluate_basis(f, X, truncidx)
        dψtrunc_basis = grad_x_basis(f, X, truncidx)
        d2ψtrunc_basis = hess_x_basis(f, X, truncidx);

        @test norm(ψ_basis[:,1:2:end] - ψtrunc_basis)<1e-8
        @test norm(dψ_basis[:,1:2:end,:] - dψtrunc_basis)<1e-8
        @test norm(d2ψt_basis[:,1:2:end,:,:] - d2ψtrunc_basis)<1e-8


        # Verify function evaluation
        @test norm(map(i->f(X[:,i]),1:Ne) - evaluate(f, X))<1e-8

        #  Verify gradient
        dψ = grad_x(f, X)
        @test norm(hcat(map(i->ForwardDiff.gradient(f, X[:,i]), 1:Ne)...)' - dψ)<1e-8

        # Verify hessian
        d2ψ = hess_x(f, X)

        for i=1:Ne
            d2ψt = ForwardDiff.hessian(f, X[:,i])
            @test norm(d2ψt - d2ψ[i,:,:])<1e-8
        end


        # Verify grad_xd
        dψxd = grad_xd(f, X)

        for i=1:Ne
            dψxd_t = ForwardDiff.gradient(f, X[:,i])[end]
            @test abs(dψxd[i] - dψxd_t)<1e-10
        end

        # Verify hess_xd
        d2ψxd = hess_xd(f, X)

        for i=1:Ne
            d2ψxd_t = ForwardDiff.hessian(f, X[:,i])[end,end]
            @test abs(d2ψxd[i] - d2ψxd_t)<1e-10
        end

        # Verify grad_x_grad_xd
        dxdxkψ = grad_x_grad_xd(f, X)

        for i=1:Ne
            dxdxkψ_t = ForwardDiff.hessian(f, X[:,i])
            for j=1:Nx
                @test abs(dxdxkψ[i,j] - dxdxkψ_t[j,end])<1e-10
            end
        end

        # Verify hess_x_grad_xd
        dxidxjdxkψ = hess_x_grad_xd(f, X, f.idx)

        for i=1:Ne
            dxidxjdxkψ_t = ForwardDiff.hessian(xi ->ForwardDiff.gradient(f, xi)[Nx], X[:,i])
            @test norm(dxidxjdxkψ[i,:,:] - dxidxjdxkψ_t)<1e-10
        end

        # Verify grad_coeff
        dψcoeff  = grad_coeff(f, X)
        dψcoefftrunc  = grad_coeff(f, X, collect(1:2:size(idx,1)))
        @test norm(dψcoeff[:,1:2:end] - dψcoefftrunc)<1e-10

        @test norm(dψcoeff - ψ_basis)<1e-10
        @test norm(dψcoefftrunc - ψtrunc_basis)<1e-10

        # Verify hess_coeff
        d2ψcoeff  = hess_coeff(f, X)
        d2ψcoefftrunc  = hess_coeff(f, X, collect(1:2:size(idx,1)))

        @test norm(d2ψcoeff)<1e-10

        # Verify grad_coeff_grad_xd
        dψcoeffxd = grad_coeff_grad_xd(f, X)
        dψcoeffxdtrunc = grad_coeff_grad_xd(f, X, collect(1:2:size(idx,1)))

        @test norm(dψcoeffxd[:,1:2:end] - dψcoeffxdtrunc)<1e-10
        @test norm(dψcoeffxd - grad_xk_basis(f, X, 1, Nx))<1e-10
        @test norm(dψcoeffxdtrunc - grad_xk_basis(f, X, 1, Nx, f.idx[collect(1:2:size(idx,1)),:]))<1e-10

        # Verify hess_coeff_grad_xd
        d2ψcoeffxd = hess_coeff_grad_xd(f, X)
        d2ψcoeffxdtrunc = hess_coeff_grad_xd(f, X, collect(1:2:size(idx,1)))

        @test norm(d2ψcoeffxd)<1e-10
        @test norm(d2ψcoeffxdtrunc)<1e-10
    end
end

@testset "Test evaluation, gradient and hessian of expanded function Nx = 2" begin

    Nx = 2
    Ne = 500
    X = randn(Nx, Ne)

    Blist = [CstProHermite(8); CstPhyHermite(8); CstLinProHermite(8); CstLinPhyHermite(8)]
    for b in Blist
        B = MultiBasis(b, Nx)


        idx = [0 0]
        truncidx = idx[1:2:end,:]

        coeff =  randn(size(idx,1))

        f = ExpandedFunction(B, idx, coeff)


        ψt_basis, dψt_basis, d2ψt_basis = alleval(f, X)

        ψ_basis = evaluate_basis(f, X)
        dψ_basis = grad_x_basis(f, X)
        d2ψ_basis = hess_x_basis(f, X);
        @test norm(ψt_basis - ψ_basis)<1e-8
        @test norm(dψt_basis - dψ_basis)<1e-8
        @test norm(d2ψt_basis - d2ψ_basis)<1e-8


        # For truncated basis
        ψtrunc_basis = evaluate_basis(f, X, truncidx)
        dψtrunc_basis = grad_x_basis(f, X, truncidx)
        d2ψtrunc_basis = hess_x_basis(f, X, truncidx);

        @test norm(ψ_basis[:,1:2:end] - ψtrunc_basis)<1e-8
        @test norm(dψ_basis[:,1:2:end,:] - dψtrunc_basis)<1e-8
        @test norm(d2ψt_basis[:,1:2:end,:,:] - d2ψtrunc_basis)<1e-8


        # Verify function evaluation
        @test norm(map(i->f(X[:,i]),1:Ne) - evaluate(f, X))<1e-8

        #  Verify gradient
        dψ = grad_x(f, X)
        @test norm(hcat(map(i->ForwardDiff.gradient(f, X[:,i]), 1:Ne)...)' - dψ)<1e-8

        # Verify hessian
        d2ψ = hess_x(f, X)

        for i=1:Ne
            d2ψt = ForwardDiff.hessian(f, X[:,i])
            @test norm(d2ψt - d2ψ[i,:,:])<1e-8
        end


        # Verify grad_xd
        dψxd = grad_xd(f, X)

        for i=1:Ne
            dψxd_t = ForwardDiff.gradient(f, X[:,i])[end]
            @test abs(dψxd[i] - dψxd_t)<1e-10
        end

        # Verify hess_xd
        d2ψxd = hess_xd(f, X)

        for i=1:Ne
            d2ψxd_t = ForwardDiff.hessian(f, X[:,i])[end,end]
            @test abs(d2ψxd[i] - d2ψxd_t)<1e-10
        end

        # Verify grad_x_grad_xd
        dxdxkψ = grad_x_grad_xd(f, X)

        for i=1:Ne
            dxdxkψ_t = ForwardDiff.hessian(f, X[:,i])
            for j=1:Nx
                @test abs(dxdxkψ[i,j] - dxdxkψ_t[j,end])<1e-10
            end
        end

        # Verify hess_x_grad_xd
        dxidxjdxkψ = hess_x_grad_xd(f, X, f.idx)

        for i=1:Ne
            dxidxjdxkψ_t = ForwardDiff.hessian(xi ->ForwardDiff.gradient(f, xi)[Nx], X[:,i])
            @test norm(dxidxjdxkψ[i,:,:] - dxidxjdxkψ_t)<1e-10
        end

        # Verify grad_coeff
        dψcoeff  = grad_coeff(f, X)
        dψcoefftrunc  = grad_coeff(f, X, collect(1:2:size(idx,1)))
        @test norm(dψcoeff[:,1:2:end] - dψcoefftrunc)<1e-10

        @test norm(dψcoeff - ψ_basis)<1e-10
        @test norm(dψcoefftrunc - ψtrunc_basis)<1e-10

        # Verify hess_coeff
        d2ψcoeff  = hess_coeff(f, X)
        d2ψcoefftrunc  = hess_coeff(f, X, collect(1:2:size(idx,1)))

        @test norm(d2ψcoeff)<1e-10

        # Verify grad_coeff_grad_xd
        dψcoeffxd = grad_coeff_grad_xd(f, X)
        dψcoeffxdtrunc = grad_coeff_grad_xd(f, X, collect(1:2:size(idx,1)))

        @test norm(dψcoeffxd[:,1:2:end] - dψcoeffxdtrunc)<1e-10
        @test norm(dψcoeffxd - grad_xk_basis(f, X, 1, Nx))<1e-10
        @test norm(dψcoeffxdtrunc - grad_xk_basis(f, X, 1, Nx, f.idx[collect(1:2:size(idx,1)),:]))<1e-10

        # Verify hess_coeff_grad_xd
        d2ψcoeffxd = hess_coeff_grad_xd(f, X)
        d2ψcoeffxdtrunc = hess_coeff_grad_xd(f, X, collect(1:2:size(idx,1)))

        @test norm(d2ψcoeffxd)<1e-10
        @test norm(d2ψcoeffxdtrunc)<1e-10
    end
end

@testset "Test evaluation, gradient and hessian of expanded function Nx = 2" begin

    Nx = 2
    Ne = 500
    X = randn(Nx, Ne)

    Blist = [CstProHermite(8); CstPhyHermite(8); CstLinProHermite(8); CstLinPhyHermite(8)]
    for b in Blist
        B = MultiBasis(b, Nx)


        idx = [0 0; 0 1; 1 0; 1 1; 1 2]
        truncidx = idx[1:2:end,:]

        coeff =  randn(size(idx,1))

        f = ExpandedFunction(B, idx, coeff)



        ψt_basis, dψt_basis, d2ψt_basis = alleval(f, X)

        ψ_basis = evaluate_basis(f, X)
        dψ_basis = grad_x_basis(f, X)
        d2ψ_basis = hess_x_basis(f, X);
        @test norm(ψt_basis - ψ_basis)<1e-8
        @test norm(dψt_basis - dψ_basis)<1e-8
        @test norm(d2ψt_basis - d2ψ_basis)<1e-8


        # For truncated basis
        ψtrunc_basis = evaluate_basis(f, X, truncidx)
        dψtrunc_basis = grad_x_basis(f, X, truncidx)
        d2ψtrunc_basis = hess_x_basis(f, X, truncidx);

        @test norm(ψ_basis[:,1:2:end] - ψtrunc_basis)<1e-8
        @test norm(dψ_basis[:,1:2:end,:] - dψtrunc_basis)<1e-8
        @test norm(d2ψt_basis[:,1:2:end,:,:] - d2ψtrunc_basis)<1e-8


        # Verify function evaluation
        @test norm(map(i->f(X[:,i]),1:Ne) - evaluate(f, X))<1e-8

        #  Verify gradient
        dψ = grad_x(f, X)
        @test norm(hcat(map(i->ForwardDiff.gradient(f, X[:,i]), 1:Ne)...)' - dψ)<1e-8

        # Verify hessian
        d2ψ = hess_x(f, X)

        for i=1:Ne
            d2ψt = ForwardDiff.hessian(f, X[:,i])
            @test norm(d2ψt - d2ψ[i,:,:])<1e-8
        end


        # Verify grad_xd
        dψxd = grad_xd(f, X)

        for i=1:Ne
            dψxd_t = ForwardDiff.gradient(f, X[:,i])[end]
            @test abs(dψxd[i] - dψxd_t)<1e-10
        end

        # Verify hess_xd
        d2ψxd = hess_xd(f, X)

        for i=1:Ne
            d2ψxd_t = ForwardDiff.hessian(f, X[:,i])[end,end]
            @test abs(d2ψxd[i] - d2ψxd_t)<1e-10
        end

        # Verify grad_x_grad_xd
        dxdxkψ = grad_x_grad_xd(f, X)

        for i=1:Ne
            dxdxkψ_t = ForwardDiff.hessian(f, X[:,i])
            for j=1:Nx
                @test abs(dxdxkψ[i,j] - dxdxkψ_t[j,end])<1e-10
            end
        end

        # Verify hess_x_grad_xd
        dxidxjdxkψ = hess_x_grad_xd(f, X, f.idx)

        for i=1:Ne
            dxidxjdxkψ_t = ForwardDiff.hessian(xi ->ForwardDiff.gradient(f, xi)[Nx], X[:,i])
            @test norm(dxidxjdxkψ[i,:,:] - dxidxjdxkψ_t)<1e-10
        end

        # Verify grad_coeff
        dψcoeff  = grad_coeff(f, X)
        dψcoefftrunc  = grad_coeff(f, X, collect(1:2:size(idx,1)))
        @test norm(dψcoeff[:,1:2:end] - dψcoefftrunc)<1e-10

        @test norm(dψcoeff - ψ_basis)<1e-10
        @test norm(dψcoefftrunc - ψtrunc_basis)<1e-10

        # Verify hess_coeff
        d2ψcoeff  = hess_coeff(f, X)
        d2ψcoefftrunc  = hess_coeff(f, X, collect(1:2:size(idx,1)))

        @test norm(d2ψcoeff)<1e-10

        # Verify grad_coeff_grad_xd
        dψcoeffxd = grad_coeff_grad_xd(f, X)
        dψcoeffxdtrunc = grad_coeff_grad_xd(f, X, collect(1:2:size(idx,1)))

        @test norm(dψcoeffxd[:,1:2:end] - dψcoeffxdtrunc)<1e-10
        @test norm(dψcoeffxd - grad_xk_basis(f, X, 1, Nx))<1e-10
        @test norm(dψcoeffxdtrunc - grad_xk_basis(f, X, 1, Nx, f.idx[collect(1:2:size(idx,1)),:]))<1e-10

        # Verify hess_coeff_grad_xd
        d2ψcoeffxd = hess_coeff_grad_xd(f, X)
        d2ψcoeffxdtrunc = hess_coeff_grad_xd(f, X, collect(1:2:size(idx,1)))

        @test norm(d2ψcoeffxd)<1e-10
        @test norm(d2ψcoeffxdtrunc)<1e-10
    end
end

@testset "Test evaluation, gradient and hessian of expanded function Nx = 3 I" begin

    Nx = 3
    Ne = 500
    X = randn(Nx, Ne)

    Blist = [CstProHermite(8); CstPhyHermite(8); CstLinProHermite(8); CstLinPhyHermite(8)]
    for b in Blist
        B = MultiBasis(b, Nx)


        idx = [0 0 0; 2 0 1; 0 1 0; 0 2 1; 0 1 2; 1 0 0; 2 2 2]
        truncidx = idx[1:2:end,:]
        coeff =  randn(size(idx,1))

        f = ExpandedFunction(B, idx, coeff)

        ψt_basis, dψt_basis, d2ψt_basis = alleval(f, X)

        ψ_basis = evaluate_basis(f, X)
        dψ_basis = grad_x_basis(f, X)
        d2ψ_basis = hess_x_basis(f, X);
        @test norm(ψt_basis - ψ_basis)<1e-8
        @test norm(dψt_basis - dψ_basis)<1e-8
        @test norm(d2ψt_basis - d2ψ_basis)<1e-8


        # For truncated basis
        ψtrunc_basis = evaluate_basis(f, X, truncidx)
        dψtrunc_basis = grad_x_basis(f, X, truncidx)
        d2ψtrunc_basis = hess_x_basis(f, X, truncidx);

        @test norm(ψ_basis[:,1:2:end] - ψtrunc_basis)<1e-8
        @test norm(dψ_basis[:,1:2:end,:] - dψtrunc_basis)<1e-8
        @test norm(d2ψt_basis[:,1:2:end,:,:] - d2ψtrunc_basis)<1e-8


        # Verify function evaluation
        @test norm(map(i->f(X[:,i]),1:Ne) - evaluate(f, X))<1e-8

        #  Verify gradient
        dψ = grad_x(f, X)
        @test norm(hcat(map(i->ForwardDiff.gradient(f, X[:,i]), 1:Ne)...)' - dψ)<1e-8

        # Verify hessian
        d2ψ = hess_x(f, X)

        for i=1:Ne
            d2ψt = ForwardDiff.hessian(f, X[:,i])
            @test norm(d2ψt - d2ψ[i,:,:])<1e-8
        end


        # Verify grad_xd
        dψxd = grad_xd(f, X)

        for i=1:Ne
            dψxd_t = ForwardDiff.gradient(f, X[:,i])[end]
            @test abs(dψxd[i] - dψxd_t)<1e-10
        end

        # Verify hess_xd
        d2ψxd = hess_xd(f, X)

        for i=1:Ne
            d2ψxd_t = ForwardDiff.hessian(f, X[:,i])[end,end]
            @test abs(d2ψxd[i] - d2ψxd_t)<1e-10
        end

        # Verify grad_x_grad_xd
        dxdxkψ = grad_x_grad_xd(f, X)

        for i=1:Ne
            dxdxkψ_t = ForwardDiff.hessian(f, X[:,i])
            for j=1:Nx
                @test abs(dxdxkψ[i,j] - dxdxkψ_t[j,end])<1e-10
            end
        end

        # Verify hess_x_grad_xd
        dxidxjdxkψ = hess_x_grad_xd(f, X, f.idx)

        for i=1:Ne
            dxidxjdxkψ_t = ForwardDiff.hessian(xi ->ForwardDiff.gradient(f, xi)[Nx], X[:,i])
            @test norm(dxidxjdxkψ[i,:,:] - dxidxjdxkψ_t)<1e-10
        end

        # Verify grad_coeff
        dψcoeff  = grad_coeff(f, X)
        dψcoefftrunc  = grad_coeff(f, X, collect(1:2:size(idx,1)))
        @test norm(dψcoeff[:,1:2:end] - dψcoefftrunc)<1e-10

        @test norm(dψcoeff - ψ_basis)<1e-10
        @test norm(dψcoefftrunc - ψtrunc_basis)<1e-10

        # Verify hess_coeff
        d2ψcoeff  = hess_coeff(f, X)
        d2ψcoefftrunc  = hess_coeff(f, X, collect(1:2:size(idx,1)))

        @test norm(d2ψcoeff)<1e-10

        # Verify grad_coeff_grad_xd
        dψcoeffxd = grad_coeff_grad_xd(f, X)
        dψcoeffxdtrunc = grad_coeff_grad_xd(f, X, collect(1:2:size(idx,1)))

        @test norm(dψcoeffxd[:,1:2:end] - dψcoeffxdtrunc)<1e-10
        @test norm(dψcoeffxd - grad_xk_basis(f, X, 1, Nx))<1e-10
        @test norm(dψcoeffxdtrunc - grad_xk_basis(f, X, 1, Nx, f.idx[collect(1:2:size(idx,1)),:]))<1e-10

        # Verify hess_coeff_grad_xd
        d2ψcoeffxd = hess_coeff_grad_xd(f, X)
        d2ψcoeffxdtrunc = hess_coeff_grad_xd(f, X, collect(1:2:size(idx,1)))

        @test norm(d2ψcoeffxd)<1e-10
        @test norm(d2ψcoeffxdtrunc)<1e-10
    end
end


@testset "Test evaluation, gradient and hessian of expanded function Nx = 3 II with only some active terms" begin

    Nx = 3
    Ne = 500
    X  = randn(Nx, Ne)

    Blist = [CstProHermite(8); CstPhyHermite(8); CstLinProHermite(8); CstLinPhyHermite(8)]
    for b in Blist
        B = MultiBasis(b, Nx)

        idx = [0 0 0; 0 0 1; 0 1 0; 0 2 1; 0 1 2; 0  3 0; 0 2 2]
        truncidx = idx[1:2:end,:]
        coeff =  randn(size(idx,1))

        f = ExpandedFunction(B, idx, coeff)

        ψt_basis, dψt_basis, d2ψt_basis = alleval(f, X)

        ψ_basis = evaluate_basis(f, X)
        dψ_basis = grad_x_basis(f, X)
        d2ψ_basis = hess_x_basis(f, X);
        @test norm(ψt_basis - ψ_basis)<1e-8
        @test norm(dψt_basis - dψ_basis)<1e-8
        @test norm(d2ψt_basis - d2ψ_basis)<1e-8


        # For truncated basis
        ψtrunc_basis = evaluate_basis(f, X, truncidx)
        dψtrunc_basis = grad_x_basis(f, X, truncidx)
        d2ψtrunc_basis = hess_x_basis(f, X, truncidx);

        @test norm(ψ_basis[:,1:2:end] - ψtrunc_basis)<1e-8
        @test norm(dψ_basis[:,1:2:end,:] - dψtrunc_basis)<1e-8
        @test norm(d2ψt_basis[:,1:2:end,:,:] - d2ψtrunc_basis)<1e-8


        # Verify function evaluation
        @test norm(map(i->f(X[:,i]),1:Ne) - evaluate(f, X))<1e-8

        #  Verify gradient
        dψ = grad_x(f, X)
        @test norm(hcat(map(i->ForwardDiff.gradient(f, X[:,i]), 1:Ne)...)' - dψ)<1e-8

        # Verify hessian
        d2ψ = hess_x(f, X)

        for i=1:Ne
            d2ψt = ForwardDiff.hessian(f, X[:,i])
            @test norm(d2ψt - d2ψ[i,:,:])<1e-8
        end


        # Verify grad_xd
        dψxd = grad_xd(f, X)

        for i=1:Ne
            dψxd_t = ForwardDiff.gradient(f, X[:,i])[end]
            @test abs(dψxd[i] - dψxd_t)<1e-10
        end

        # Verify hess_xd
        d2ψxd = hess_xd(f, X)

        for i=1:Ne
            d2ψxd_t = ForwardDiff.hessian(f, X[:,i])[end,end]
            @test abs(d2ψxd[i] - d2ψxd_t)<1e-10
        end

        # Verify grad_x_grad_xd
        dxdxkψ = grad_x_grad_xd(f, X)

        for i=1:Ne
            dxdxkψ_t = ForwardDiff.hessian(f, X[:,i])
            for j=1:Nx
                @test abs(dxdxkψ[i,j] - dxdxkψ_t[j,end])<1e-10
            end
        end

        # Verify hess_x_grad_xd
        dxidxjdxkψ = hess_x_grad_xd(f, X, f.idx)

        for i=1:Ne
            dxidxjdxkψ_t = ForwardDiff.hessian(xi ->ForwardDiff.gradient(f, xi)[Nx], X[:,i])
            @test norm(dxidxjdxkψ[i,:,:] - dxidxjdxkψ_t)<1e-10
        end

        # Verify grad_coeff
        dψcoeff  = grad_coeff(f, X)
        dψcoefftrunc  = grad_coeff(f, X, collect(1:2:size(idx,1)))
        @test norm(dψcoeff[:,1:2:end] - dψcoefftrunc)<1e-10

        @test norm(dψcoeff - ψ_basis)<1e-10
        @test norm(dψcoefftrunc - ψtrunc_basis)<1e-10

        # Verify hess_coeff
        d2ψcoeff  = hess_coeff(f, X)
        d2ψcoefftrunc  = hess_coeff(f, X, collect(1:2:size(idx,1)))

        @test norm(d2ψcoeff)<1e-10

        # Verify grad_coeff_grad_xd
        dψcoeffxd = grad_coeff_grad_xd(f, X)
        dψcoeffxdtrunc = grad_coeff_grad_xd(f, X, collect(1:2:size(idx,1)))

        @test norm(dψcoeffxd[:,1:2:end] - dψcoeffxdtrunc)<1e-10
        @test norm(dψcoeffxd - grad_xk_basis(f, X, 1, Nx))<1e-10
        @test norm(dψcoeffxdtrunc - grad_xk_basis(f, X, 1, Nx, f.idx[collect(1:2:size(idx,1)),:]))<1e-10

        # Verify hess_coeff_grad_xd
        d2ψcoeffxd = hess_coeff_grad_xd(f, X)
        d2ψcoeffxdtrunc = hess_coeff_grad_xd(f, X, collect(1:2:size(idx,1)))

        @test norm(d2ψcoeffxd)<1e-10
        @test norm(d2ψcoeffxdtrunc)<1e-10
    end
end

@testset "Test evaluation, gradient and hessian of expanded function Nx = 3 III" begin

    Nx = 3
    Ne = 500
    X = randn(Nx, Ne)

    Blist = [CstProHermite(8); CstPhyHermite(8); CstLinProHermite(8); CstLinPhyHermite(8)]
    for b in Blist
        B = MultiBasis(b, Nx)

        idx = [0 0 0]
        truncidx = idx[1:2:end,:]
        coeff =  randn(size(idx,1))

        f = ExpandedFunction(B, idx, coeff)

        ψt_basis, dψt_basis, d2ψt_basis = alleval(f, X)

        ψ_basis = evaluate_basis(f, X)
        dψ_basis = grad_x_basis(f, X)
        d2ψ_basis = hess_x_basis(f, X);
        @test norm(ψt_basis - ψ_basis)<1e-8
        @test norm(dψt_basis - dψ_basis)<1e-8
        @test norm(d2ψt_basis - d2ψ_basis)<1e-8


        # For truncated basis
        ψtrunc_basis = evaluate_basis(f, X, truncidx)
        dψtrunc_basis = grad_x_basis(f, X, truncidx)
        d2ψtrunc_basis = hess_x_basis(f, X, truncidx);

        @test norm(ψ_basis[:,1:2:end] - ψtrunc_basis)<1e-8
        @test norm(dψ_basis[:,1:2:end,:] - dψtrunc_basis)<1e-8
        @test norm(d2ψt_basis[:,1:2:end,:,:] - d2ψtrunc_basis)<1e-8


        # Verify function evaluation
        @test norm(map(i->f(X[:,i]),1:Ne) - evaluate(f, X))<1e-8

        #  Verify gradient
        dψ = grad_x(f, X)
        @test norm(hcat(map(i->ForwardDiff.gradient(f, X[:,i]), 1:Ne)...)' - dψ)<1e-8

        # Verify hessian
        d2ψ = hess_x(f, X)

        for i=1:Ne
            d2ψt = ForwardDiff.hessian(f, X[:,i])
            @test norm(d2ψt - d2ψ[i,:,:])<1e-8
        end


        # Verify grad_xd
        dψxd = grad_xd(f, X)

        for i=1:Ne
            dψxd_t = ForwardDiff.gradient(f, X[:,i])[end]
            @test abs(dψxd[i] - dψxd_t)<1e-10
        end

        # Verify hess_xd
        d2ψxd = hess_xd(f, X)

        for i=1:Ne
            d2ψxd_t = ForwardDiff.hessian(f, X[:,i])[end,end]
            @test abs(d2ψxd[i] - d2ψxd_t)<1e-10
        end

        # Verify grad_x_grad_xd
        dxdxkψ = grad_x_grad_xd(f, X)

        for i=1:Ne
            dxdxkψ_t = ForwardDiff.hessian(f, X[:,i])
            for j=1:Nx
                @test abs(dxdxkψ[i,j] - dxdxkψ_t[j,end])<1e-10
            end
        end

        # Verify hess_x_grad_xd
        dxidxjdxkψ = hess_x_grad_xd(f, X, f.idx)

        for i=1:Ne
            dxidxjdxkψ_t = ForwardDiff.hessian(xi ->ForwardDiff.gradient(f, xi)[Nx], X[:,i])
            @test norm(dxidxjdxkψ[i,:,:] - dxidxjdxkψ_t)<1e-10
        end

        # Verify grad_coeff
        dψcoeff  = grad_coeff(f, X)
        dψcoefftrunc  = grad_coeff(f, X, collect(1:2:size(idx,1)))
        @test norm(dψcoeff[:,1:2:end] - dψcoefftrunc)<1e-10

        @test norm(dψcoeff - ψ_basis)<1e-10
        @test norm(dψcoefftrunc - ψtrunc_basis)<1e-10

        # Verify hess_coeff
        d2ψcoeff  = hess_coeff(f, X)
        d2ψcoefftrunc  = hess_coeff(f, X, collect(1:2:size(idx,1)))

        @test norm(d2ψcoeff)<1e-10

        # Verify grad_coeff_grad_xd
        dψcoeffxd = grad_coeff_grad_xd(f, X)
        dψcoeffxdtrunc = grad_coeff_grad_xd(f, X, collect(1:2:size(idx,1)))

        @test norm(dψcoeffxd[:,1:2:end] - dψcoeffxdtrunc)<1e-10
        @test norm(dψcoeffxd - grad_xk_basis(f, X, 1, Nx))<1e-10
        @test norm(dψcoeffxdtrunc - grad_xk_basis(f, X, 1, Nx, f.idx[collect(1:2:size(idx,1)),:]))<1e-10

        # Verify hess_coeff_grad_xd
        d2ψcoeffxd = hess_coeff_grad_xd(f, X)
        d2ψcoeffxdtrunc = hess_coeff_grad_xd(f, X, collect(1:2:size(idx,1)))

        @test norm(d2ψcoeffxd)<1e-10
        @test norm(d2ψcoeffxdtrunc)<1e-10
    end
end
