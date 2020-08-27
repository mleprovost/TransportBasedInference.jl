using AdaptiveTransportMap: evaluate


@testset "Test evaluation, gradient and hessian of expanded function Nx = 1 I" begin

    Nx = 1
    Ne = 500
    ens = EnsembleState(Nx, Ne)
    ens.S .= randn(Nx, Ne)


    B = MultiBasis(CstProHermite(3), Nx)

    idx = reshape([0], (1, 1))
    truncidx = idx[1:2:end,:]

    coeff =  randn(size(idx,1))

    f = ExpandedFunction(B, idx, coeff)



    ψt_basis, dψt_basis, d2ψt_basis = alleval(f, ens)

    ψ_basis = evaluate_basis(f, ens.S)
    dψ_basis = grad_x_basis(f, ens.S)
    d2ψ_basis = hess_x_basis(f, ens.S);
    @test norm(ψt_basis - ψ_basis)<1e-8
    @test norm(dψt_basis - dψ_basis)<1e-8
    @test norm(d2ψt_basis - d2ψ_basis)<1e-8


    # For truncated basis
    ψtrunc_basis = evaluate_basis(f, ens.S, truncidx)
    dψtrunc_basis = grad_x_basis(f, ens.S, truncidx)
    d2ψtrunc_basis = hess_x_basis(f, ens.S, truncidx);

    @test norm(ψ_basis[:,1:2:end] - ψtrunc_basis)<1e-8
    @test norm(dψ_basis[:,1:2:end,:] - dψtrunc_basis)<1e-8
    @test norm(d2ψt_basis[:,1:2:end,:,:] - d2ψtrunc_basis)<1e-8


    # Verify function evaluation
    @test norm(map(i->f(member(ens,i)),1:Ne) - evaluate(f, ens.S))<1e-8

    #  Verify gradient
    dψ = grad_x(f, ens.S)
    @test norm(hcat(map(i->ForwardDiff.gradient(f, member(ens,i)), 1:Ne)...)' - dψ)<1e-8

    # Verify hessian
    d2ψ = hess_x(f, ens.S)

    for i=1:Ne
        d2ψt = ForwardDiff.hessian(f, member(ens,i))
        @test norm(d2ψt - d2ψ[i,:,:])<1e-8
    end


    # Verify grad_xd
    dψxd = grad_xd(f, ens.S)

    for i=1:Ne
        dψxd_t = ForwardDiff.gradient(f, member(ens,i))[end]
        @test abs(dψxd[i] - dψxd_t)<1e-10
    end

    # Verify hess_xd
    d2ψxd = hess_xd(f, ens.S)

    for i=1:Ne
        d2ψxd_t = ForwardDiff.hessian(f, member(ens,i))[end,end]
        @test abs(d2ψxd[i] - d2ψxd_t)<1e-10
    end

    # Verify grad_x_grad_xd
    dxdxkψ = grad_x_grad_xd(f, ens.S)

    for i=1:Ne
        dxdxkψ_t = ForwardDiff.hessian(f, member(ens,i))
        for j=1:Nx
            @test abs(dxdxkψ[i,j] - dxdxkψ_t[j,end])<1e-10
        end
    end

    # Verify hess_x_grad_xd
    dxidxjdxkψ = hess_x_grad_xd(f, ens.S, f.idx)

    for i=1:Ne
        dxidxjdxkψ_t = ForwardDiff.hessian(xi ->ForwardDiff.gradient(f, xi)[Nx], member(ens,i))
        @test norm(dxidxjdxkψ[i,:,:] - dxidxjdxkψ_t)<1e-10
    end

    # Verify grad_coeff
    dψcoeff  = grad_coeff(f, ens.S)
    dψcoefftrunc  = grad_coeff(f, ens.S, collect(1:2:size(idx,1)))
    @test norm(dψcoeff[:,1:2:end] - dψcoefftrunc)<1e-10

    @test norm(dψcoeff - ψ_basis)<1e-10
    @test norm(dψcoefftrunc - ψtrunc_basis)<1e-10

    # Verify hess_coeff
    d2ψcoeff  = hess_coeff(f, ens.S)
    d2ψcoefftrunc  = hess_coeff(f, ens.S, collect(1:2:size(idx,1)))

    @test norm(d2ψcoeff)<1e-10

    # Verify grad_coeff_grad_xd
    dψcoeffxd = grad_coeff_grad_xd(f, ens.S)
    dψcoeffxdtrunc = grad_coeff_grad_xd(f, ens.S, collect(1:2:size(idx,1)))

    @test norm(dψcoeffxd[:,1:2:end] - dψcoeffxdtrunc)<1e-10
    @test norm(dψcoeffxd - grad_xk_basis(f, ens.S, 1, Nx))<1e-10
    @test norm(dψcoeffxdtrunc - grad_xk_basis(f, ens.S, 1, Nx, f.idx[collect(1:2:size(idx,1)),:]))<1e-10

    # Verify hess_coeff_grad_xd
    d2ψcoeffxd = hess_coeff_grad_xd(f, ens.S)
    d2ψcoeffxdtrunc = hess_coeff_grad_xd(f, ens.S, collect(1:2:size(idx,1)))

    @test norm(d2ψcoeffxd)<1e-10
    @test norm(d2ψcoeffxdtrunc)<1e-10
end

@testset "Test evaluation, gradient and hessian of expanded function Nx = 1 II" begin

    Nx = 1
    Ne = 500
    ens = EnsembleState(Nx, Ne)
    ens.S .= randn(Nx, Ne)


    B = MultiBasis(CstProHermite(3), Nx)

    idx = reshape([0; 1; 2; 3], (4, 1))
    truncidx = idx[1:2:end,:]

    coeff =  randn(size(idx,1))

    f = ExpandedFunction(B, idx, coeff)



    ψt_basis, dψt_basis, d2ψt_basis = alleval(f, ens)

    ψ_basis = evaluate_basis(f, ens.S)
    dψ_basis = grad_x_basis(f, ens.S)
    d2ψ_basis = hess_x_basis(f, ens.S);
    @test norm(ψt_basis - ψ_basis)<1e-8
    @test norm(dψt_basis - dψ_basis)<1e-8
    @test norm(d2ψt_basis - d2ψ_basis)<1e-8


    # For truncated basis
    ψtrunc_basis = evaluate_basis(f, ens.S, truncidx)
    dψtrunc_basis = grad_x_basis(f, ens.S, truncidx)
    d2ψtrunc_basis = hess_x_basis(f, ens.S, truncidx);

    @test norm(ψ_basis[:,1:2:end] - ψtrunc_basis)<1e-8
    @test norm(dψ_basis[:,1:2:end,:] - dψtrunc_basis)<1e-8
    @test norm(d2ψt_basis[:,1:2:end,:,:] - d2ψtrunc_basis)<1e-8


    # Verify function evaluation
    @test norm(map(i->f(member(ens,i)),1:Ne) - evaluate(f, ens.S))<1e-8

    #  Verify gradient
    dψ = grad_x(f, ens.S)
    @test norm(hcat(map(i->ForwardDiff.gradient(f, member(ens,i)), 1:Ne)...)' - dψ)<1e-8

    # Verify hessian
    d2ψ = hess_x(f, ens.S)

    for i=1:Ne
        d2ψt = ForwardDiff.hessian(f, member(ens,i))
        @test norm(d2ψt - d2ψ[i,:,:])<1e-8
    end


    # Verify grad_xd
    dψxd = grad_xd(f, ens.S)

    for i=1:Ne
        dψxd_t = ForwardDiff.gradient(f, member(ens,i))[end]
        @test abs(dψxd[i] - dψxd_t)<1e-10
    end

    # Verify hess_xd
    d2ψxd = hess_xd(f, ens.S)

    for i=1:Ne
        d2ψxd_t = ForwardDiff.hessian(f, member(ens,i))[end,end]
        @test abs(d2ψxd[i] - d2ψxd_t)<1e-10
    end

    # Verify grad_x_grad_xd
    dxdxkψ = grad_x_grad_xd(f, ens.S)

    for i=1:Ne
        dxdxkψ_t = ForwardDiff.hessian(f, member(ens,i))
        for j=1:Nx
            @test abs(dxdxkψ[i,j] - dxdxkψ_t[j,end])<1e-10
        end
    end

    # Verify hess_x_grad_xd
    dxidxjdxkψ = hess_x_grad_xd(f, ens.S, f.idx)

    for i=1:Ne
        dxidxjdxkψ_t = ForwardDiff.hessian(xi ->ForwardDiff.gradient(f, xi)[Nx], member(ens,i))
        @test norm(dxidxjdxkψ[i,:,:] - dxidxjdxkψ_t)<1e-10
    end

    # Verify grad_coeff
    dψcoeff  = grad_coeff(f, ens.S)
    dψcoefftrunc  = grad_coeff(f, ens.S, collect(1:2:size(idx,1)))
    @test norm(dψcoeff[:,1:2:end] - dψcoefftrunc)<1e-10

    @test norm(dψcoeff - ψ_basis)<1e-10
    @test norm(dψcoefftrunc - ψtrunc_basis)<1e-10

    # Verify hess_coeff
    d2ψcoeff  = hess_coeff(f, ens.S)
    d2ψcoefftrunc  = hess_coeff(f, ens.S, collect(1:2:size(idx,1)))

    @test norm(d2ψcoeff)<1e-10

    # Verify grad_coeff_grad_xd
    dψcoeffxd = grad_coeff_grad_xd(f, ens.S)
    dψcoeffxdtrunc = grad_coeff_grad_xd(f, ens.S, collect(1:2:size(idx,1)))

    @test norm(dψcoeffxd[:,1:2:end] - dψcoeffxdtrunc)<1e-10
    @test norm(dψcoeffxd - grad_xk_basis(f, ens.S, 1, Nx))<1e-10
    @test norm(dψcoeffxdtrunc - grad_xk_basis(f, ens.S, 1, Nx, f.idx[collect(1:2:size(idx,1)),:]))<1e-10

    # Verify hess_coeff_grad_xd
    d2ψcoeffxd = hess_coeff_grad_xd(f, ens.S)
    d2ψcoeffxdtrunc = hess_coeff_grad_xd(f, ens.S, collect(1:2:size(idx,1)))

    @test norm(d2ψcoeffxd)<1e-10
    @test norm(d2ψcoeffxdtrunc)<1e-10
end

@testset "Test evaluation, gradient and hessian of expanded function Nx = 2" begin

    Nx = 2
    Ne = 500
    ens = EnsembleState(Nx, Ne)
    ens.S .= randn(Nx, Ne)


    B = MultiBasis(CstProHermite(3), Nx)

    idx = [0 0]
    truncidx = idx[1:2:end,:]

    coeff =  randn(size(idx,1))

    f = ExpandedFunction(B, idx, coeff)


    ψt_basis, dψt_basis, d2ψt_basis = alleval(f, ens)

    ψ_basis = evaluate_basis(f, ens.S)
    dψ_basis = grad_x_basis(f, ens.S)
    d2ψ_basis = hess_x_basis(f, ens.S);
    @test norm(ψt_basis - ψ_basis)<1e-8
    @test norm(dψt_basis - dψ_basis)<1e-8
    @test norm(d2ψt_basis - d2ψ_basis)<1e-8


    # For truncated basis
    ψtrunc_basis = evaluate_basis(f, ens.S, truncidx)
    dψtrunc_basis = grad_x_basis(f, ens.S, truncidx)
    d2ψtrunc_basis = hess_x_basis(f, ens.S, truncidx);

    @test norm(ψ_basis[:,1:2:end] - ψtrunc_basis)<1e-8
    @test norm(dψ_basis[:,1:2:end,:] - dψtrunc_basis)<1e-8
    @test norm(d2ψt_basis[:,1:2:end,:,:] - d2ψtrunc_basis)<1e-8


    # Verify function evaluation
    @test norm(map(i->f(member(ens,i)),1:Ne) - evaluate(f, ens.S))<1e-8

    #  Verify gradient
    dψ = grad_x(f, ens.S)
    @test norm(hcat(map(i->ForwardDiff.gradient(f, member(ens,i)), 1:Ne)...)' - dψ)<1e-8

    # Verify hessian
    d2ψ = hess_x(f, ens.S)

    for i=1:Ne
        d2ψt = ForwardDiff.hessian(f, member(ens,i))
        @test norm(d2ψt - d2ψ[i,:,:])<1e-8
    end


    # Verify grad_xd
    dψxd = grad_xd(f, ens.S)

    for i=1:Ne
        dψxd_t = ForwardDiff.gradient(f, member(ens,i))[end]
        @test abs(dψxd[i] - dψxd_t)<1e-10
    end

    # Verify hess_xd
    d2ψxd = hess_xd(f, ens.S)

    for i=1:Ne
        d2ψxd_t = ForwardDiff.hessian(f, member(ens,i))[end,end]
        @test abs(d2ψxd[i] - d2ψxd_t)<1e-10
    end

    # Verify grad_x_grad_xd
    dxdxkψ = grad_x_grad_xd(f, ens.S)

    for i=1:Ne
        dxdxkψ_t = ForwardDiff.hessian(f, member(ens,i))
        for j=1:Nx
            @test abs(dxdxkψ[i,j] - dxdxkψ_t[j,end])<1e-10
        end
    end

    # Verify hess_x_grad_xd
    dxidxjdxkψ = hess_x_grad_xd(f, ens.S, f.idx)

    for i=1:Ne
        dxidxjdxkψ_t = ForwardDiff.hessian(xi ->ForwardDiff.gradient(f, xi)[Nx], member(ens,i))
        @test norm(dxidxjdxkψ[i,:,:] - dxidxjdxkψ_t)<1e-10
    end

    # Verify grad_coeff
    dψcoeff  = grad_coeff(f, ens.S)
    dψcoefftrunc  = grad_coeff(f, ens.S, collect(1:2:size(idx,1)))
    @test norm(dψcoeff[:,1:2:end] - dψcoefftrunc)<1e-10

    @test norm(dψcoeff - ψ_basis)<1e-10
    @test norm(dψcoefftrunc - ψtrunc_basis)<1e-10

    # Verify hess_coeff
    d2ψcoeff  = hess_coeff(f, ens.S)
    d2ψcoefftrunc  = hess_coeff(f, ens.S, collect(1:2:size(idx,1)))

    @test norm(d2ψcoeff)<1e-10

    # Verify grad_coeff_grad_xd
    dψcoeffxd = grad_coeff_grad_xd(f, ens.S)
    dψcoeffxdtrunc = grad_coeff_grad_xd(f, ens.S, collect(1:2:size(idx,1)))

    @test norm(dψcoeffxd[:,1:2:end] - dψcoeffxdtrunc)<1e-10
    @test norm(dψcoeffxd - grad_xk_basis(f, ens.S, 1, Nx))<1e-10
    @test norm(dψcoeffxdtrunc - grad_xk_basis(f, ens.S, 1, Nx, f.idx[collect(1:2:size(idx,1)),:]))<1e-10

    # Verify hess_coeff_grad_xd
    d2ψcoeffxd = hess_coeff_grad_xd(f, ens.S)
    d2ψcoeffxdtrunc = hess_coeff_grad_xd(f, ens.S, collect(1:2:size(idx,1)))

    @test norm(d2ψcoeffxd)<1e-10
    @test norm(d2ψcoeffxdtrunc)<1e-10
end

@testset "Test evaluation, gradient and hessian of expanded function Nx = 2" begin

    Nx = 2
    Ne = 500
    ens = EnsembleState(Nx, Ne)
    ens.S .= randn(Nx, Ne)


    B = MultiBasis(CstProHermite(3), Nx)

    idx = [0 0; 0 1; 1 0; 1 1; 1 2]
    truncidx = idx[1:2:end,:]

    coeff =  randn(size(idx,1))

    f = ExpandedFunction(B, idx, coeff)



    ψt_basis, dψt_basis, d2ψt_basis = alleval(f, ens)

    ψ_basis = evaluate_basis(f, ens.S)
    dψ_basis = grad_x_basis(f, ens.S)
    d2ψ_basis = hess_x_basis(f, ens.S);
    @test norm(ψt_basis - ψ_basis)<1e-8
    @test norm(dψt_basis - dψ_basis)<1e-8
    @test norm(d2ψt_basis - d2ψ_basis)<1e-8


    # For truncated basis
    ψtrunc_basis = evaluate_basis(f, ens.S, truncidx)
    dψtrunc_basis = grad_x_basis(f, ens.S, truncidx)
    d2ψtrunc_basis = hess_x_basis(f, ens.S, truncidx);

    @test norm(ψ_basis[:,1:2:end] - ψtrunc_basis)<1e-8
    @test norm(dψ_basis[:,1:2:end,:] - dψtrunc_basis)<1e-8
    @test norm(d2ψt_basis[:,1:2:end,:,:] - d2ψtrunc_basis)<1e-8


    # Verify function evaluation
    @test norm(map(i->f(member(ens,i)),1:Ne) - evaluate(f, ens.S))<1e-8

    #  Verify gradient
    dψ = grad_x(f, ens.S)
    @test norm(hcat(map(i->ForwardDiff.gradient(f, member(ens,i)), 1:Ne)...)' - dψ)<1e-8

    # Verify hessian
    d2ψ = hess_x(f, ens.S)

    for i=1:Ne
        d2ψt = ForwardDiff.hessian(f, member(ens,i))
        @test norm(d2ψt - d2ψ[i,:,:])<1e-8
    end


    # Verify grad_xd
    dψxd = grad_xd(f, ens.S)

    for i=1:Ne
        dψxd_t = ForwardDiff.gradient(f, member(ens,i))[end]
        @test abs(dψxd[i] - dψxd_t)<1e-10
    end

    # Verify hess_xd
    d2ψxd = hess_xd(f, ens.S)

    for i=1:Ne
        d2ψxd_t = ForwardDiff.hessian(f, member(ens,i))[end,end]
        @test abs(d2ψxd[i] - d2ψxd_t)<1e-10
    end

    # Verify grad_x_grad_xd
    dxdxkψ = grad_x_grad_xd(f, ens.S)

    for i=1:Ne
        dxdxkψ_t = ForwardDiff.hessian(f, member(ens,i))
        for j=1:Nx
            @test abs(dxdxkψ[i,j] - dxdxkψ_t[j,end])<1e-10
        end
    end

    # Verify hess_x_grad_xd
    dxidxjdxkψ = hess_x_grad_xd(f, ens.S, f.idx)

    for i=1:Ne
        dxidxjdxkψ_t = ForwardDiff.hessian(xi ->ForwardDiff.gradient(f, xi)[Nx], member(ens,i))
        @test norm(dxidxjdxkψ[i,:,:] - dxidxjdxkψ_t)<1e-10
    end

    # Verify grad_coeff
    dψcoeff  = grad_coeff(f, ens.S)
    dψcoefftrunc  = grad_coeff(f, ens.S, collect(1:2:size(idx,1)))
    @test norm(dψcoeff[:,1:2:end] - dψcoefftrunc)<1e-10

    @test norm(dψcoeff - ψ_basis)<1e-10
    @test norm(dψcoefftrunc - ψtrunc_basis)<1e-10

    # Verify hess_coeff
    d2ψcoeff  = hess_coeff(f, ens.S)
    d2ψcoefftrunc  = hess_coeff(f, ens.S, collect(1:2:size(idx,1)))

    @test norm(d2ψcoeff)<1e-10

    # Verify grad_coeff_grad_xd
    dψcoeffxd = grad_coeff_grad_xd(f, ens.S)
    dψcoeffxdtrunc = grad_coeff_grad_xd(f, ens.S, collect(1:2:size(idx,1)))

    @test norm(dψcoeffxd[:,1:2:end] - dψcoeffxdtrunc)<1e-10
    @test norm(dψcoeffxd - grad_xk_basis(f, ens.S, 1, Nx))<1e-10
    @test norm(dψcoeffxdtrunc - grad_xk_basis(f, ens.S, 1, Nx, f.idx[collect(1:2:size(idx,1)),:]))<1e-10

    # Verify hess_coeff_grad_xd
    d2ψcoeffxd = hess_coeff_grad_xd(f, ens.S)
    d2ψcoeffxdtrunc = hess_coeff_grad_xd(f, ens.S, collect(1:2:size(idx,1)))

    @test norm(d2ψcoeffxd)<1e-10
    @test norm(d2ψcoeffxdtrunc)<1e-10
end

@testset "Test evaluation, gradient and hessian of expanded function Nx = 3 I" begin

    Nx = 3
    Ne = 500
    ens = EnsembleState(Nx, Ne)
    ens.S .= randn(Nx, Ne)

    B = MultiBasis(CstProHermite(3), Nx)

    idx = [0 0 0; 2 0 1; 0 1 0; 0 2 1; 0 1 2; 1 0 0; 2 2 2]
    truncidx = idx[1:2:end,:]
    coeff =  randn(size(idx,1))

    f = ExpandedFunction(B, idx, coeff)

    ψt_basis, dψt_basis, d2ψt_basis = alleval(f, ens)

    ψ_basis = evaluate_basis(f, ens.S)
    dψ_basis = grad_x_basis(f, ens.S)
    d2ψ_basis = hess_x_basis(f, ens.S);
    @test norm(ψt_basis - ψ_basis)<1e-8
    @test norm(dψt_basis - dψ_basis)<1e-8
    @test norm(d2ψt_basis - d2ψ_basis)<1e-8


    # For truncated basis
    ψtrunc_basis = evaluate_basis(f, ens.S, truncidx)
    dψtrunc_basis = grad_x_basis(f, ens.S, truncidx)
    d2ψtrunc_basis = hess_x_basis(f, ens.S, truncidx);

    @test norm(ψ_basis[:,1:2:end] - ψtrunc_basis)<1e-8
    @test norm(dψ_basis[:,1:2:end,:] - dψtrunc_basis)<1e-8
    @test norm(d2ψt_basis[:,1:2:end,:,:] - d2ψtrunc_basis)<1e-8


    # Verify function evaluation
    @test norm(map(i->f(member(ens,i)),1:Ne) - evaluate(f, ens.S))<1e-8

    #  Verify gradient
    dψ = grad_x(f, ens.S)
    @test norm(hcat(map(i->ForwardDiff.gradient(f, member(ens,i)), 1:Ne)...)' - dψ)<1e-8

    # Verify hessian
    d2ψ = hess_x(f, ens.S)

    for i=1:Ne
        d2ψt = ForwardDiff.hessian(f, member(ens,i))
        @test norm(d2ψt - d2ψ[i,:,:])<1e-8
    end


    # Verify grad_xd
    dψxd = grad_xd(f, ens.S)

    for i=1:Ne
        dψxd_t = ForwardDiff.gradient(f, member(ens,i))[end]
        @test abs(dψxd[i] - dψxd_t)<1e-10
    end

    # Verify hess_xd
    d2ψxd = hess_xd(f, ens.S)

    for i=1:Ne
        d2ψxd_t = ForwardDiff.hessian(f, member(ens,i))[end,end]
        @test abs(d2ψxd[i] - d2ψxd_t)<1e-10
    end

    # Verify grad_x_grad_xd
    dxdxkψ = grad_x_grad_xd(f, ens.S)

    for i=1:Ne
        dxdxkψ_t = ForwardDiff.hessian(f, member(ens,i))
        for j=1:Nx
            @test abs(dxdxkψ[i,j] - dxdxkψ_t[j,end])<1e-10
        end
    end

    # Verify hess_x_grad_xd
    dxidxjdxkψ = hess_x_grad_xd(f, ens.S, f.idx)

    for i=1:Ne
        dxidxjdxkψ_t = ForwardDiff.hessian(xi ->ForwardDiff.gradient(f, xi)[Nx], member(ens,i))
        @test norm(dxidxjdxkψ[i,:,:] - dxidxjdxkψ_t)<1e-10
    end

    # Verify grad_coeff
    dψcoeff  = grad_coeff(f, ens.S)
    dψcoefftrunc  = grad_coeff(f, ens.S, collect(1:2:size(idx,1)))
    @test norm(dψcoeff[:,1:2:end] - dψcoefftrunc)<1e-10

    @test norm(dψcoeff - ψ_basis)<1e-10
    @test norm(dψcoefftrunc - ψtrunc_basis)<1e-10

    # Verify hess_coeff
    d2ψcoeff  = hess_coeff(f, ens.S)
    d2ψcoefftrunc  = hess_coeff(f, ens.S, collect(1:2:size(idx,1)))

    @test norm(d2ψcoeff)<1e-10

    # Verify grad_coeff_grad_xd
    dψcoeffxd = grad_coeff_grad_xd(f, ens.S)
    dψcoeffxdtrunc = grad_coeff_grad_xd(f, ens.S, collect(1:2:size(idx,1)))

    @test norm(dψcoeffxd[:,1:2:end] - dψcoeffxdtrunc)<1e-10
    @test norm(dψcoeffxd - grad_xk_basis(f, ens.S, 1, Nx))<1e-10
    @test norm(dψcoeffxdtrunc - grad_xk_basis(f, ens.S, 1, Nx, f.idx[collect(1:2:size(idx,1)),:]))<1e-10

    # Verify hess_coeff_grad_xd
    d2ψcoeffxd = hess_coeff_grad_xd(f, ens.S)
    d2ψcoeffxdtrunc = hess_coeff_grad_xd(f, ens.S, collect(1:2:size(idx,1)))

    @test norm(d2ψcoeffxd)<1e-10
    @test norm(d2ψcoeffxdtrunc)<1e-10
end

@testset "Test evaluation, gradient and hessian of expanded function Nx = 3 II" begin

    Nx = 3
    Ne = 500
    ens = EnsembleState(Nx, Ne)
    ens.S .= randn(Nx, Ne)

    B = MultiBasis(CstProHermite(3), Nx)

    idx = [0 0 0]
    truncidx = idx[1:2:end,:]
    coeff =  randn(size(idx,1))

    f = ExpandedFunction(B, idx, coeff)

    ψt_basis, dψt_basis, d2ψt_basis = alleval(f, ens)

    ψ_basis = evaluate_basis(f, ens.S)
    dψ_basis = grad_x_basis(f, ens.S)
    d2ψ_basis = hess_x_basis(f, ens.S);
    @test norm(ψt_basis - ψ_basis)<1e-8
    @test norm(dψt_basis - dψ_basis)<1e-8
    @test norm(d2ψt_basis - d2ψ_basis)<1e-8


    # For truncated basis
    ψtrunc_basis = evaluate_basis(f, ens.S, truncidx)
    dψtrunc_basis = grad_x_basis(f, ens.S, truncidx)
    d2ψtrunc_basis = hess_x_basis(f, ens.S, truncidx);

    @test norm(ψ_basis[:,1:2:end] - ψtrunc_basis)<1e-8
    @test norm(dψ_basis[:,1:2:end,:] - dψtrunc_basis)<1e-8
    @test norm(d2ψt_basis[:,1:2:end,:,:] - d2ψtrunc_basis)<1e-8


    # Verify function evaluation
    @test norm(map(i->f(member(ens,i)),1:Ne) - evaluate(f, ens.S))<1e-8

    #  Verify gradient
    dψ = grad_x(f, ens.S)
    @test norm(hcat(map(i->ForwardDiff.gradient(f, member(ens,i)), 1:Ne)...)' - dψ)<1e-8

    # Verify hessian
    d2ψ = hess_x(f, ens.S)

    for i=1:Ne
        d2ψt = ForwardDiff.hessian(f, member(ens,i))
        @test norm(d2ψt - d2ψ[i,:,:])<1e-8
    end


    # Verify grad_xd
    dψxd = grad_xd(f, ens.S)

    for i=1:Ne
        dψxd_t = ForwardDiff.gradient(f, member(ens,i))[end]
        @test abs(dψxd[i] - dψxd_t)<1e-10
    end

    # Verify hess_xd
    d2ψxd = hess_xd(f, ens.S)

    for i=1:Ne
        d2ψxd_t = ForwardDiff.hessian(f, member(ens,i))[end,end]
        @test abs(d2ψxd[i] - d2ψxd_t)<1e-10
    end

    # Verify grad_x_grad_xd
    dxdxkψ = grad_x_grad_xd(f, ens.S)

    for i=1:Ne
        dxdxkψ_t = ForwardDiff.hessian(f, member(ens,i))
        for j=1:Nx
            @test abs(dxdxkψ[i,j] - dxdxkψ_t[j,end])<1e-10
        end
    end

    # Verify hess_x_grad_xd
    dxidxjdxkψ = hess_x_grad_xd(f, ens.S, f.idx)

    for i=1:Ne
        dxidxjdxkψ_t = ForwardDiff.hessian(xi ->ForwardDiff.gradient(f, xi)[Nx], member(ens,i))
        @test norm(dxidxjdxkψ[i,:,:] - dxidxjdxkψ_t)<1e-10
    end

    # Verify grad_coeff
    dψcoeff  = grad_coeff(f, ens.S)
    dψcoefftrunc  = grad_coeff(f, ens.S, collect(1:2:size(idx,1)))
    @test norm(dψcoeff[:,1:2:end] - dψcoefftrunc)<1e-10

    @test norm(dψcoeff - ψ_basis)<1e-10
    @test norm(dψcoefftrunc - ψtrunc_basis)<1e-10

    # Verify hess_coeff
    d2ψcoeff  = hess_coeff(f, ens.S)
    d2ψcoefftrunc  = hess_coeff(f, ens.S, collect(1:2:size(idx,1)))

    @test norm(d2ψcoeff)<1e-10

    # Verify grad_coeff_grad_xd
    dψcoeffxd = grad_coeff_grad_xd(f, ens.S)
    dψcoeffxdtrunc = grad_coeff_grad_xd(f, ens.S, collect(1:2:size(idx,1)))

    @test norm(dψcoeffxd[:,1:2:end] - dψcoeffxdtrunc)<1e-10
    @test norm(dψcoeffxd - grad_xk_basis(f, ens.S, 1, Nx))<1e-10
    @test norm(dψcoeffxdtrunc - grad_xk_basis(f, ens.S, 1, Nx, f.idx[collect(1:2:size(idx,1)),:]))<1e-10

    # Verify hess_coeff_grad_xd
    d2ψcoeffxd = hess_coeff_grad_xd(f, ens.S)
    d2ψcoeffxdtrunc = hess_coeff_grad_xd(f, ens.S, collect(1:2:size(idx,1)))

    @test norm(d2ψcoeffxd)<1e-10
    @test norm(d2ψcoeffxdtrunc)<1e-10
end
