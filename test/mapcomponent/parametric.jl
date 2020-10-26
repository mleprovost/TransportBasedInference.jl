using AdaptiveTransportMap: evaluate

@testset "Test tools for parametric function Nx = 1 " begin

    Nx = 1
    Ne = 500
    X = zeros(Nx, Ne)

    X .= randn(Nx, Ne)

    B = MultiBasis(CstProHermite(4), Nx)

    idx = reshape([0], (1,1))
    truncidx = idx[1:2:end,:]
    Nψ = 1

    coeff = randn(Nψ)
    f = ExpandedFunction(B, idx, coeff)
    fp = ParametricFunction(f)

    ## Test evaluate_offdiagbasis
    ψoff = evaluate_offdiagbasis(fp, X)
    ψofftrunc = evaluate_offdiagbasis(fp, X, truncidx)

    @test norm(ψoff[:,1:2:end] - ψofftrunc) < 1e-8

    ψofft = zeros(Ne, Nψ)


    for j=1:Nψ
        fj = MultiFunction(MultiBasis(f.B.B,Nx-1), f.idx[j,1:Nx-1])
        for i=1:Ne
            ψofft[i,j] += fj(X[:,i][1:end-1])
        end
    end

    @test norm(ψofft - ψoff)<1e-8


    ## Test evaluate_diagbasis
    ψdiag = evaluate_diagbasis(fp, X)
    ψdiagtrunc = evaluate_diagbasis(fp, X, truncidx)

    @test norm(ψdiag[:,1:2:end] - ψdiagtrunc) < 1e-8


    ψdiagt = zeros(Ne, Nψ)


    for j=1:Nψ
        fj = MultiFunction(MultiBasis(f.B.B,1), f.idx[j,Nx:Nx])
        for i=1:Ne
            ψdiagt[i,j] += fj([X[:,i][end]])
        end
    end

    @test norm(ψdiagt - ψdiag)<1e-8

    ## Test grad_xd_diagbasis
    Gψ_xd_diag = grad_xd_diagbasis(fp, X)
    Gψ_xd_diagtrunc = grad_xd_diagbasis(fp, X, truncidx)
    @test Gψ_xd_diag[:,1:2:end] == Gψ_xd_diagtrunc

    Gψ_xd_diagt = zeros(Ne, Nψ)

    for j=1:Nψ
        fj = f.B.B[f.idx[j,end]+1]
        for i=1:Ne
            Gψ_xd_diagt[i,j] += ForwardDiff.derivative(fj, X[:,i][end])
        end
    end

    @test norm(Gψ_xd_diagt - Gψ_xd_diag)<1e-10

    ## Test hess_xd_diagbasis

    Hψ_xd_diag = hess_xd_diagbasis(fp, X)
    Hψ_xd_diagtrunc = hess_xd_diagbasis(fp, X, truncidx)
    @test Hψ_xd_diag[:,1:2:end] == Hψ_xd_diagtrunc

    Hψ_xd_diagt = zeros(Ne, Nψ)

    for j=1:Nψ
        fj = f.B.B[f.idx[j,end]+1]
        for i=1:Ne
            Hψ_xd_diagt[i,j] += ForwardDiff.derivative(x->ForwardDiff.derivative(fj, x), X[:,i][end])
        end
    end

    @test norm(Hψ_xd_diagt - Hψ_xd_diag)<1e-10

    ## Test function evaluation

    ψ = evaluate(fp, X)
    ψt = zeros(Ne)

    for i=1:Ne
        ψt[i] = fp.f(X[:,i])
    end
    @test norm(ψ - ψt)<1e-10

    ## Test grad_xd

    dψxd = grad_xd(fp, X)
    dψxdt = zeros(Ne)

    for i=1:Ne
        dψxdt[i] = ForwardDiff.gradient(f, X[:,i])[end]
    end
    @test norm(dψxd - dψxdt)<1e-10


    ## Test hess_xd

    d2ψxd = hess_xd(fp, X)
    d2ψxdt = zeros(Ne)

    for i=1:Ne
        d2ψxdt[i] = ForwardDiff.hessian(f, X[:,i])[end, end]
    end

    @test norm(d2ψxd - d2ψxdt)<1e-10

    ## Test grad_coeff

    dψ_coefft = evaluate_basis(fp.f, X)
    dψ_coeff = grad_coeff(fp, X)
    dψ_coeffexpanded = grad_coeff(fp.f, X)

    @test norm(dψ_coeff - dψ_coefft) < 1e-10
    @test norm(dψ_coeffexpanded - dψ_coefft) < 1e-10

    ## Test grad_coeff_grad_xd

    dψ_coeff_xd = grad_coeff_grad_xd(fp, X)
    dψ_coeff_xdt = zeros(Ne, Nψ)

    for j=1:Nψ
        foffj = MultiFunction(MultiBasis(fp.f.B.B,Nx-1), fp.f.idx[j,1:end-1])
        fdiagj = f.B.B[f.idx[j,end]+1]
        for i=1:Ne
            dψ_coeff_xdt[i,j] = foffj(X[:,i][1:end-1])*ForwardDiff.derivative(fdiagj, X[:,i][end])
        end
    end

    @test norm(dψ_coeff_xd - dψ_coeff_xdt)<1e-10

end

@testset "Test tools for parametric function Nx = 1 II " begin

    Nx = 1
    Ne = 500
    X = zeros(Nx, Ne)

    X .= randn(Nx, Ne)

    B = MultiBasis(CstProHermite(4), Nx)

    idx = reshape([0; 1; 2; 3; 4], (5,1))
    truncidx = idx[1:2:end,:]
    Nψ = 5

    coeff = randn(Nψ)
    f = ExpandedFunction(B, idx, coeff)
    fp = ParametricFunction(f)

    ## Test evaluate_offdiagbasis
    ψoff = evaluate_offdiagbasis(fp, X)
    ψofftrunc = evaluate_offdiagbasis(fp, X, truncidx)

    @test ψoff[:,1:2:end] == ψofftrunc

    ψofft = zeros(Ne, Nψ)


    for j=1:Nψ
        fj = MultiFunction(MultiBasis(f.B.B,Nx-1), f.idx[j,1:Nx-1])
        for i=1:Ne
            ψofft[i,j] += fj(X[:,i][1:end-1])
        end
    end

    @test norm(ψofft - ψoff)<1e-8


    ## Test evaluate_diagbasis
    ψdiag = evaluate_diagbasis(fp, X)
    ψdiagtrunc = evaluate_diagbasis(fp, X, truncidx)

    @test ψdiag[:,1:2:end] == ψdiagtrunc


    ψdiagt = zeros(Ne, Nψ)


    for j=1:Nψ
        fj = MultiFunction(MultiBasis(f.B.B,1), f.idx[j,Nx:Nx])
        for i=1:Ne
            ψdiagt[i,j] += fj([X[:,i][end]])
        end
    end

    @test norm(ψdiagt - ψdiag)<1e-8

    ## Test grad_xd_diagbasis
    Gψ_xd_diag = grad_xd_diagbasis(fp, X)
    Gψ_xd_diagtrunc = grad_xd_diagbasis(fp, X, truncidx)
    @test Gψ_xd_diag[:,1:2:end] == Gψ_xd_diagtrunc

    Gψ_xd_diagt = zeros(Ne, Nψ)

    for j=1:Nψ
        fj = f.B.B[f.idx[j,end]+1]
        for i=1:Ne
            Gψ_xd_diagt[i,j] += ForwardDiff.derivative(fj, X[:,i][end])
        end
    end

    @test norm(Gψ_xd_diagt - Gψ_xd_diag)<1e-10

    ## Test hess_xd_diagbasis

    Hψ_xd_diag = hess_xd_diagbasis(fp, X)
    Hψ_xd_diagtrunc = hess_xd_diagbasis(fp, X, truncidx)
    @test Hψ_xd_diag[:,1:2:end] == Hψ_xd_diagtrunc

    Hψ_xd_diagt = zeros(Ne, Nψ)

    for j=1:Nψ
        fj = f.B.B[f.idx[j,end]+1]
        for i=1:Ne
            Hψ_xd_diagt[i,j] += ForwardDiff.derivative(x->ForwardDiff.derivative(fj, x), X[:,i][end])
        end
    end

    @test norm(Hψ_xd_diagt - Hψ_xd_diag)<1e-10

    ## Test function evaluation

    ψ = evaluate(fp, X)
    ψt = zeros(Ne)

    for i=1:Ne
        ψt[i] = fp.f(X[:,i])
    end
    @test norm(ψ - ψt)<1e-10

    ## Test grad_xd

    dψxd = grad_xd(fp, X)
    dψxdt = zeros(Ne)

    for i=1:Ne
        dψxdt[i] = ForwardDiff.gradient(f, X[:,i])[end]
    end
    @test norm(dψxd - dψxdt)<1e-10


    ## Test hess_xd

    d2ψxd = hess_xd(fp, X)
    d2ψxdt = zeros(Ne)

    for i=1:Ne
        d2ψxdt[i] = ForwardDiff.hessian(f, X[:,i])[end, end]
    end

    @test norm(d2ψxd - d2ψxdt)<1e-10

    ## Test grad_coeff

    dψ_coefft = evaluate_basis(fp.f, X)
    dψ_coeff = grad_coeff(fp, X)
    dψ_coeffexpanded = grad_coeff(fp.f, X)

    @test norm(dψ_coeff - dψ_coefft) < 1e-10
    @test norm(dψ_coeffexpanded - dψ_coefft) < 1e-10

    ## Test grad_coeff_grad_xd

    dψ_coeff_xd = grad_coeff_grad_xd(fp, X)
    dψ_coeff_xdt = zeros(Ne, Nψ)

    for j=1:Nψ
        foffj = MultiFunction(MultiBasis(fp.f.B.B,Nx-1), fp.f.idx[j,1:end-1])
        fdiagj = f.B.B[f.idx[j,end]+1]
        for i=1:Ne
            dψ_coeff_xdt[i,j] = foffj(X[:,i][1:end-1])*ForwardDiff.derivative(fdiagj, X[:,i][end])
        end
    end

    @test norm(dψ_coeff_xd - dψ_coeff_xdt)<1e-10


end

@testset "Test tools for parametric function Nx = 2" begin

    Nx = 2
    Ne = 500
    X = zeros(Nx, Ne)

    X .= randn(Nx, Ne)

    B = MultiBasis(CstProHermite(4), Nx)

    idx = [0 0; 0 1; 1 0; 1 1; 1 2; 2 1]
    truncidx = idx[1:2:end,:]
    Nψ = 6

    coeff = randn(Nψ)
    f = ExpandedFunction(B, idx, coeff)
    fp = ParametricFunction(f)

    ## Test evaluate_offdiagbasis
    ψoff = evaluate_offdiagbasis(fp, X)
    ψofftrunc = evaluate_offdiagbasis(fp, X, truncidx)

    @test norm(ψoff[:,1:2:end] - ψofftrunc) < 1e-8

    ψofft = zeros(Ne, Nψ)


    for j=1:Nψ
        fj = MultiFunction(MultiBasis(f.B.B,Nx-1), f.idx[j,1:Nx-1])
        for i=1:Ne
            ψofft[i,j] += fj(X[:,i][1:end-1])
        end
    end

    @test norm(ψofft - ψoff)<1e-8


    ## Test evaluate_diagbasis
    ψdiag = evaluate_diagbasis(fp, X)
    ψdiagtrunc = evaluate_diagbasis(fp, X, truncidx)

    @test norm(ψdiag[:,1:2:end] - ψdiagtrunc) < 1e-8


    ψdiagt = zeros(Ne, Nψ)


    for j=1:Nψ
        fj = MultiFunction(MultiBasis(f.B.B,1), f.idx[j,Nx:Nx])
        for i=1:Ne
            ψdiagt[i,j] += fj([X[:,i][end]])
        end
    end

    @test norm(ψdiagt - ψdiag)<1e-8

    ## Test grad_xd_diagbasis
    Gψ_xd_diag = grad_xd_diagbasis(fp, X)
    Gψ_xd_diagtrunc = grad_xd_diagbasis(fp, X, truncidx)
    @test Gψ_xd_diag[:,1:2:end] == Gψ_xd_diagtrunc

    Gψ_xd_diagt = zeros(Ne, Nψ)

    for j=1:Nψ
        fj = f.B.B[f.idx[j,end]+1]
        for i=1:Ne
            Gψ_xd_diagt[i,j] += ForwardDiff.derivative(fj, X[:,i][end])
        end
    end

    @test norm(Gψ_xd_diagt - Gψ_xd_diag)<1e-10

    ## Test hess_xd_diagbasis

    Hψ_xd_diag = hess_xd_diagbasis(fp, X)
    Hψ_xd_diagtrunc = hess_xd_diagbasis(fp, X, truncidx)
    @test Hψ_xd_diag[:,1:2:end] == Hψ_xd_diagtrunc

    Hψ_xd_diagt = zeros(Ne, Nψ)

    for j=1:Nψ
        fj = f.B.B[f.idx[j,end]+1]
        for i=1:Ne
            Hψ_xd_diagt[i,j] += ForwardDiff.derivative(x->ForwardDiff.derivative(fj, x), X[:,i][end])
        end
    end

    @test norm(Hψ_xd_diagt - Hψ_xd_diag)<1e-10

    ## Test function evaluation

    ψ = evaluate(fp, X)
    ψt = zeros(Ne)

    for i=1:Ne
        ψt[i] = fp.f(X[:,i])
    end
    @test norm(ψ - ψt)<1e-10

    ## Test grad_xd

    dψxd = grad_xd(fp, X)
    dψxdt = zeros(Ne)

    for i=1:Ne
        dψxdt[i] = ForwardDiff.gradient(f, X[:,i])[end]
    end
    @test norm(dψxd - dψxdt)<1e-10


    ## Test hess_xd

    d2ψxd = hess_xd(fp, X)
    d2ψxdt = zeros(Ne)

    for i=1:Ne
        d2ψxdt[i] = ForwardDiff.hessian(f, X[:,i])[end, end]
    end

    @test norm(d2ψxd - d2ψxdt)<1e-10

    ## Test grad_coeff

    dψ_coefft = evaluate_basis(fp.f, X)
    dψ_coeff = grad_coeff(fp, X)
    dψ_coeffexpanded = grad_coeff(fp.f, X)

    @test norm(dψ_coeff - dψ_coefft) < 1e-10
    @test norm(dψ_coeffexpanded - dψ_coefft) < 1e-10

    ## Test grad_coeff_grad_xd

    dψ_coeff_xd = grad_coeff_grad_xd(fp, X)
    dψ_coeff_xdt = zeros(Ne, Nψ)

    for j=1:Nψ
        foffj = MultiFunction(MultiBasis(fp.f.B.B,Nx-1), fp.f.idx[j,1:end-1])
        fdiagj = f.B.B[f.idx[j,end]+1]
        for i=1:Ne
            dψ_coeff_xdt[i,j] = foffj(X[:,i][1:end-1])*ForwardDiff.derivative(fdiagj, X[:,i][end])
        end
    end

    @test norm(dψ_coeff_xd - dψ_coeff_xdt)<1e-10


end


@testset "Test tools for parametric function Nx = 3" begin

    Nx = 3
    Ne = 500
    X = zeros(Nx, Ne)

    X .= randn(Nx, Ne)

    B = MultiBasis(CstProHermite(4), Nx)

    idx = [0 0 0 ;0  0 1; 0 1 0; 1 0 0;1 1 0; 0 1 1; 1 0 1; 1 1 1; 1 2 0; 2 1 0]
    truncidx = idx[1:2:end,:]
    Nψ = 10

    coeff = randn(Nψ)
    f = ExpandedFunction(B, idx, coeff)
    fp = ParametricFunction(f)

    ## Test evaluate_offdiagbasis
    ψoff = evaluate_offdiagbasis(fp, X)
    ψofftrunc = evaluate_offdiagbasis(fp, X, truncidx)

    @test norm(ψoff[:,1:2:end] - ψofftrunc) < 1e-8

    ψofft = zeros(Ne, Nψ)


    for j=1:Nψ
        fj = MultiFunction(MultiBasis(f.B.B,Nx-1), f.idx[j,1:Nx-1])
        for i=1:Ne
            ψofft[i,j] += fj(X[:,i][1:end-1])
        end
    end

    @test norm(ψofft - ψoff)<1e-8


    ## Test evaluate_diagbasis
    ψdiag = evaluate_diagbasis(fp, X)
    ψdiagtrunc = evaluate_diagbasis(fp, X, truncidx)

    @test norm(ψdiag[:,1:2:end] - ψdiagtrunc) < 1e-8


    ψdiagt = zeros(Ne, Nψ)


    for j=1:Nψ
        fj = MultiFunction(MultiBasis(f.B.B,1), f.idx[j,Nx:Nx])
        for i=1:Ne
            ψdiagt[i,j] += fj([X[:,i][end]])
        end
    end

    @test norm(ψdiagt - ψdiag)<1e-8

    ## Test grad_xd_diagbasis
    Gψ_xd_diag = grad_xd_diagbasis(fp, X)
    Gψ_xd_diagtrunc = grad_xd_diagbasis(fp, X, truncidx)
    @test Gψ_xd_diag[:,1:2:end] == Gψ_xd_diagtrunc

    Gψ_xd_diagt = zeros(Ne, Nψ)

    for j=1:Nψ
        fj = f.B.B[f.idx[j,end]+1]
        for i=1:Ne
            Gψ_xd_diagt[i,j] += ForwardDiff.derivative(fj, X[:,i][end])
        end
    end

    @test norm(Gψ_xd_diagt - Gψ_xd_diag)<1e-10

    ## Test hess_xd_diagbasis

    Hψ_xd_diag = hess_xd_diagbasis(fp, X)
    Hψ_xd_diagtrunc = hess_xd_diagbasis(fp, X, truncidx)
    @test Hψ_xd_diag[:,1:2:end] == Hψ_xd_diagtrunc

    Hψ_xd_diagt = zeros(Ne, Nψ)

    for j=1:Nψ
        fj = f.B.B[f.idx[j,end]+1]
        for i=1:Ne
            Hψ_xd_diagt[i,j] += ForwardDiff.derivative(x->ForwardDiff.derivative(fj, x), X[:,i][end])
        end
    end

    @test norm(Hψ_xd_diagt - Hψ_xd_diag)<1e-10

    ## Test function evaluation

    ψ = evaluate(fp, X)
    ψt = zeros(Ne)

    for i=1:Ne
        ψt[i] = fp.f(X[:,i])
    end
    @test norm(ψ - ψt)<1e-10

    ## Test grad_xd

    dψxd = grad_xd(fp, X)
    dψxdt = zeros(Ne)

    for i=1:Ne
        dψxdt[i] = ForwardDiff.gradient(f, X[:,i])[end]
    end
    @test norm(dψxd - dψxdt)<1e-10


    ## Test hess_xd

    d2ψxd = hess_xd(fp, X)
    d2ψxdt = zeros(Ne)

    for i=1:Ne
        d2ψxdt[i] = ForwardDiff.hessian(f, X[:,i])[end, end]
    end

    @test norm(d2ψxd - d2ψxdt)<1e-10

    ## Test grad_coeff

    dψ_coefft = evaluate_basis(fp.f, X)
    dψ_coeff = grad_coeff(fp, X)
    dψ_coeffexpanded = grad_coeff(fp.f, X)

    @test norm(dψ_coeff - dψ_coefft) < 1e-10
    @test norm(dψ_coeffexpanded - dψ_coefft) < 1e-10

    ## Test grad_coeff_grad_xd

    dψ_coeff_xd = grad_coeff_grad_xd(fp, X)
    dψ_coeff_xdt = zeros(Ne, Nψ)

    for j=1:Nψ
        foffj = MultiFunction(MultiBasis(fp.f.B.B,Nx-1), fp.f.idx[j,1:end-1])
        fdiagj = f.B.B[f.idx[j,end]+1]
        for i=1:Ne
            dψ_coeff_xdt[i,j] = foffj(X[:,i][1:end-1])*ForwardDiff.derivative(fdiagj, X[:,i][end])
        end
    end

    @test norm(dψ_coeff_xd - dψ_coeff_xdt)<1e-10
end

@testset "Test tools for parametric function Nx = 3" begin

    Nx = 3
    Ne = 500
    X = zeros(Nx, Ne)

    X .= randn(Nx, Ne)

    B = MultiBasis(CstProHermite(4), Nx)

    idx = [0 0 0]
    truncidx = idx[1:2:end,:]
    Nψ = 1

    coeff = randn(Nψ)
    f = ExpandedFunction(B, idx, coeff)
    fp = ParametricFunction(f)

    ## Test evaluate_offdiagbasis
    ψoff = evaluate_offdiagbasis(fp, X)
    ψofftrunc = evaluate_offdiagbasis(fp, X, truncidx)

    @test norm(ψoff[:,1:2:end] - ψofftrunc) < 1e-8

    ψofft = zeros(Ne, Nψ)


    for j=1:Nψ
        fj = MultiFunction(MultiBasis(f.B.B,Nx-1), f.idx[j,1:Nx-1])
        for i=1:Ne
            ψofft[i,j] += fj(X[:,i][1:end-1])
        end
    end

    @test norm(ψofft - ψoff)<1e-8


    ## Test evaluate_diagbasis
    ψdiag = evaluate_diagbasis(fp, X)
    ψdiagtrunc = evaluate_diagbasis(fp, X, truncidx)

    @test norm(ψdiag[:,1:2:end] - ψdiagtrunc) < 1e-8


    ψdiagt = zeros(Ne, Nψ)


    for j=1:Nψ
        fj = MultiFunction(MultiBasis(f.B.B,1), f.idx[j,Nx:Nx])
        for i=1:Ne
            ψdiagt[i,j] += fj([X[:,i][end]])
        end
    end

    @test norm(ψdiagt - ψdiag)<1e-8

    ## Test grad_xd_diagbasis
    Gψ_xd_diag = grad_xd_diagbasis(fp, X)
    Gψ_xd_diagtrunc = grad_xd_diagbasis(fp, X, truncidx)
    @test Gψ_xd_diag[:,1:2:end] == Gψ_xd_diagtrunc

    Gψ_xd_diagt = zeros(Ne, Nψ)

    for j=1:Nψ
        fj = f.B.B[f.idx[j,end]+1]
        for i=1:Ne
            Gψ_xd_diagt[i,j] += ForwardDiff.derivative(fj, X[:,i][end])
        end
    end

    @test norm(Gψ_xd_diagt - Gψ_xd_diag)<1e-10

    ## Test hess_xd_diagbasis

    Hψ_xd_diag = hess_xd_diagbasis(fp, X)
    Hψ_xd_diagtrunc = hess_xd_diagbasis(fp, X, truncidx)
    @test Hψ_xd_diag[:,1:2:end] == Hψ_xd_diagtrunc

    Hψ_xd_diagt = zeros(Ne, Nψ)

    for j=1:Nψ
        fj = f.B.B[f.idx[j,end]+1]
        for i=1:Ne
            Hψ_xd_diagt[i,j] += ForwardDiff.derivative(x->ForwardDiff.derivative(fj, x), X[:,i][end])
        end
    end

    @test norm(Hψ_xd_diagt - Hψ_xd_diag)<1e-10

    ## Test function evaluation

    ψ = evaluate(fp, X)
    ψt = zeros(Ne)

    for i=1:Ne
        ψt[i] = fp.f(X[:,i])
    end
    @test norm(ψ - ψt)<1e-10

    ## Test grad_xd

    dψxd = grad_xd(fp, X)
    dψxdt = zeros(Ne)

    for i=1:Ne
        dψxdt[i] = ForwardDiff.gradient(f, X[:,i])[end]
    end
    @test norm(dψxd - dψxdt)<1e-10


    ## Test hess_xd

    d2ψxd = hess_xd(fp, X)
    d2ψxdt = zeros(Ne)

    for i=1:Ne
        d2ψxdt[i] = ForwardDiff.hessian(f, X[:,i])[end, end]
    end

    @test norm(d2ψxd - d2ψxdt)<1e-10

    ## Test grad_coeff

    dψ_coefft = evaluate_basis(fp.f, X)
    dψ_coeff = grad_coeff(fp, X)
    dψ_coeffexpanded = grad_coeff(fp.f, X)

    @test norm(dψ_coeff - dψ_coefft) < 1e-10
    @test norm(dψ_coeffexpanded - dψ_coefft) < 1e-10

    ## Test grad_coeff_grad_xd

    dψ_coeff_xd = grad_coeff_grad_xd(fp, X)
    dψ_coeff_xdt = zeros(Ne, Nψ)

    for j=1:Nψ
        foffj = MultiFunction(MultiBasis(fp.f.B.B,Nx-1), fp.f.idx[j,1:end-1])
        fdiagj = f.B.B[f.idx[j,end]+1]
        for i=1:Ne
            dψ_coeff_xdt[i,j] = foffj(X[:,i][1:end-1])*ForwardDiff.derivative(fdiagj, X[:,i][end])
        end
    end

    @test norm(dψ_coeff_xd - dψ_coeff_xdt)<1e-10


end
