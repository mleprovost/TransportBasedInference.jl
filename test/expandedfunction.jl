
@testset "Test evaluation, gradient and hessian of expanded function Nx = 1" begin

    Nx = 1
    Ne = 500
    ens = EnsembleState(Nx, Ne)
    ens.S .= randn(Nx, Ne)


    B = MultiBasis(CstProHermite(3; scaled =true), Nx)

    idx = reshape([0 ; 1; 2; 3],(4,1))
    coeff =  randn(size(idx,1))

    f = ExpandedFunction(B, idx, coeff)

    # Truth obtained from automatic-differentiation
    ψt, dψt, d2ψt = alleval(f, ens)

    ψ = evaluate_basis(f, ens)
    dψ = grad_x_basis(f, ens)
    d2ψ = hess_x_basis(f, ens)

    @test norm(ψt - ψ)<1e-8
    @test norm(dψt - dψ)<1e-8
    @test norm(d2ψt - d2ψ)<1e-8
end

@testset "Test evaluation, gradient and hessian of expanded function Nx = 2" begin

    Nx = 2
    Ne = 500
    ens = EnsembleState(Nx, Ne)
    ens.S .= randn(Nx, Ne)


    B = MultiBasis(CstProHermite(3; scaled =true), Nx)

    idx = [0 0]
    coeff =  randn(size(idx,1))

    f = ExpandedFunction(B, idx, coeff)

    # Truth obtained from automatic-differentiation
    ψt, dψt, d2ψt = alleval(f, ens)

    ψ = evaluate_basis(f, ens)
    dψ = grad_x_basis(f, ens)
    d2ψ = hess_x_basis(f, ens)

    @test norm(ψt - ψ)<1e-8
    @test norm(dψt - dψ)<1e-8
    @test norm(d2ψt - d2ψ)<1e-8
end

@testset "Test evaluation, gradient and hessian of expanded function Nx = 2" begin

    Nx = 2
    Ne = 500
    ens = EnsembleState(Nx, Ne)
    ens.S .= randn(Nx, Ne)


    B = MultiBasis(CstProHermite(3; scaled =true), Nx)

    idx = [0 0; 0 1; 1 0; 1 1; 1 2]
    coeff =  randn(size(idx,1))

    f = ExpandedFunction(B, idx, coeff)

    # Truth obtained from automatic-differentiation
    ψt, dψt, d2ψt = alleval(f, ens)

    ψ = evaluate_basis(f, ens)
    dψ = grad_x_basis(f, ens)
    d2ψ = hess_x_basis(f, ens)

    @test norm(ψt - ψ)<1e-8
    @test norm(dψt - dψ)<1e-8
    @test norm(d2ψt - d2ψ)<1e-8
end

@testset "Test evaluation, gradient and hessian of expanded function Nx = 3" begin

    Nx = 3
    Ne = 500
    ens = EnsembleState(Nx, Ne)
    ens.S .= randn(Nx, Ne)


    B = MultiBasis(CstProHermite(3; scaled =true), Nx)

    idx = [0 0 0; 0 0 1; 0 1 0; 0 1 1; 0 1 2; 1 0 0]
    coeff =  randn(size(idx,1))

    f = ExpandedFunction(B, idx, coeff)

    # Truth obtained from automatic-differentiation
    ψt, dψt, d2ψt = alleval(f, ens)

    ψ = evaluate_basis(f, ens)
    dψ = grad_x_basis(f, ens)
    d2ψ = hess_x_basis(f, ens)

    @test norm(ψt - ψ)<1e-8
    @test norm(dψt - dψ)<1e-8
    @test norm(d2ψt - d2ψ)<1e-8
end

@testset "Test evaluation, gradient and hessian of expanded function Nx = 3" begin

    Nx = 3
    Ne = 500
    ens = EnsembleState(Nx, Ne)
    ens.S .= randn(Nx, Ne)


    B = MultiBasis(CstProHermite(3; scaled =true), Nx)

    idx = [0 0 0]
    coeff =  randn(size(idx,1))

    f = ExpandedFunction(B, idx, coeff)

    # Truth obtained from automatic-differentiation
    ψt, dψt, d2ψt = alleval(f, ens)

    ψ = evaluate_basis(f, ens)
    dψ = grad_x_basis(f, ens)
    d2ψ = hess_x_basis(f, ens)

    @test norm(ψt - ψ)<1e-8
    @test norm(dψt - dψ)<1e-8
    @test norm(d2ψt - d2ψ)<1e-8
end
