# Test all the reduced functions


@testset "Test reduced_grad_x_grad_xd and reduced_hess_x_grad_xd with no active dimensions" begin
    Nx_tab = [1;2;4;8]
    Ne = 100
    m = 10

    for Nx in Nx_tab

        X = randn(Nx, Ne) .* randn(Nx, Ne) .+ randn(Nx)
        B = MultiBasis(CstProHermiteBasis(m-2), Nx)
        idx = zeros(Int64, 1, Nx)
        coeff = randn(size(idx,1))
        f = ExpandedFunction(B, idx, coeff)

        # Test reduced_grad_x_grad_xd
        dψ = grad_x_grad_xd(f, X)
        dψr = reduced_grad_x_grad_xd(f, X)

        @test norm(dψ[:,f.dim] - dψr)<1e-8

        # Test reduced_hess_x_grad_xd
        d2ψ = hess_x_grad_xd(f, X)
        d2ψr = reduced_hess_x_grad_xd(f, X)

        @test norm(d2ψ[:,f.dim, f.dim] - d2ψr)<1e-8

    end
end


@testset "Test reduced_grad_x_grad_xd and reduced_hess_x_grad_xd with dimensions" begin
    Nx_tab = [1;2;2;2;4;6;8]
    Ne = 100
    m = 10
    Nψ = 5

    idx_tab = (reshape([0; 1; 2; 3; 4], (5, 1)),
               [0 0; 0 1; 1 0; 1 1; 1 2],
               [0 0; 0 1; 0 2; 0 3; 0 4],
               [0 0; 1 0; 2 0; 3 0; 4 0],
                [1  0  0  0;
                 2  0  0  0;
                 0  0  1  0;
                 0  0  2  0;
                 3  0  0  0],
                [0  0  0  0  0  1;
                 1  0  2  0  3  2;
                 1  0  0  0  0  0;
                 2  0  0  0  0  0;
                 1  0  2  2  1  1],
               [0  0  0  0  0  1  0  0;
                0  0  0  0  0  0  0  1;
                0  0  0  0  0  0  0  2;
                0  0  0  0  0  2  0  0;
                0  0  1  0  0  0  0  0])

    for (i, Nx) in enumerate(Nx_tab)
        X = randn(Nx, Ne) .* randn(Nx, Ne) .+ randn(Nx)
        B = MultiBasis(CstProHermiteBasis(m-2), Nx)
        idx = idx_tab[i]

        coeff = randn(size(idx,1))
        f = ExpandedFunction(B, idx, coeff)

        # Test reduced_grad_x_grad_xd
        dψ = grad_x_grad_xd(f, X)
        dψr = reduced_grad_x_grad_xd(f, X)

        @test norm(dψ[:,f.dim] - dψr)<1e-8

        # Test reduced_hess_x_grad_xd
        d2ψ = hess_x_grad_xd(f, X)
        d2ψr = reduced_hess_x_grad_xd(f, X)

        @test norm(d2ψ[:,f.dim, f.dim] - d2ψr)<1e-8

    end
end
