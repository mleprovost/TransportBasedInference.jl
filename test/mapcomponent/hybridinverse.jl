

@testset "Test bisection method for strictly increasing function" begin
    f(x) = tanh(3.0 + 2.0* x)

    a = -3.0
    b = 3.0
    ϵx = 1e-14
    ϵf = 1e-14
    x = a + 0.5*(b-a)

    ya = f(a)
    yb = f(b)

    while b - a > ϵx || abs(f(x)) > ϵf
        x, fx, a, ya, b, yb = bisection(x, f(x), a , ya, b, yb)
    end

    @test (b - a) <= ϵx
    @test abs(f(x)) <= ϵf
end

@testset "Test bisection, Newton, hybrid method for wavy globally increasing function" begin
    f(x) = 3*x - exp(0.5-x) + cos(3.5π*x)
    g(x) = ForwardDiff.derivative(f, x)

    xroot = 0.4144911998972488

    a = -5.0
    b = 7.0
    ϵx = 1e-6
    ϵf = 1e-6
    # Start from midpoint
    x = -4.9#a + 0.5*(b-a)
    xold = x
    fx = f(x)
    gx = g(x)


    ya = f(a)
    yb = f(b)

    # newton method
    while abs(fx) > ϵf
        fx = f(x)
        gx = g(x)
        x -= fx/gx
    end

    xrootnewton = x
    @test isapprox(xroot, xrootnewton, atol = ϵx)
    @test isapprox(f(xrootnewton), 0.0, atol = ϵx)

    # bisection method

    a = -5.0
    b = 7.0
    ϵx = 1e-4
    ϵf = 1e-4
    # Start from midpoint
    x = -4.9#a + 0.5*(b-a)
    xold = x
    fx = f(x)
    gx = g(x)


    ya = f(a)
    yb = f(b)

    # bisection method
    while b - a > ϵx
        x, fx, a, ya, b, yb = bisection(x, f(x), a , ya, b, yb)
    end

    xrootbisection = x
    @test isapprox(xroot, xrootbisection, atol = ϵx)
    @test isapprox(a, b, atol = ϵx)

    # hybrid method Newton + bisection

    a = -5.0
    b = 7.0
    ϵx = 1e-4
    ϵf = 1e-4

    # Start with a bad initial guees
    x = -4.9

    xroothybrid = hybridsolver(f, g, x, a, b)

    @test isapprox(xroot, xroothybrid, atol = ϵx)
end


@testset "Test hybridinverse function" begin

    Ne = 200
    Nx = 3

    idx = [0 0 0; 0 0 1; 0 0 2; 0 1 0; 0 0 1; 0 1 1; 0 2 1; 0 3 1; 1 3 1]

    Nψ = size(idx,1)

    coeff = randn(Nψ)

    B = MultiBasis(CstProHermite(6), Nx)
    f = ParametricFunction(ExpandedFunction(B, idx, coeff))
    R = IntegratedFunction(f)

    X = randn(Nx, Ne) .* randn(Nx, Ne)

    F = evaluate(R, X)

    Xmodified = deepcopy(X)

    Xmodified[end,:] .+= cos.(2*π*randn(Ne)) .* exp.(-randn(Ne).^2/2)

    @test norm(Xmodified[end,:]-X[end,:])>1e-6

    S = Storage(f, Xmodified)

    hybridinverse!(Xmodified, F, R, S)

    @test norm(Xmodified[end,:] - X[end,:])<2e-8

    @test norm(Xmodified[1:end-1,:] - X[1:end-1,:])<2e-8
end
