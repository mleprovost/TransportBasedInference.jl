@testset "Test Rectifier" begin 
    atol = 1e-9

    for T in [:square, :softplus, :sigmoid, :explinearunit]
        x = rand()
        r = Rectifier(T)
        # Test gradient
        @test abs(ForwardDiff.derivative(y->r(y), x) - grad_x(r, x) ) < 1e-10

        # Test hessian
        @test abs(ForwardDiff.derivative(z->ForwardDiff.derivative(y->r(y), z),x) - hess_x(r, x) ) < 1e-10

        # Test gradient of log evaluation
        @test abs(ForwardDiff.derivative(y->log(r(y)), x) - grad_x_logeval(r, x) ) < 1e-10

        # Test hessian of log evaluation
        @test abs(ForwardDiff.hessian(y->log(r(y[1])), [x])[1,1] - hess_x_logeval(r, x) ) < 1e-10
    end
end

@testset "Test ExpRectifier" begin 
    atol = 1e-9

    x = rand()
    r = ExpRectifier
    # Test gradient
    @test abs(ForwardDiff.derivative(y->r(y), x) - grad_x(r, x) ) < 1e-10

    # Test hessian
    @test abs(ForwardDiff.derivative(z->ForwardDiff.derivative(y->r(y), z),x) - hess_x(r, x) ) < 1e-10

    # Test gradient of log evaluation
    @test abs(ForwardDiff.derivative(y->log(r(y)), x) - grad_x_logeval(r, x) ) < 1e-10

    # Test hessian of log evaluation
    @test abs(ForwardDiff.hessian(y->log(r(y[1])), [x])[1,1] - hess_x_logeval(r, x) ) < 1e-10
end
