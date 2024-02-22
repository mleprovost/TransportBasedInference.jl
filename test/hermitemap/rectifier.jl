
@testset "Rectifier square" begin

r = Rectifier("squared")
@test r.T == "squared"

    x = 2.5
@test abs(r(x) - x^2)<1e-10


# Test gradient
@test abs(ForwardDiff.derivative(y->r(y), x) - grad_x(r, x) ) < 1e-10

# Test hessian
@test abs(ForwardDiff.derivative(z->ForwardDiff.derivative(y->r(y), z),x) - hess_x(r, x) ) < 1e-10

# Test gradient of log evaluation
@test abs(ForwardDiff.derivative(y->log(r(y)), x) - grad_x_logeval(r, x) ) < 1e-10

# Test hessian of log evaluation
@test abs(ForwardDiff.hessian(y->log(r(y[1])), [x])[1,1] - hess_x_logeval(r, x) ) < 1e-10

x = -2.5
@test abs(r(x) - x^2)<1e-10


# Test gradient
@test abs(ForwardDiff.derivative(y->r(y), x) - grad_x(r, x) ) < 1e-10

# Test hessian
@test abs(ForwardDiff.derivative(z->ForwardDiff.derivative(y->r(y), z),x) - hess_x(r, x) ) < 1e-10

# Test gradient of log evaluation
@test abs(ForwardDiff.derivative(y->log(r(y)), x) - grad_x_logeval(r, x) ) < 1e-10

# Test hessian of log evaluation
@test abs(ForwardDiff.hessian(y->log(r(y[1])), [x])[1,1] - hess_x_logeval(r, x) ) < 1e-10

end

@testset "Rectifier exponential" begin

r = Rectifier("exponential")
@test r.T == "exponential"

    x = 2.46
@test abs(r(x) - exp(x))<1e-10


# Test gradient
@test abs(ForwardDiff.derivative(y->r(y), x) - grad_x(r, x) ) < 1e-10

# Test hessian
@test abs(ForwardDiff.derivative(z->ForwardDiff.derivative(y->r(y), z),x) - hess_x(r, x) ) < 1e-10

# Test gradient of log evaluation
@test abs(ForwardDiff.derivative(y->log(r(y)), x) - grad_x_logeval(r, x) ) < 1e-10

# Test hessian of log evaluation
@test abs(ForwardDiff.hessian(y->log(r(y[1])), [x])[1,1] - hess_x_logeval(r, x) ) < 1e-10

x = -2.46
@test abs(r(x) - exp(x))<1e-10


# Test gradient
@test abs(ForwardDiff.derivative(y->r(y), x) - grad_x(r, x) ) < 1e-10

# Test hessian
@test abs(ForwardDiff.derivative(z->ForwardDiff.derivative(y->r(y), z),x) - hess_x(r, x) ) < 1e-10

# Test gradient of log evaluation
@test abs(ForwardDiff.derivative(y->log(r(y)), x) - grad_x_logeval(r, x) ) < 1e-10

# Test hessian of log evaluation
@test abs(ForwardDiff.hessian(y->log(r(y[1])), [x])[1,1] - hess_x_logeval(r, x) ) < 1e-10

end

@testset "Rectifier sigmoid" begin

r = Rectifier("sigmoid")
@test r.T == "sigmoid"

x = 0.4
@test abs(r(x) - exp(x)/(1+exp(x)))<1e-10


# Test gradient
@test abs(ForwardDiff.derivative(y->r(y), x) - grad_x(r, x) ) < 1e-10

# Test hessian
@test abs(ForwardDiff.derivative(z->ForwardDiff.derivative(y->r(y), z),x) - hess_x(r, x) ) < 1e-10

# Test gradient of log evaluation
@test abs(ForwardDiff.derivative(y->log(r(y)), x) - grad_x_logeval(r, x) ) < 1e-10

# Test hessian of log evaluation
@test abs(ForwardDiff.hessian(y->log(r(y[1])), [x])[1,1] - hess_x_logeval(r, x) ) < 1e-10


x = -0.4
@test abs(r(x) - exp(x)/(1+exp(x)))<1e-10

# Test gradient
@test abs(ForwardDiff.derivative(y->r(y), x) - grad_x(r, x) ) < 1e-10

# Test hessian
@test abs(ForwardDiff.derivative(z->ForwardDiff.derivative(y->r(y), z),x) - hess_x(r, x) ) < 1e-10

# Test gradient of log evaluation
@test abs(ForwardDiff.derivative(y->log(r(y)), x) - grad_x_logeval(r, x) ) < 1e-10

# Test hessian of log evaluation
@test abs(ForwardDiff.hessian(y->log(r(y[1])), [x])[1,1] - hess_x_logeval(r, x) ) < 1e-10

end

@testset "Rectifier sigmoid_" begin

    Kmin = 1e-4
    Kmax = 1e7
    ϵ = 1e-9

    r = Rectifier("sigmoid_"; Kmin = Kmin, Kmax = Kmax)
    @test r.T == "sigmoid_"
    
    x = 0.4
    @test abs(r(x) - (Kmin + (Kmax - Kmin)*exp(x)/(1+exp(x))))<ϵ
    
    
    # Test gradient
    @test abs(ForwardDiff.derivative(y->r(y), x) - grad_x(r, x) ) < ϵ
    
    # Test hessian
    @test abs(ForwardDiff.derivative(z->ForwardDiff.derivative(y->r(y), z),x) - hess_x(r, x) ) < ϵ
    
    # Test gradient of log evaluation
    @test abs(ForwardDiff.derivative(y->log(r(y)), x) - grad_x_logeval(r, x) ) < ϵ
    
#     # Test hessian of log evaluation
    @test abs(ForwardDiff.hessian(y->log(r(y[1])), [x])[1,1] - hess_x_logeval(r, x) ) < ϵ
    
    
    x = -0.4
    @test abs(r(x) - (Kmin + (Kmax - Kmin)*exp(x)/(1+exp(x))))<ϵ
    
    # Test gradient
    @test abs(ForwardDiff.derivative(y->r(y), x) - grad_x(r, x) ) < ϵ
    
    # Test hessian
    @test abs(ForwardDiff.derivative(z->ForwardDiff.derivative(y->r(y), z),x) - hess_x(r, x) ) < ϵ
    
    # Test gradient of log evaluation
    @test abs(ForwardDiff.derivative(y->log(r(y)), x) - grad_x_logeval(r, x) ) < ϵ
    
    # Test hessian of log evaluation
    @test abs(ForwardDiff.hessian(y->log(r(y[1])), [x])[1,1] - hess_x_logeval(r, x) ) < ϵ
    
end


@testset "Rectifier softplus" begin

r = Rectifier("softplus")
@test r.T == "softplus"

    x = 2.5
    a = log(2)
@test abs(r(x) - (log(1 + exp(-abs(a*x))) + max(a*x, 0))/a)<1e-10

#Test inversion
@test abs(inverse(r,r(x))-x)<1e-10
@test abs(r(inverse(r,x))-x)<1e-10

#Test inversion with x large
xlarge = 10.0^6
@test abs(inverse(r,xlarge) -xlarge)<1e-10

# Test gradient
@test abs(ForwardDiff.derivative(y->r(y), x) - grad_x(r, x) ) < 1e-10

# Test hessian
@test abs(ForwardDiff.derivative(z->ForwardDiff.derivative(y->r(y), z),x) - hess_x(r, x) ) < 1e-10

# Test gradient of log evaluation
@test abs(ForwardDiff.derivative(y->log(r(y)), x) - grad_x_logeval(r, x) ) < 1e-10

# Test hessian of log evaluation
@test abs(ForwardDiff.hessian(y->log(r(y[1])), [x])[1,1] - hess_x_logeval(r, x) ) < 1e-10


x = -2.5
a = log(2)
@test abs(r(x) - (log(1 + exp(-abs(a*x))) + max(a*x, 0))/a)<1e-10


#Test inversion with x large
xlarge = 10.0^6
@test abs(inverse(r,xlarge) -xlarge)<1e-10

# Test gradient
@test abs(ForwardDiff.derivative(y->r(y), x) - grad_x(r, x) ) < 1e-10

# Test hessian
@test abs(ForwardDiff.derivative(z->ForwardDiff.derivative(y->r(y), z),x) - hess_x(r, x) ) < 1e-10

# Test gradient of log evaluation
@test abs(ForwardDiff.derivative(y->log(r(y)), x) - grad_x_logeval(r, x) ) < 1e-10

# Test hessian of log evaluation
@test abs(ForwardDiff.hessian(y->log(r(y[1])), [x])[1,1] - hess_x_logeval(r, x) ) < 1e-10

end

@testset "Rectifier explinearunit" begin

r = Rectifier("explinearunit")
@test r.T == "explinearunit"

## Test x<0
xneg = -0.5
@test abs(r(xneg) - exp(xneg))<1e-10

# Test gradient
@test abs(ForwardDiff.derivative(y->r(y), xneg) - grad_x(r, xneg) ) < 1e-10
@test abs(ForwardDiff.derivative(y->r(y), xneg) - exp(xneg) ) < 1e-10


# Test hessian
@test abs(ForwardDiff.derivative(z->ForwardDiff.derivative(y->r(y), z),xneg) - hess_x(r, xneg) ) < 1e-10
@test abs(ForwardDiff.derivative(z->ForwardDiff.derivative(y->r(y), z),xneg) - exp(xneg) ) < 1e-10

# Test gradient of log evaluation
@test abs(ForwardDiff.derivative(y->log(r(y)), xneg) - grad_x_logeval(r, xneg) ) < 1e-10

# Test hessian of log evaluation
@test abs(ForwardDiff.hessian(y->log(r(y[1])), [xneg])[1,1] - hess_x_logeval(r, xneg) ) < 1e-10

## Test x>=0
x = 2.5
@test abs(r(x) - (x + 1.0))<1e-10

#Test inversion
@test abs(inverse(r,r(x))-x)<1e-10
@test abs(r(inverse(r,x))-x)<1e-10

# Test gradient
@test abs(ForwardDiff.derivative(y->r(y), x) - grad_x(r, x) ) < 1e-10
@test abs(ForwardDiff.derivative(y->r(y), x) - 1.0 ) < 1e-10

# Test hessian
@test abs(ForwardDiff.derivative(z->ForwardDiff.derivative(y->r(y), z),x) - hess_x(r, x) ) < 1e-10
@test abs(ForwardDiff.derivative(z->ForwardDiff.derivative(y->r(y), z),x) - 0.0) < 1e-10

# Test gradient of log evaluation
@test abs(ForwardDiff.derivative(y->log(r(y)), x) - grad_x_logeval(r, x) ) < 1e-10

# Test hessian of log evaluation
@test abs(ForwardDiff.hessian(y->log(r(y[1])), [x])[1,1] - hess_x_logeval(r, x) ) < 1e-10

end
