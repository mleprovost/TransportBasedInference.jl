
@testset "Rectifier square" begin

r = Rectifier("squared")
@test r.T == "squared"

    x = 2.5
@test abs(r(x) - x^2)<1e-10


# Test gradient
@test abs(ForwardDiff.derivative(y->r(y), x) - gradient(r, x) ) < 1e-10

# Test hessian
@test abs(ForwardDiff.derivative(z->ForwardDiff.derivative(y->r(y), z),x) - hessian(r, x) ) < 1e-10

end

@testset "Rectifier exponential" begin

r = Rectifier("exponential")
@test r.T == "exponential"

    x = 2.46
@test abs(r(x) - exp(x))<1e-10


# Test gradient
@test abs(ForwardDiff.derivative(y->r(y), x) - gradient(r, x) ) < 1e-10

# Test hessian
@test abs(ForwardDiff.derivative(z->ForwardDiff.derivative(y->r(y), z),x) - hessian(r, x) ) < 1e-10

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
@test abs(ForwardDiff.derivative(y->r(y), x) - gradient(r, x) ) < 1e-10

# Test hessian
@test abs(ForwardDiff.derivative(z->ForwardDiff.derivative(y->r(y), z),x) - hessian(r, x) ) < 1e-10

end

@testset "Rectifier explinearunit" begin

r = Rectifier("explinearunit")
@test r.T == "explinearunit"

## Test x<0
xneg = -0.5
@test abs(r(xneg) - exp(xneg))<1e-10

# Test gradient
@test abs(ForwardDiff.derivative(y->r(y), xneg) - gradient(r, xneg) ) < 1e-10
@test abs(ForwardDiff.derivative(y->r(y), xneg) - exp(xneg) ) < 1e-10

# Test hessian
@test abs(ForwardDiff.derivative(z->ForwardDiff.derivative(y->r(y), z),xneg) - hessian(r, xneg) ) < 1e-10
@test abs(ForwardDiff.derivative(z->ForwardDiff.derivative(y->r(y), z),xneg) - exp(xneg) ) < 1e-10

## Test x>=0
x = 2.5
@test abs(r(x) - (x + 1.0))<1e-10

#Test inversion
@test abs(inverse(r,r(x))-x)<1e-10
@test abs(r(inverse(r,x))-x)<1e-10

# Test gradient
@test abs(ForwardDiff.derivative(y->r(y), x) - gradient(r, x) ) < 1e-10
@test abs(ForwardDiff.derivative(y->r(y), x) - 1.0 ) < 1e-10

# Test hessian
@test abs(ForwardDiff.derivative(z->ForwardDiff.derivative(y->r(y), z),x) - hessian(r, x) ) < 1e-10
@test abs(ForwardDiff.derivative(z->ForwardDiff.derivative(y->r(y), z),x) - 0.0) < 1e-10

end
