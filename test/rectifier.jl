
@testset "Rectifier" begin

r = Rectifier("softplus")
@test r.T == "softplus"

    x = 2.5
    a = log(2)
@test abs(r(x) - (log(1 + exp(-abs(a*x))) + max(a*x, 0))/a)<1e-10

#Test inversion
@test abs(inverse(r,r(x))-x)<1e-10
@test abs(r(inverse(r,x))-x)<1e-10

# Test gradient
@test abs(ForwardDiff.derivative(y->r(y), x) - gradient(r, x) ) < 1e-10

# Test hessian
@test abs(ForwardDiff.derivative(z->ForwardDiff.derivative(y->r(y), z),x) - hessian(r, x) ) < 1e-10

end
