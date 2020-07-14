
@testset "Test integrated function Nx = 2" begin

Nx = 2
Ne = 500
ens = EnsembleState(Nx, Ne)

ens.S .= randn(Nx, Ne)

# ens.S .=  [0.267333   1.43021;
#           0.364979   0.607224;
#          -1.23693    0.249277;
#          -2.0526     0.915629;
#          -0.182465   0.415874;
#           0.412907   1.01672;
#           1.41332   -0.918205;
#           0.766647  -1.00445]';
B = MultiBasis(CstProHermite(6; scaled =true), Nx)

idx = [0 0; 0 1; 1 0; 1 1; 1 2]
truncidx = idx[1:2:end,:]
Nψ = 5

coeff = randn(Nψ)

# coeff =   [0.6285037650645056;
#  -0.4744029092496623;
#   1.1405280011620331;
#  -0.7217760771894809;
#   0.11855056306742319]
f = ExpandedFunction(B, idx, coeff)
fp = ParametricFunction(f);
R = IntegratedFunction(fp)


# Test grad_xd
gdψ =  grad_xd(R, ens)

gdψt = zeros(Ne)

for i=1:Ne
    gdψt[i] = R.g(ForwardDiff.gradient(R.f.f, member(ens,i))[end])
end

@test norm(gdψ - gdψt) <1e-10

# Test grad_coeff_grad_xd
dψ_xd_dc = grad_coeff_grad_xd(R, ens)
dψ_xd_dct = zeros(Ne, Nψ)

for j=1:Nψ
    fj = MultiFunction(R.f.f.B, f.idx[j,:])
    ∂kfj(y) = ForwardDiff.gradient(fj, y)[end]
    ∂kf(y) = ForwardDiff.gradient(R.f.f, y)[end]
    for i=1:Ne
        dψ_xd_dct[i,j] = ∂kfj(member(ens,i))*grad_x(R.g, ∂kf(member(ens,i)))
    end
end

@test norm(dψ_xd_dc - dψ_xd_dct)<1e-10

# Test hess_coeff_grad_xd
d2ψ_xd_dc = hess_coeff_grad_xd(R, ens)
d2ψ_xd_dct = zeros(Ne, Nψ, Nψ)

for j=1:Nψ
    fj = MultiFunction(R.f.f.B, f.idx[j,:])
    ∂kfj(y) = ForwardDiff.gradient(fj, y)[end]
    ∂kf(y) = ForwardDiff.gradient(R.f.f, y)[end]
    for l=1:Nψ
    fl = MultiFunction(R.f.f.B, f.idx[l,:])
    ∂kfl(y) = ForwardDiff.gradient(fl, y)[end]

        for i=1:Ne
            d2ψ_xd_dct[i,j,l] = 0.0*grad_x(R.g, ∂kf(member(ens,i))) +
                hess_x(R.g, ∂kf(member(ens,i)))*∂kfj(member(ens,i))*∂kfl(member(ens,i))
        end
    end
end

@test norm(d2ψ_xd_dc - d2ψ_xd_dct)<1e-10



# Test integrate_xd
intψt = zeros(Ne)

for i=1:Ne
    xi = member(ens,i)
    intψt[i] = quadgk(t->R.g(ForwardDiff.gradient(y->R.f.f(y), vcat(xi[1:end-1],t))[end]), 0, xi[end],rtol = 1e-4)[1]
end

intψ = integrate_xd(R, ens)

@test norm(intψt - intψ)<1e-10


# Test grad_coeff_integrate_xd

dcintψt = zeros(Ne, Nψ)
xi = zeros(Nx)
for j=1:Nψ
    fj = MultiFunction(R.f.f.B, f.idx[j,:])
    ∂kfj(y) = ForwardDiff.gradient(fj, y)[end]
    ∂kf(y) = ForwardDiff.gradient(R.f.f, y)[end]
    for i=1:Ne
        xi .= member(ens,i)
        dcintψt[i,j] = quadgk(t->∂kfj(vcat(xi[1:end-1],t))*grad_x(R.g, ∂kf(vcat(xi[1:end-1],t))),0,xi[end])[1]
    end
end


dcintψ = grad_coeff_integrate_xd(R, ens)
@test norm(dcintψ - dcintψt)<1e-6


# Test hess_coeff_integrate_xd
d2cintψt = zeros(Ne, Nψ, Nψ)
xi = zeros(Nx)
∂kf(y) = ForwardDiff.gradient(R.f.f, y)[end]
for j=1:Nψ
    fj = MultiFunction(R.f.f.B, f.idx[j,:])
    ∂kfj(y) = ForwardDiff.gradient(fj, y)[end]

    for l=1:Nψ

        fl = MultiFunction(R.f.f.B, f.idx[l,:])
        ∂kfl(y) = ForwardDiff.gradient(fl, y)[end]

        for i=1:Ne
            xi .= member(ens,i)
#                 @show quadgk(t->∂kfj(vcat(xi[1:end-1],t))*∂kfl(vcat(xi[1:end-1],t))*hess_x(R.g, ∂kf(vcat(xi[1:end-1],t))),0,xi[end])[1]
                d2cintψt[i,j,l] = quadgk(t->∂kfj(vcat(xi[1:end-1],t))*∂kfl(vcat(xi[1:end-1],t))*hess_x(R.g, ∂kf(vcat(xi[1:end-1],t))),0,xi[end])[1]
        end
    end
end

d2cintψ = hess_coeff_integrate_xd(R, ens)

@test norm(d2cintψt - d2cintψ)<1e-10

# Test grad_coeff
dcRt = zeros(Ne, Nψ)

ens0 = deepcopy(ens)
ens0.S[end,:] .= zeros(Ne)
# ∂_c f(x_{1:k-1},0)
dcRt += evaluate_basis(R.f.f, ens0)
xi = zeros(Nx)

for j=1:Nψ
    fj = MultiFunction(R.f.f.B, f.idx[j,:])
    ∂kfj(y) = ForwardDiff.gradient(fj, y)[end]
    ∂kf(y) = ForwardDiff.gradient(R.f.f, y)[end]
    for i=1:Ne
        xi .= member(ens,i)
        dcRt[i,j] += quadgk(t->∂kfj(vcat(xi[1:end-1],t))*grad_x(R.g, ∂kf(vcat(xi[1:end-1],t))),0,xi[end])[1]

    end
end

dcR = grad_coeff(R, ens)

@test norm(dcRt - dcR)<1e-8


# Test hess_coeff
d2cRt = zeros(Ne, Nψ, Nψ)

xi = zeros(Nx)

@time for j=1:Nψ
    fj = MultiFunction(R.f.f.B, f.idx[j,:])
    ∂kfj(y) = ForwardDiff.gradient(fj, y)[end]

    for l=1:Nψ

        fl = MultiFunction(R.f.f.B, f.idx[l,:])
        ∂kfl(y) = ForwardDiff.gradient(fl, y)[end]

        for i=1:Ne
            xi .= member(ens,i)
#                 @show quadgk(t->∂kfj(vcat(xi[1:end-1],t))*∂kfl(vcat(xi[1:end-1],t))*hess_x(R.g, ∂kf(vcat(xi[1:end-1],t))),0,xi[end])[1]
                d2cRt[i,j,l] = quadgk(t->∂kfj(vcat(xi[1:end-1],t))*∂kfl(vcat(xi[1:end-1],t))*hess_x(R.g, ∂kf(vcat(xi[1:end-1],t))),0,xi[end])[1]
        end
    end
end

d2cR = hess_coeff(R, ens)

@test norm(d2cRt - d2cR)<1e-8


end



@testset "Test integrated function Nx = 3" begin

    Nx = 3
    Ne = 500
    ens = EnsembleState(Nx, Ne)

    ens.S .= randn(Nx, Ne)

    # ens.S .=  [0.267333   1.43021;
    #           0.364979   0.607224;
    #          -1.23693    0.249277;
    #          -2.0526     0.915629;
    #          -0.182465   0.415874;
    #           0.412907   1.01672;
    #           1.41332   -0.918205;
    #           0.766647  -1.00445]';
    B = MultiBasis(CstProHermite(6; scaled =true), Nx)

    # idx = [0 0; 0 1; 1 0; 1 1; 1 2]
    idx = [0 0 0 ;0  0 1; 0 1 0; 1 0 0;1 1 0; 0 1 1; 1 0 1; 1 1 1; 1 2 0; 2 1 0]

    truncidx = idx[1:2:end,:]
    Nψ = 5

    coeff = randn(Nψ)

    # coeff =   [0.6285037650645056;
    #  -0.4744029092496623;
    #   1.1405280011620331;
    #  -0.7217760771894809;
    #   0.11855056306742319]
    f = ExpandedFunction(B, idx, coeff)
    fp = ParametricFunction(f);
    R = IntegratedFunction(fp)


    # Test grad_xd
    gdψ =  grad_xd(R, ens)

    gdψt = zeros(Ne)

    for i=1:Ne
        gdψt[i] = R.g(ForwardDiff.gradient(R.f.f, member(ens,i))[end])
    end

    @test norm(gdψ - gdψt) <1e-10

    # Test grad_coeff_grad_xd
    dψ_xd_dc = grad_coeff_grad_xd(R, ens)
    dψ_xd_dct = zeros(Ne, Nψ)

    for j=1:Nψ
        fj = MultiFunction(R.f.f.B, f.idx[j,:])
        ∂kfj(y) = ForwardDiff.gradient(fj, y)[end]
        ∂kf(y) = ForwardDiff.gradient(R.f.f, y)[end]
        for i=1:Ne
            dψ_xd_dct[i,j] = ∂kfj(member(ens,i))*grad_x(R.g, ∂kf(member(ens,i)))
        end
    end

    @test norm(dψ_xd_dc - dψ_xd_dct)<1e-10

    # Test hess_coeff_grad_xd
    d2ψ_xd_dc = hess_coeff_grad_xd(R, ens)
    d2ψ_xd_dct = zeros(Ne, Nψ, Nψ)

    for j=1:Nψ
        fj = MultiFunction(R.f.f.B, f.idx[j,:])
        ∂kfj(y) = ForwardDiff.gradient(fj, y)[end]
        ∂kf(y) = ForwardDiff.gradient(R.f.f, y)[end]
        for l=1:Nψ
        fl = MultiFunction(R.f.f.B, f.idx[l,:])
        ∂kfl(y) = ForwardDiff.gradient(fl, y)[end]

            for i=1:Ne
                d2ψ_xd_dct[i,j,l] = 0.0*grad_x(R.g, ∂kf(member(ens,i))) +
                    hess_x(R.g, ∂kf(member(ens,i)))*∂kfj(member(ens,i))*∂kfl(member(ens,i))
            end
        end
    end

    @test norm(d2ψ_xd_dc - d2ψ_xd_dct)<1e-10



    # Test integrate_xd
    intψt = zeros(Ne)

    for i=1:Ne
        xi = member(ens,i)
        intψt[i] = quadgk(t->R.g(ForwardDiff.gradient(y->R.f.f(y), vcat(xi[1:end-1],t))[end]), 0, xi[end],rtol = 1e-4)[1]
    end

    intψ = integrate_xd(R, ens)

    @test norm(intψt - intψ)<1e-10


    # Test grad_coeff_integrate_xd

    dcintψt = zeros(Ne, Nψ)
    xi = zeros(Nx)
    for j=1:Nψ
        fj = MultiFunction(R.f.f.B, f.idx[j,:])
        ∂kfj(y) = ForwardDiff.gradient(fj, y)[end]
        ∂kf(y) = ForwardDiff.gradient(R.f.f, y)[end]
        for i=1:Ne
            xi .= member(ens,i)
            dcintψt[i,j] = quadgk(t->∂kfj(vcat(xi[1:end-1],t))*grad_x(R.g, ∂kf(vcat(xi[1:end-1],t))),0,xi[end])[1]
        end
    end


    dcintψ = grad_coeff_integrate_xd(R, ens)
    @test norm(dcintψ - dcintψt)<1e-6


    # Test hess_coeff_integrate_xd
    d2cintψt = zeros(Ne, Nψ, Nψ)
    xi = zeros(Nx)
    ∂kf(y) = ForwardDiff.gradient(R.f.f, y)[end]
    for j=1:Nψ
        fj = MultiFunction(R.f.f.B, f.idx[j,:])
        ∂kfj(y) = ForwardDiff.gradient(fj, y)[end]

        for l=1:Nψ

            fl = MultiFunction(R.f.f.B, f.idx[l,:])
            ∂kfl(y) = ForwardDiff.gradient(fl, y)[end]

            for i=1:Ne
                xi .= member(ens,i)
    #                 @show quadgk(t->∂kfj(vcat(xi[1:end-1],t))*∂kfl(vcat(xi[1:end-1],t))*hess_x(R.g, ∂kf(vcat(xi[1:end-1],t))),0,xi[end])[1]
                    d2cintψt[i,j,l] = quadgk(t->∂kfj(vcat(xi[1:end-1],t))*∂kfl(vcat(xi[1:end-1],t))*hess_x(R.g, ∂kf(vcat(xi[1:end-1],t))),0,xi[end])[1]
            end
        end
    end

    d2cintψ = hess_coeff_integrate_xd(R, ens)

    @test norm(d2cintψt - d2cintψ)<1e-10

    # Test grad_coeff
    dcRt = zeros(Ne, Nψ)

    ens0 = deepcopy(ens)
    ens0.S[end,:] .= zeros(Ne)
    # ∂_c f(x_{1:k-1},0)
    dcRt += evaluate_basis(R.f.f, ens0)
    xi = zeros(Nx)

    for j=1:Nψ
        fj = MultiFunction(R.f.f.B, f.idx[j,:])
        ∂kfj(y) = ForwardDiff.gradient(fj, y)[end]
        ∂kf(y) = ForwardDiff.gradient(R.f.f, y)[end]
        for i=1:Ne
            xi .= member(ens,i)
            dcRt[i,j] += quadgk(t->∂kfj(vcat(xi[1:end-1],t))*grad_x(R.g, ∂kf(vcat(xi[1:end-1],t))),0,xi[end])[1]

        end
    end

    dcR = grad_coeff(R, ens)

    @test norm(dcRt - dcR)<1e-8


    # Test hess_coeff
    d2cRt = zeros(Ne, Nψ, Nψ)

    xi = zeros(Nx)

    @time for j=1:Nψ
        fj = MultiFunction(R.f.f.B, f.idx[j,:])
        ∂kfj(y) = ForwardDiff.gradient(fj, y)[end]

        for l=1:Nψ

            fl = MultiFunction(R.f.f.B, f.idx[l,:])
            ∂kfl(y) = ForwardDiff.gradient(fl, y)[end]

            for i=1:Ne
                xi .= member(ens,i)
    #                 @show quadgk(t->∂kfj(vcat(xi[1:end-1],t))*∂kfl(vcat(xi[1:end-1],t))*hess_x(R.g, ∂kf(vcat(xi[1:end-1],t))),0,xi[end])[1]
                    d2cRt[i,j,l] = quadgk(t->∂kfj(vcat(xi[1:end-1],t))*∂kfl(vcat(xi[1:end-1],t))*hess_x(R.g, ∂kf(vcat(xi[1:end-1],t))),0,xi[end])[1]
            end
        end
    end

    d2cR = hess_coeff(R, ens)

    @test norm(d2cRt - d2cR)<1e-8
end
