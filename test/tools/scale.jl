

@testset "Linear rescaling of multi-dimensional sample" begin

Nx = 1
Ne = 100
ens = EnsembleState(Nx, Ne)

ens.S .= 1.0 .+ 5.0*randn(Nx, Ne)

scale!(ens; diag = true)

@test norm(mean(ens))<1e-10
@test norm(cov(ens) .- 1.0)<1e-10


Nx = 10
Ne = 100
ens = EnsembleState(Nx, Ne)

ens.S .= randn(Nx) .+ rand(N).*randn(Nx, Ne)

scale!(ens; diag = true)

@test norm(mean(ens))<1e-10
@test norm(cov(ens) .- 1.0)<1e-10



end
