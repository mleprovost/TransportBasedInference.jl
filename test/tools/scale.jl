

@testset "Linear rescaling of multi-dimensional sample" begin

# With diagonal rescaling
Nx = 1
Ne = 100
ens = EnsembleState(Nx, Ne)

ens.S .= 1.0 .+ 5.0*randn(Nx, Ne)

AdaptiveTransportMap.scale!(ens; diag = true)

@test norm(mean(ens))<1e-10
@test norm(cov(ens) .- 1.0)<1e-10


Nx = 10
Ne = 100
ens = EnsembleState(Nx, Ne)

ens.S .= randn(Nx) .+ rand(Nx).*randn(Nx, Ne)

AdaptiveTransportMap.scale!(ens; diag = true)

@test norm(mean(ens))<1e-10
@test norm(diag(cov(ens)) .- 1.0)<1e-10




# With Cholesky refactoring

Nx = 1
Ne = 100
ens = EnsembleState(Nx, Ne)

ens.S .= 1.0 .+ 5.0*randn(Nx, Ne)

AdaptiveTransportMap.scale!(ens; diag = false)

@test norm(mean(ens))<1e-10
@test norm(cov(ens)  - I)<1e-10


Nx = 10
Ne = 500
ens = EnsembleState(Nx, Ne)

ens.S .= randn(Nx) .+ rand(Nx) .* randn(Nx, Ne)

AdaptiveTransportMap.scale!(ens; diag = false)

@test norm(mean(ens))<1e-10
@test norm(cov(ens)  - I)<1e-10

end
