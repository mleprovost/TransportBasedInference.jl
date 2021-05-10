
@testset "Bracket root" begin
    u = uk(3)
    u.ξk .= sort!(randn(5))
    u.σk .= σscale(u.ξk,2.0)
    u.coeffk .= rand(6);
    κ = 5.0
    xlim = (u.ξk[1]-κ*u.σk[1], u.ξk[end]+κ*u.σk[end])

    zlim = bracket(u, xlim[1], xlim[2])

    @test u(zlim[1])*u(zlim[2])<0.0

end
@testset "inverse_uk p=-1" begin
    u = uk(-1)
    zopt = inverse_uk(u,2.0,500.0)
    @test zopt == 2.0
    @test u(zopt) == 2.0
end

@testset "inverse_uk p=0" begin
    u = uk(0)
    u.coeffk .= [1.0;2.0]
    zopt = inverse_uk(u,2.0,500.0)
    @test zopt ==(2.0-1.0)/2.0
    @test u(zopt) == 2.0
end

@testset "inverse_uk p>0" begin
    u = uk(5)
    u.ξk .= sort!(randn(7))
    u.σk .= σscale(u.ξk,2.0)
    u.coeffk .= rand(8);

    zopt = inverse_uk(u,1.0,500.0)
    @test norm(u(zopt) -1.0)<1e-9

    # u = uk(5, randn(7), rand(7), [-0.5; rand(7)])
    #
    # zopt = inverse_uk(u,1.0;z0 = 0.1)
    # @test norm(u(zopt) -1.0)<1e-10
end

@testset "Invert S" begin
    Nx = 100
    Ne = 200
    p = 2
    γ = 2.0
    λ = 0.1
    δ = 1e-5
    κ = 10.0

    X = randn(Nx,Ne) .* randn(Nx,Ne)
    Xpost = zero(X)

    S = RadialMap(Nx, p, γ=γ, λ=λ, δ=δ, κ=κ)
    optimize(S, X);

    F = S(X)
    # Truth is a weak pertrubation
    ystar = X[1:50,1] + 0.01*cos.(randn(50))
    for i=1:Ne
        col = view(Xpost,:,i)
        inverse(col, view(F,:,i), S, ystar)
    end

    @test norm(S(Xpost)[51:Nx,:] - F[51:Nx,:])<1e-8
end


@testset "Invert S with Multi-threading" begin
    Nx = 100
    Ne = 200
    p = 2
    γ = 2.0
    λ = 0.1
    δ = 1e-5
    κ = 10.0

    X = randn(Nx,Ne) .* randn(Nx,Ne)
    Xserial = zero(X)
    Xthread = zero(X)
    Sserial = RadialMap(Nx, p, γ=γ, λ=λ, δ=δ, κ=κ)
    Sthread = RadialMap(Nx, p, γ=γ, λ=λ, δ=δ, κ=κ)
    optimize(Sserial, X; P = serial);
    optimize(Sthread, X; P = thread);

    F = Sserial(X)
    @test norm(Sserial(X)-Sthread(X))<1e-12
    # Truth is a weak pertrubation
    ystar = X[1:50,1] + 0.01*cos.(randn(50))
    Threads.@threads for i=1:Ne
        colserial = view(Xserial,:,i)
        colthread = view(Xthread,:,i)
        inverse(colserial, view(F,:,i), Sserial, ystar)
        inverse(colthread, view(F,:,i), Sthread, ystar)

    end

    @test norm(Sserial(Xserial)[51:Nx,:] - F[51:Nx,:])<1e-8
    @test norm(Sthread(Xthread)[51:Nx,:] - F[51:Nx,:])<1e-8

end

#
# @testset "Benchmark Algorithm" begin
# function compare_algo()
# @btime begin
# for i=1:1000
# u = uk(3)
# u.ξk .= sort!(randn(5))
# u.σk .= σscale(u.ξk,2.0)
# u.coeffk .= rand(6);
# κ = 5.0
# xlim = (u.ξk[1]-κ*u.σk[1], u.ξk[end]+κ*u.σk[end])
# zlim = bracket(u, xlim[1], xlim[2])
# end
# end
#
# @btime begin
# for i=1:1000
# u = uk(3)
# u.ξk .= sort!(randn(5))
# u.σk .= σscale(u.ξk,2.0)
# u.coeffk .= rand(6);
# κ = 5.0
# xlim = (u.ξk[1]-κ*u.σk[1], u.ξk[end]+κ*u.σk[end])
# zlim = bracket(u, xlim[1], xlim[2])
# Roots.bisection(u, zlim[1], zlim[2])
# end
# end
#
# @btime begin
# for i=1:1000
# u = uk(3)
# u.ξk .= sort!(randn(5))
# u.σk .= σscale(u.ξk,2.0)
# u.coeffk .= rand(6);
# κ = 5.0
# xlim = (u.ξk[1]-κ*u.σk[1], u.ξk[end]+κ*u.σk[end])
# zlim = bracket(u, xlim[1], xlim[2])
# find_zero(u, zlim, Bisection())
# end
# end
#
# #
# # @btime begin
# # for i=1:1000
# # u = uk(3)
# # u.ξk .= sort!(randn(5))
# # u.σk .= σscale(u.ξk,2.0)
# # u.coeffk .= rand(6);
# # κ = 5.0
# # xlim = (u.ξk[1]-κ*u.σk[1], u.ξk[end]+κ*u.σk[end])
# # zlim = bracket(u, xlim[1], xlim[2])
# # find_zero(u, zlim, Roots.A42())
# # end
# # end
#
# @btime begin
# for i=1:1000
# u = uk(3)
# u.ξk .= sort!(randn(5))
# u.σk .= σscale(u.ξk,2.0)
# u.coeffk .= rand(6);
# κ = 5.0
# xlim = (u.ξk[1]-κ*u.σk[1], u.ξk[end]+κ*u.σk[end])
# zlim = bracket(u, xlim[1], xlim[2])
# find_zero(u, zlim, Roots.Brent())
# end
# end
# #
# # @btime begin
# # for i=1:1000
# # u = uk(3)
# # u.ξk .= sort!(randn(5))
# # u.σk .= σscale(u.ξk,2.0)
# # u.coeffk .= rand(6);
# # κ = 5.0
# # xlim = (u.ξk[1]-κ*u.σk[1], u.ξk[end]+κ*u.σk[end])
# # zlim = bracket(u, xlim[1], xlim[2])
# # Roots.secant_method(u, zlim)
# # end
# # end
# #
# # @btime begin
# # for i=1:1000
# # u = uk(3)
# # u.ξk .= sort!(randn(5))
# # u.σk .= σscale(u.ξk,2.0)
# # u.coeffk .= rand(6);
# # κ = 5.0
# # xlim = (u.ξk[1]-κ*u.σk[1], u.ξk[end]+κ*u.σk[end])
# # zlim = bracket(u, xlim[1], xlim[2])
# # find_zero(u, zlim, Roots.Secant())
# # end
# # end
#
# # @btime begin
# # for i=1:1000
# # u = uk(3)
# # u.ξk .= sort!(randn(5))
# # u.σk .= σscale(u.ξk,2.0)
# # u.coeffk .= rand(6);
# # κ = 5.0
# # xlim = (u.ξk[1]-κ*u.σk[1], u.ξk[end]+κ*u.σk[end])
# # zlim = bracket(u, xlim[1], xlim[2])
# # find_zero((z->u(z), D(u),H(u)), mean(zlim), Roots.Newton())
# # end
# # end
# #
# # @btime begin
# # for i=1:1000
# # u = uk(3)
# # u.ξk .= sort!(randn(5))
# # u.σk .= σscale(u.ξk,2.0)
# # u.coeffk .= rand(6);
# # κ = 5.0
# # xlim = (u.ξk[1]-κ*u.σk[1], u.ξk[end]+κ*u.σk[end])
# # zlim = bracket(u, xlim[1], xlim[2])
# # find_zero((z->u(z), D(u),H(u)), mean(zlim), Roots.Newton())
# # end
# # end
# #
# # @btime begin
# # for i=1:1000
# # u = uk(3)
# # u.ξk .= sort!(randn(5))
# # u.σk .= σscale(u.ξk,2.0)
# # u.coeffk .= rand(6);
# # κ = 5.0
# # xlim = (u.ξk[1]-κ*u.σk[1], u.ξk[end]+κ*u.σk[end])
# # zlim = bracket(u, xlim[1], xlim[2])
# # find_zero((z->u(z), D(u),H(u)), mean(zlim), Roots.Halley())
# # end
# # end
#
# end
# compare_algo()
#
# end
