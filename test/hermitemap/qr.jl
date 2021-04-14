
@testset "qraddcol with β = 0" begin
    m = 100
    A = randn(m,0)
    R = zeros(0,0)

    for i in 1:m
        a = randn(m)
        R = qraddcol(A, R, a)
        A = [A a]
        @test  norm( R'*R - A'*A ) <1e-5
    end
end

@testset "qraddcol with β > 0" begin
    m = 100
    A = randn(m,0)
    R = zeros(0,0)
    β = 0.1
    for i in 1:m
        a = randn(m)
        B = [A a]
        R = qraddcol(A, R, a, β)
        @test norm( R'*R - B'*B - β^2*I ) <1e-5
    end
end


@testset "qraddrow" begin
    m, n = 3, 3
    A = randn(m,m)
    Q, R = qr(A)
    for i in 1:100
        a = randn(m)'
        A = [A; a]
        R = qraddrow(R, a)
        @test norm( R'R - A'*A ) <1e-5
    end

end

@testset "qrdelcol" begin
    m = 100
    A = randn(m,m)
    Q, R = qr(A)
    for i in 100:-1:1
        k = rand(1:i)
        A = A[:,1:i .!= k]
        R = qrdelcol(R, k)
        @test norm( R'*R - A'*A ) <1e-5#--> less_than(1e-5)
    end
end
