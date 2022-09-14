

@testset "Test logpdf of a banana distribution" begin
    @test abs(log_pdf_banana([-0.5; 2.0]) - (-3.494127066409345))<1e-8
end
