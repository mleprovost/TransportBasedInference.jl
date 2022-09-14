export elementproductmatmul!

"""
    elementproductmatmul!(d, A, B, c)

Compute in-place the product `(A ∘ B)*c`, where `∘` denotes the element-wise  product of two matrices.
"""
function elementproductmatmul!(d::Array{Float64,1}, A::Array{Float64,2}, B::Array{Float64,2}, c::Array{Float64,1})
    nx, ny = size(A)
    @avx for i = 1:nx
        di = zero(eltype(d))
        for j = 1:ny
            di += (A[i, j] * B[i, j]) * c[j]
        end
        d[i] = di
    end
end
