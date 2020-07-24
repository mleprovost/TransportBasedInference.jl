import LinearAlgebra: ldiv!, dot

export Preconditioner

struct Preconditioner
    P::Symmetric{Float64}

    F::Cholesky{Float64, Matrix{Float64}}
end

function Preconditioner(P::Matrix{Float64})
    return Preconditioner(Symmetric(P), cholesky(Symmetric(P)))
end

ldiv!(x, P::Preconditioner, b) = copyto!(x, P.F \ b)
dot(A::Array, P::Preconditioner, B::Vector) = dot(A, P.P, B)
