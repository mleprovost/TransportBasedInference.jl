# export    UpdatableQR,
#           add_column!,
#           add_column_householder!,
#           remove_column!,
#           update_views,
#           qraddcol!
#
# # Credit to https://github.com/mpf/QRupdate.jl/blob/master/src/QRupdate.jl
#
# # function qraddcol(A::AbstractMatrix{T}, Rin::AbstractMatrix{T}, a::Vector{T}, β::T = 0.0) where {T <:Real}
#
# function qraddcol!(A::Array{Float64,2}, R::UpperTriangular{Float64,Array{Float64,2}}, a::Array{Float64,1}, β::Float64 = 0.0)
#
#   m, n = size(A)
#   anorm  = norm(a)
#   anorm2 = anorm^2
#   β2  = β^2
#   if β != 0
#       anorm2 = anorm2 + β2
#       anorm  = sqrt(anorm2)
#   end
#
#   if n == 0
#       return reshape([anorm], 1, 1)
#   end
#
#   # R = UpperTriangular(Rin)
#
#   c      = A'*a           # = [A' β*I 0]*[a; 0; β]
#   u      = R'\c
#   unorm2 = norm(u)^2
#   d2     = anorm2 - unorm2
#
#   if d2 > anorm2 #DISABLE 0.01*anorm2     # Cheap case: γ is not too small
#       γ = sqrt(d2)
#   else
#       z = R\u          # First approximate solution to min ||Az - a||
#       r = a - A*z
#       c = A'*r
#       if β != 0
#           c = c - β2*z
#       end
#       du = R'\c
#       dz = R\du
#       z  += dz          # Refine z
#     # u  = R*z          # Original:     Bjork's version.
#       u  += du          # Modification: Refine u
#       r  = a - A*z
#       γ = norm(r)       # Safe computation (we know gamma >= 0).
#       if β != 0
#           γ = sqrt(γ^2 + β2*norm(z)^2 + β2)
#       end
#   end
#
#   # This seems to be faster than concatenation, ie:
#   # [ Rin         u
#   #   zeros(1,n)  γ ]
#   # Rout = Array{T}(n+1, n+1)
#   # Rout[1:n,1:n] = R
#   # Rout[1:n,n+1] = u
#   # Rout[n+1,n+1] = γ
#   # Rout[n+1,1:n] = 0.0
#   R = UpperTriangular(hcat(vcat(R, zeros(1,n)),zeros(n+1,1)))
#   @show size(R)
#   # Rout = UpperTriangular(zeros(n+1, n+1))
#
#   # view(Rout,1:n,1:n) .= R.data
#   view(R,1:n,n+1) .= u
#   R[n+1,n+1] = γ
#   # fill!(view(Rout, n+1,1:n), 0.0)
#   # Rout[n+1,1:n] = 0.0
#
#   return R
#   # nothing
# end
#
#
# # Credit to https://github.com/oxfordcontrol/GeneralQP.jl/blob/master/src/linear_algebra.jl
# mutable struct UpdatableQR{T} <: Factorization{T}
#     """
#     Gives the qr factorization an (n, m) matrix as Q1*R1
#     Q2 is such that Q := [Q1 Q2] is orthogonal and R is an (n, n) matrix where R1 "views into".
#     """
#     Q::Matrix{T}
#     R::Matrix{T}
#     n::Int
#     m::Int
#
#     Q1::SubArray{T, 2, Matrix{T}, Tuple{Base.Slice{Base.OneTo{Int}}, UnitRange{Int}}, true}
#     Q2::SubArray{T, 2, Matrix{T}, Tuple{Base.Slice{Base.OneTo{Int}}, UnitRange{Int}}, true}
#     R1::UpperTriangular{T, SubArray{T, 2, Matrix{T}, Tuple{UnitRange{Int},UnitRange{Int}}, false}}
#
#     function UpdatableQR(A::AbstractMatrix{T}) where {T}
#         n, m = size(A)
#         @assert(m <= n, "Too many columns in the matrix.")
#
#         F = qr(A)
#         Q = F.Q*Matrix(I, n, n)
#         R = zeros(T, n, n)
#         R[1:m, 1:m] .= F.R
#
#         new{T}(Q, R, n, m,
#             view(Q, :, 1:m), view(Q, :, m+1:n),
#             UpperTriangular(view(R, 1:m, 1:m)))
#     end
# end
#
# function add_column!(F::UpdatableQR{T}, a::AbstractVector{T}) where {T}
#     a1 = F.Q1'*a;
#     a2 = F.Q2'*a;
#
#     x = copy(a2)
#     for i = length(x):-1:2
#         G, r = givens(x[i-1], x[i], i-1, i)
#         lmul!(G, x)
#         lmul!(G, F.Q2')
#     end
#
#     F.R[1:F.m, F.m+1] .= a1
#     F.R[F.m+1, F.m+1] = x[1]
#
#     F.m += 1; update_views!(F)
#
#     return a2
# end
#
# function add_column_householder!(F::UpdatableQR{T}, a::AbstractVector{T}) where {T}
#     a1 = F.Q1'*a;
#     a2 = F.Q2'*a;
#
#     Z = qr(a2)
#     LAPACK.gemqrt!('R','N', Z.factors, Z.T, F.Q2) # Q2 .= Q2*F.Q
#     F.R[1:F.m, F.m+1] .= a1
#     F.R[F.m+1, F.m+1] = Z.factors[1, 1]
#     F.m += 1; update_views!(F)
#
#     return Z
# end
#
# function remove_column!(F::UpdatableQR{T}, idx::Int) where {T}
#     Q12 = view(F.Q, :, idx:F.m)
#     R12 = view(F.R, idx:F.m, idx+1:F.m)
#
#     for i in 1:size(R12, 1)-1
#         G, r = givens(R12[i, i], R12[i + 1, i], i, i+1)
#         lmul!(G, R12)
#         rmul!(Q12, G')
#     end
#
#     for i in 1:F.m, j in idx:F.m-1
#         F.R[i, j] = F.R[i, j+1]
#     end
#     F.R[:, F.m] .= zero(T)
#
#     F.m -= 1; update_views!(F)
#
#     return nothing
# end
#
# function update_views!(F::UpdatableQR{T}) where {T}
#     F.R1 = UpperTriangular(view(F.R, 1:F.m, 1:F.m))
#     F.Q1 = view(F.Q, :, 1:F.m)
#     F.Q2 = view(F.Q, :, F.m+1:F.n)
# end
