
# This test shows the interest to update the QR basis instead
# of computing from scratch at every time step

# function timing()
#     @btime begin
#         m = 1000
#         n = 50
#
#         A = randn(m, n)
#         qr(A)
#         for i=1:20
#             v = randn(m)
#             A = hcat(A, reshape(v,(m,1)))
#         end
#     end
#     @btime begin
#         m = 1000
#         n = 50
#
#         A = randn(m, n)
#         qr(A)
#         for i=1:20
#             v = randn(m)
#             A = hcat(A, reshape(v,(m,1)))
#             qr(A)
#         end
#     end
#     @btime begin
#                 m = 1000
#         n = 50
#
#         A = randn(m, n)
#         F = qr(A)
#         R = F.R
#         for i=1:20
#             v = randn(m)
#             R = qraddcol(A, R, v)
#
#             A = hcat(A, reshape(v,(m,1)))
#         end
#     end
# end
#
# 1.364 ms (86 allocations: 10.18 MiB)
# 17.929 ms (194 allocations: 20.08 MiB)
# 2.417 ms (368 allocations: 11.47 MiB)
