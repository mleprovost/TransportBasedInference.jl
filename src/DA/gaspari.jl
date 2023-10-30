export Localization, Locgaspari, periodicmetric!, periodicmetric, cartesianmetric, gaspari



struct Localization
    L::Float64
    Gxx::Function
    Gxy::Function
    Gyy::Function
end

# function Locgaspari(d, L, periodic::Bool)
#
#     G = zeros(d,d)
#     @inbounds for i=1:d
#         # Check if the domain is periodic
#         if periodic == true
#             @inbounds for j=i:d
#                 rdom = min(abs(j - i), abs(-d + (j-1) - i), abs(d + j - i))
#                 G[i,j] = gaspari(rdom/L)
#             end
#         else
#             @inbounds for j=i:d
#             rdom = abs(j - i)
#             G[i,j] = gaspari(rdom/L)
#             end
#         end
#
#     end
#     return Symmetric(G)
# end

# Some metric for the distance between variables

periodicmetric!(i,j,d) = min(abs(j - i), abs(-d + j  - i), abs(d + j - i))
periodicmetric(d) = (i,j) -> periodicmetric!(i,j,d)

cartesianmetric(i,j) = abs(i-j)

# Construct a possibly non-square localisation matrix using
# the Gaspari-Cohn kernel
function Locgaspari(d::Tuple{Int64, Int64}, L, metric::Function)
    dx, dy = d
    G = zeros(dx,dy)
    @inbounds for i=1:dx
                 for j=1:dy
                 G[i,j] = gaspari(metric(i,j)/L)
                 end
               end
    return G
end

# Caspari-Cohn kernel, Formula found in Data assimilation in the geosciences:
# An overview of methods, issues, and perspectives
g1(r) = 1 - (5/3)*r^2 +(5/8)*r^3 +(1/2)*r^4 -0.25*r^5
g2(r) = 4 - 5r +(5/3)*r^2 + (5/8)*r^3 -(1/2)*r^4 +(1/12)*r^5 -(2/3)*r^(-1)
gaspari(r) = abs(r)>2.0 ? 0.0 : abs(r)<1.0 ? g1(abs(r)) : g2(abs(r))
