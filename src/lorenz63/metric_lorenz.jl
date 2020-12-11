export metric_lorenz, distancecircle!, distancecircle

function metric_lorenz(Nx::Int64)
    #Pairwise distance matrix
    dist = zeros(Int64, Nx, Nx)
    for i=2:Nx
        for j=1:i-1
            dist[i,j] = distancecircle!(i, j, Nx)
        end
    end
    dist  += dist'
end

function distancecircle!(i, j, Nx)
d1 = abs(i-j)

if i<j
    d2 = i + Nx - j
elseif i>j
    d2 = j + Nx -i
else
    d2 = 0
end
return minimum([d1, d2])
end

distancecircle(Nx) = (i,j)-> distancecircle!(i, j, Nx)
