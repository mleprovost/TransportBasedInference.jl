export  center_std_diag!, center_std_off!, center_std!

# Set ξ and σ for the diagonal entry, i.e. the last element of C
function center_std_diag!(C::RadialMapComponent, X::AbstractMatrix{Float64}, γ::Float64)
    @get C (Nx, p)
    @assert (p>0 && γ >0.0) || (p==0 && γ>=0.0) "Error scaling factor γ"

    # As long as he have p>1, we have in fact p+2 centers
    if p>0
    ξ′ = view(C.ξ[end],:)
    σ′ = view(C.σ[end],:)
    tmp = view(X,Nx,:)
    quantile!(ξ′, tmp, collect(1:p+2)./(p+3); sorted=true, alpha = 0.5, beta = 0.5)

    σ′[1] = ξ′[2]-ξ′[1]
    σ′[end] = ξ′[end]-ξ′[end-1]
    @inbounds for j=2:p+1
        σ′[j] = 0.5*(ξ′[j+1]-ξ′[j-1])
    end
    rmul!(σ′, γ)

    end
end

# Set ξ and σ for the off-diagonal entries, i.e. i=1:Nx-1 entries of C
function center_std_off!(C::RadialMapComponent, X::AbstractMatrix{Float64}, γ::Float64)
    @get C (Nx, p)
    @assert (p>0 && γ >0.0) || (p==0 && γ>=0.0) "Error scaling factor γ"

    if p==0

    elseif p==1 # only one rbf
        qq = zeros(3)
        p_range = [0.25;0.5;0.75]
        @inbounds for i=1:Nx-1
            ξ = view(C.ξ[i],:)
            σ = view(C.σ[i],:)
            tmp = view(X, i, :)
            quantile!(qq, tmp, p_range; sorted=true, alpha = 0.5, beta = 0.5)
            ξ .= qq[2]
            σ .= (qq[3]-qq[1])*0.5
            rmul!(σ, γ)
        end
    elseif p==2
        p_range = collect(1.0:p)./(p+1.0)
        @inbounds for i=1:Nx-1
            ξ = view(C.ξ[i],:)
            σ = view(C.σ[i],:)
            tmp = view(X,i,:)
            quantile!(ξ, tmp, p_range; sorted=true, alpha = 0.5, beta = 0.5)
            σ[1:2] .= (ξ[2]-ξ[1])*ones(2)
            rmul!(σ, γ)
        end

    else
        p_range = collect(1.0:p)./(p+1.0)
        @inbounds for i=1:Nx-1
            ξ = view(C.ξ[i],:)
            σ = view(C.σ[i],:)
            tmp = view(X,i,:)
            quantile!(ξ, tmp, p_range; sorted=true, alpha = 0.5, beta = 0.5)

            σ[1] = (ξ[2]-ξ[1])
            σ[end] = (ξ[end]-ξ[end-1])
            for j=2:p-1
                σ[j] = 0.5*(ξ[j+1]-ξ[j-1])
            end
            rmul!(σ, γ)
        end
    end
end


# Assume an unsorted array
function center_std!(T::RadialMap, X::AbstractMatrix{Float64};start::Int64=1)
    @get T (Nx, p, γ)
    NxX, Ne = size(X)
    @assert NxX == Nx "Wrong dimension of the ensemble matrix"
    if p>0
    Xsort = deepcopy(X)
    sort!(Xsort; dims = 2)
    if Nx==1
        center_std_diag!(T.C[1], Xsort, γ)
    else
        @inbounds for i=start:Nx
            center_std_diag!(T.C[i], Xsort, γ)
            center_std_off!(T.C[i], Xsort, γ)
        end
    end
    end
end

function center_std!(C::RadialMapComponent, X::AbstractMatrix{Float64}; γ::Float64 = 2.0)
    @get C (Nx, p)
    NxX, Ne = size(X)
    @assert NxX == Nx "Wrong dimension of the ensemble matrix"
    if p>0
        Xsort = deepcopy(X)
        sort!(Xsort; dims = 2)
        if Nx==1
            center_std_diag!(C, Xsort, γ)
        else
            center_std_diag!(C, Xsort, γ)
            center_std_off!(C, Xsort, γ)
        end
    end
end




## Define center_std for Sparse maps

# Set ξ and σ for the diagonal entry, i.e. the last element of SparseC
function center_std_diag!(C::SparseRadialMapComponent, X::AbstractMatrix{Float64}, γ::Float64)
    @get C (Nx, p)
    @assert (p[Nx]>0 && γ >0.0) || (p[Nx]==0 && γ>=0.0) "Error scaling factor γ"

    # As long as he have p>1, we have in fact p+2 centers
    if p[Nx]>0
    ξ′ = view(C.ξ[end],:)
    σ′ = view(C.σ[end],:)
    tmp = view(X,Nx,:)
    quantile!(ξ′, tmp, collect(1:p[Nx]+2)./(p[Nx]+3); sorted=true, alpha = 0.5, beta = 0.5)

    σ′[1] = ξ′[2]-ξ′[1]
    σ′[end] = ξ′[end]-ξ′[end-1]
    @inbounds for j=2:p[Nx]+1
        σ′[j] = 0.5*(ξ′[j+1]-ξ′[j-1])
    end
    rmul!(σ′, γ)

    end
end


# Set ξ and σ for the off-diagonal entries, i.e. i=1:Nx-1 entries of C
function center_std_off!(C::SparseRadialMapComponent, X::AbstractMatrix{Float64}, γ::Float64, dims::Union{Int64, Array{Int64,1}, UnitRange{Int64}})
    @get C (Nx, p)
    @assert γ>=0.0 "Error scaling factor γ"
    # "The maximum of dims must be smaller than Nx-1"
    @inbounds for i in dims
        pidx = p[i]

        ξ = view(C.ξ[i],:)
        σ = view(C.σ[i],:)
        tmp = view(X, i, :)

        # Testcase according to pidx

        if pidx<1
            #There is no rbf
        elseif pidx ==1 #only one rbf
            qq = zeros(3)
            p_range = [0.25;0.5;0.75]
            quantile!(qq, tmp, p_range; sorted=true, alpha = 0.5, beta = 0.5)
            ξ .= qq[2]
            σ .= (qq[3]-qq[1])*0.5
            rmul!(σ, γ)

        elseif pidx ==2
            p_range = collect(1.0:pidx)./(pidx+1.0)
            quantile!(ξ, tmp, p_range; sorted=true, alpha = 0.5, beta = 0.5)
            σ[1:2] .= (ξ[2]-ξ[1])*ones(2)
            rmul!(σ, γ)

        else
            p_range = collect(1.0:pidx)./(pidx+1.0)
            quantile!(ξ, tmp, p_range; sorted=true, alpha = 0.5, beta = 0.5)

            σ[1] = (ξ[2]-ξ[1])
            σ[end] = (ξ[end]-ξ[end-1])
            @inbounds for j=2:pidx-1
                σ[j] = 0.5*(ξ[j+1]-ξ[j-1])
            end
            rmul!(σ, γ)
        end
    end
end

# By default compute the centers and the widths for all the dimensions
center_std_off!(C::SparseRadialMapComponent, X::AbstractMatrix{Float64}, γ::Float64) =
               center_std_off!(C, X, γ, 1:C.Nx-1)



# Assume an unsorted array
function center_std!(T::SparseRadialMap, X::AbstractMatrix{Float64}; start::Int64=1)
    @get T (Nx, p, γ)
    NxX, Ne = size(X)
    @assert NxX == Nx "Wrong dimension of the ensemble matrix"
    Xsort = sort(X; dims = 2)
    if Nx==1 && !allequal(p[Nx], -1)
        center_std_diag!(T.C[1], Xsort, γ)
    else
        @inbounds for i=start:Nx
            if !allequal(p[i], -1)
            center_std_diag!(T.C[i], Xsort, γ)
            center_std_off!(T.C[i], Xsort, γ)
            end
        end
    end
end

# Assume an unsorted array
function center_std!(C::SparseRadialMapComponent, X::AbstractMatrix{Float64}; γ::Float64 = 2.0)
    @get C (Nx, p)
    NxX, Ne = size(X)
    @assert NxX == Nx "Wrong dimension of the ensemble matrix"
    Xsort = sort(X; dims = 2)
    if Nx==1 && !allequal(p, -1)
        center_std_diag!(C, Xsort, γ)
    else
        if !allequal(p, -1)
            center_std_diag!(C, Xsort, γ)
            center_std_off!(C, Xsort, γ)
        end
    end
end


# customquantile equivalent to quantile from Statistics with alpha = beta = 0.5
# function customquantile(v::Array{Float64,1},p::Int64)
#     # From Ricardo Baptista and Youssef Marzouk
#     n= length(v)
#
#     r = zeros(p)
#     r .= collect(1:p)/(p+1)
#
#     rmul!(r,n)
#     k = zeros(Int64, size(r,1))
#     kp1 = zeros(Int64, size(r,1))
#     for (i, ri) in enumerate(r)
#         k[i] = floor(ri + 0.5)
#         kp1[i] = k[i] +1
#     end
#     # k = convert(Array{Int64,1},floor.(r .+ 0.5))
#     # kp1 = k .+1
#     r .-= k
#
#     # Find indices that are out of the range 1 to n and cap them
#     k[k .< 1] .=1
#     kp1 .= min.(kp1, n)
#
#     # Use simple linear interpolation for the valid percentages
#     return @. (0.5 + r)*v[kp1] + (0.5 - r)*v[k]
# end
#
# function customquantile(v::Array{Float64,1},p::Array{Float64,1})
#     # From Ricardo Baptista and Youssef Marzouk
#     n= length(v)
#     r = p
#     rmul!(r,n)
#     k = zeros(Int64, size(r,1))
#     kp1 = zeros(Int64, size(r,1))
#     for (i, ri) in enumerate(r)
#         k[i] = floor(ri + 0.5)
#         kp1[i] = k[i] + 1
#     end
#     # k = convert(Array{Int64,1},floor.(r .+ 0.5))
#
#     # k = convert(Array{Int64,1},floor.(r .+ 0.5))
#     # kp1 = k .+1
#     r .-= k
#
#     # Find indices that are out of the range 1 to n and cap them
#     k[k .< 1] .=1
#     kp1 .= min.(kp1, n)
#
#     # Use simple linear interpolation for the valid percentages
#     return @. (0.5 + r)*v[kp1] + (0.5 - r)*v[k]
# end
