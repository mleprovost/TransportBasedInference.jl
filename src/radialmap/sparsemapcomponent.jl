export SparseUk, component, construct, evaluate, off_diagonal, set_id

#### Structure for the k-th component SparseUk of the lower triangular map U

# For now Uk = Σi=1,k ui(k)

# The last array for ξ, σ and a will be of size p+2, p+2, p+3 versus p, p, p+1
# for the components i=1,...,k-1

struct SparseUk
        k::Int64
        p::Array{Int64,1}
        ξ::Array{Array{Float64,1}, 1}
        σ::Array{Array{Float64,1}, 1}
        a::Array{Array{Float64,1}, 1}

        # Cache for evaluation of function, gradient and hessian on basis

        # Inner constructor
        function SparseUk(k::Int64, p::Array{Int64,1}, ξ::Array{Array{Float64,1},1}, σ::Array{Array{Float64,1},1}, a::Array{Array{Float64,1},1})

        @assert size(ξ,1)==k "Size of ξ doesn't match the order of Uk"
        @assert size(σ,1)==k "Size of σ doesn't match the order of Uk"
        @assert size(a,1)==k "Size of a doesn't match the order of Uk"
                return new(k, p, ξ, σ, a)
        end
end

function SparseUk(k::Int64, p::Array{Int64,1})
        @assert k>=1 "k must be >0"

        if k==1
                if p[1]==-1
                return SparseUk(k, p, [Float64[]], [Float64[]],  [Float64[]])
                end
                if p[1]==0
                return SparseUk(k, p, [Float64[]], [Float64[]],  [zeros(2)])
                end
                if p[1]>0
                return SparseUk(k, p, [zeros(p[1]+2)], [ones(p[1]+2)], [zeros(p[1]+3)])
                end

        else # k>1
                ξ = Array{Float64,1}[]
                σ = Array{Float64,1}[]
                a = Array{Float64,1}[]
                # Fill Off Diagonal components
                for pj in p[1:end-1]
                if pj==-1
                        # Null component
                        push!(ξ, Float64[])
                        push!(σ, Float64[])
                        push!(a, Float64[])
                elseif pj==0
                        # Linear function
                        push!(ξ, Float64[])
                        push!(σ, Float64[])
                        push!(a, zeros(1))
                else    # Linear function + p rbf functions
                        push!(ξ, zeros(pj))
                        push!(σ, ones(pj))
                        push!(a, zeros(pj+1))
                end
                end

                # Fill Diagonal Component
                if p[end]==-1
                        # Null component
                        push!(ξ, Float64[])
                        push!(σ, Float64[])
                        push!(a, Float64[])
                elseif p[end]==0
                        # Affine function
                        push!(ξ, Float64[])
                        push!(σ, Float64[])
                        push!(a, zeros(2))
                else
                        #Constant + p+2 ψ functions
                        push!(ξ, zeros(p[end]+2))
                        push!(σ, ones(p[end]+2))
                        push!(a, zeros(p[end]+3))
                end
                return SparseUk(k, p, ξ, σ, a)
        end
end

SparseUk(k::Int64, p::Int64) = SparseUk(k, fill(p, k))



function show(io::IO, Vk::SparseUk)
if Vk.k==1
println(io,"$(Vk.k)-st component of a Sparse KR rearrangement of order p = $(Vk.p)")
else
println(io,"$(Vk.k)-th component of a Sparse KR rearrangement of order p = $(Vk.p)")
end
end

function set_id(Vk::SparseUk)
        k = Vk.k
if k==1
        Vk.p[1] = -1
        Vk.ξ[1] = Float64[]
        Vk.σ[1] = Float64[]
        Vk.a[1] = Float64[]
else
        Vk.p .= rmul!(ones(k), -1.0)

        @inbounds for i=1:k
        Vk.ξ[i] = Float64[]
        Vk.σ[i] = Float64[]
        Vk.a[i] = Float64[]
        end
end
end

# Transform vector of coefficients to a form SparseUk.a
function modify_a(A::Array{Float64,1}, Vk::SparseUk)
        @assert size(A,1)==size(extract_a(Vk),1) "A doesn't match the size of Vk"
        @get Vk (k, p)
                count = 0
        if k==1
                Vk.a[1] .= A
                count += size(A,1)
                # Vk.a[1] .= A
        else
                # count will store the last position of each assignment

                # Off-diagonal components
                @inbounds for idx=1:k-1
                pidx = p[idx]
                if pidx==-1
                        #No assignment
                elseif pidx==0
                        Vk.a[idx] .= [A[count+1]]
                        count += 1
                else
                        tmp_ai  = view(Vk.a, idx)[1]
                        tmp_ai .= view(A, count+1:count+(pidx+1))
                        count += deepcopy(pidx)+1
                end
                end

                # Diagonal components
                pidx = p[k]
                if pidx==-1
                        #No assignment, set to the identity without coefficient
                elseif pidx==0
                        # Affine function with two coefficients
                        tmp_ak  = view(Vk.a,k)[1]
                        tmp_ak .= view(A, count+1:count+2)

                        count +=2
                else
                        # Constant + p+2 ψ functions
                        tmp_ak  = view(Vk.a,k)[1]
                        tmp_ak .= view(A, count+1:count+pidx+3)
                        count += deepcopy(pidx)+3
                        # Vk.a[k] .= view(A, (k-1)*(p+1)+1:k*(p+1)+2)
                        # Vk.a[k] .= A[(k-1)*(p+1)+1:k*(p+1)+2]
                end
        end
        @assert count == size(A,1) "Error in the assignment of the coefficients"
end

extract_a(Vk::SparseUk) = vcat(Vk.a...)


size(Vk::SparseUk) = (Vk.k, Vk.p)

# Evaluate the map SparseUk at z = (x,...,x)
function (Vk::SparseUk)(z::T) where {T<:Real}
        @get Vk (k, p)

        out = uk(p[k], Vk.ξ[k], Vk.σ[k], Vk.a[k])(z)
        if k>1
                for i=1:k-1
                @inbounds out += ui(p[i], Vk.ξ[i], Vk.σ[i], Vk.a[i])(z)
                end
        end
        return out
end

# Evaluate the map SparseUk at z = (z1,...,zk)
# Optimized with views
function (Vk::SparseUk)(z)
        @get Vk (k, p)
        @assert size(z,1)<=k  "The vector z has more components than Uk"
        if p[k]<1
        out = uk(p[k], Vk.ξ[k], Vk.σ[k], Vk.a[k])(z[k])
        else
        out = uk(p[k], view(Vk.ξ,k)[1], view(Vk.σ,k)[1], view(Vk.a,k)[1])(z[k])
        end

        if k>1
                @inbounds for idx=1:k-1
                pidx = p[idx]

                if pidx==-1
                # ui(z) = 0.0 can skip this
                elseif pidx==0
                out += ui(pidx, Vk.ξ[idx], Vk.σ[idx], Vk.a[idx])(z[idx])
                else
                out += ui(pidx, view(Vk.ξ,idx)[1], view(Vk.σ,idx)[1], view(Vk.a,idx)[1])(z[idx])
                end
                end
        end
        return out
end


# Evaluate the map Vk at z = (z1,...,zk)
# Optimized with views
function off_diagonal(Vk::SparseUk, z)
        @get Vk (k, p)
        @assert size(z,1)<=k  "The vector z has more components than Uk"
        out = 0.0
        if k>1
        @inbounds for idx=1:k-1
                        pidx = p[idx]
                        if pidx==-1
                                # No computation
                        elseif pidx==0
                        out += ui(pidx, Vk.ξ[idx], Vk.σ[idx], Vk.a[idx])(z[idx])
                        else
                        out += ui(pidx, view(Vk.ξ,idx)[1], view(Vk.σ,idx)[1], view(Vk.a,idx)[1])(z[idx])
                        end
                end
        end
        return out
end

function component(Vk::SparseUk, idx::Int64)
        @get Vk (k, p)
        @assert k>= idx "Can't access this component idx>k"
        if p[idx]<1
                if idx==k
                        return uk(p[idx], Vk.ξ[idx], Vk.σ[idx], Vk.a[idx])
                else
                        return ui(p[idx], Vk.ξ[idx], Vk.σ[idx], Vk.a[idx])
                end
        else
                if idx==k
                        return uk(p[idx], Vk.ξ[idx], Vk.σ[idx], Vk.a[idx])
                else
                        return ui(p[idx], Vk.ξ[idx], Vk.σ[idx], Vk.a[idx])
                end
        end
end
