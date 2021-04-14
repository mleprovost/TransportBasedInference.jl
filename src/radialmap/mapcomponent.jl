export Uk, component, construct, evaluate, off_diagonal

#### Structure for the k-th component Uk of the lower triangular map U

# For now Uk = Σi=1,k ui(k)

# The last array for ξ, σ and a will be of size p+2, p+2, p+3 versus p, p, p+1
# for the components i=1,...,k-1

struct Uk#{T,S} where {T, S}
        k::Int64
        p::Int64
        ξ::Array{Array{Float64,1}, 1}#{Union{Array{Float64,1}, 1}
        σ::Array{Array{Float64,1}, 1}
        a::Array{Array{Float64,1}, 1}

        # Cache for evaluation of function, gradient and hessian on basis

        # Inner constructor
        function Uk(k::Int64, p::Int64, ξ::Array{Array{Float64,1},1}, σ::Array{Array{Float64,1},1}, a::Array{Array{Float64,1},1})

        @assert size(ξ,1)==k "Size of ξ doesn't match the order of Uk"
        @assert size(σ,1)==k "Size of σ doesn't match the order of Uk"
        @assert size(a,1)==k "Size of a doesn't match the order of Uk"
                return new(k, p, ξ, σ, a)
        end
end

function Uk(k::Int64, p::Int64)
        @assert k>=1 "k must be >0"
        @assert p>=0 "p must be >=0"

        if k==1
                if p>0
                return Uk(k, p, [zeros(p+2)], [ones(p+2)], [zeros(p+3)])
                end

                if p==0
                return Uk(k, p, [Float64[]], [Float64[]],  [zeros(2)])
                end
        end
        if k>1 && p>0
                return Uk(k, p, push!([zeros(p) for i=1:k-1], zeros(p+2)),
                                push!([ones(p) for i=1:k-1], ones(p+2)),
                                push!([zeros(p+1) for i=1:k-1], zeros(p+3)))
        end

        if k>1 && p==0
                return Uk(k, p, [Float64[] for i=1:k],
                                [Float64[] for i=1:k],
                                push!([zeros(1) for i=1:k-1], zeros(2)))
        end
end

# Function to construct
function construct(k::Int64, p::Int64, ξ::Array{Array{Float64,1},1}, σ::Array{Array{Float64,1},1})

if k==1
        f = [construct_uk(p, ξ[1], σ[1])]
else
        f = [construct_ui(p, ξ[i], σ[i]) for i=1:k-1]
        push!(f, construct_uk(p, ξ[k], σ[k]))
end
return f
end


function Base.show(io::IO, Vk::Uk)
if Vk.k==1
println(io,"$(Vk.k)-st component of a KR rearrangement of order p = $(Vk.p)")
else
println(io,"$(Vk.k)-th component of a KR rearrangement of order p = $(Vk.p)")
end
end


# Transform vector of coefficients to a form Uk.a
function modify_a(A::Array{Float64,1}, Vk::Uk)
        @get Vk (k, p)
        if k==1
                Vk.a[1] .= A
                # Vk.a[1] .= A
        else
                if p==0
                        for idx=1:k-1
                        @inbounds Vk.a[idx] .= [A[idx]]
                        end
                        tmp_ak = view(Vk.a,k)[1]
                        tmp_ak .= view(A, k:k+1)
                        # Vk.a[k] .= A[end-1:end]
                        # Vk.a[k] .= A[end-1:end]
                else
                        @inbounds  for idx=1:k-1
                        tmp_ai = view(Vk.a, idx)[1]
                        tmp_ai .= view(A, (idx-1)*(p+1)+1:idx*(p+1))
                        # @inbounds Vk.a[idx] .= view(A, (idx-1)*(p+1)+1:idx*(p+1))
                        # @inbounds Vk.a[idx] .= A[(idx-1)*(p+1)+1:idx*(p+1)]
                        end
                        tmp_ak  = view(Vk.a,k)[1]
                        tmp_ak .= view(A, (k-1)*(p+1)+1:k*(p+1)+2)
                        # Vk.a[k] .= view(A, (k-1)*(p+1)+1:k*(p+1)+2)
                        # Vk.a[k] .= A[(k-1)*(p+1)+1:k*(p+1)+2]
                end
        end
end

extract_a(Vk::Uk) = vcat(Vk.a...)


function evaluate(Vk::Uk, z::Array{T,1}) where {T<:Real}
        @get Vk (k, p)
        @assert k==size(z,1) "Wrong dimension of z"
        A = extract_a(Vk)

        n = size(A,1)
        basis = zeros(n)

        if p==0
                if k==1
                basis .= [1; z[1]]
                else
                basis[1:n-1] .= z
                basis[n] = 1.0
                basis[n+1] = z[end]
                end
        else
                wo = zeros(p+1)
                wd = zeros(p+3)
                #Off diagonal component
                for i=1:k-1
                        weights(component(Vk, i), z[i], wo)
                        basis[(i-1)*(p+1)+1:i*(p+1)] .= deepcopy(wo)
                end
                # Diagonal component
                weights(component(Vk, k), z[k], wd)
                basis[(k-1)*(p+1)+1:k*(p+1)+2] .= deepcopy(wd)
        end
        return basis
end


size(Vk::Uk) = (Vk.k, Vk.p)

# Evaluate the map Uk at z = (x,...,x)
function (Vk::Uk)(z::T) where {T<:Real}
        @get Vk (k, p)
        out = uk(p, Vk.ξ[k], Vk.σ[k], Vk.a[k])(z)
        if k>1
                for i=1:k-1
                @inbounds out += ui(p, Vk.ξ[i], Vk.σ[i], Vk.a[i])(z)
                end
        end
        return out
end

# Evaluate the map Uk at z = (z1,...,zk)
# Optimized with views
function (Vk::Uk)(z)
        @get Vk (k, p)
        @assert size(z,1)<=k  "The vector z has more components than Uk"
        if p==0
        out = uk(p, Vk.ξ[k], Vk.σ[k], Vk.a[k])(z[end])
        else
        out = uk(p, view(Vk.ξ,k)[1], view(Vk.σ,k)[1], view(Vk.a,k)[1])(z[end])
        end

        if k>1
                if p==0
                for i=1:k-1
                @inbounds out += ui(p, Vk.ξ[i], Vk.σ[i], Vk.a[i])(z[i])
                end
                else
                for i=1:k-1
                @inbounds out += ui(p, view(Vk.ξ,i)[1], view(Vk.σ,i)[1], view(Vk.a,i)[1])(z[i])
                end
                end
        end
        return out
end


# Evaluate the map Uk at z = (z1,...,zk)
# Optimized with views
function off_diagonal(Vk::Uk, z)
        @get Vk (k, p)
        @assert size(z,1)<=k  "The vector z has more components than Uk"
        out = 0.0
        if k>1
                if p==0
                for i=1:k-1
                @inbounds out += ui(p, Vk.ξ[i], Vk.σ[i], Vk.a[i])(z[i])
                end
                else
                for i=1:k-1
                @inbounds out += ui(p, view(Vk.ξ,i)[1], view(Vk.σ,i)[1], view(Vk.a,i)[1])(z[i])
                end
                end
        end
        return out
end

function component(Vk::Uk, idx::Int64)
        @get Vk (k, p)
        @assert k>= idx "Can't access this component idx>k"
        if p==0
                if idx==k
                        return uk(p, Vk.ξ[idx], Vk.σ[idx], Vk.a[idx])
                else
                        return ui(p, Vk.ξ[idx], Vk.σ[idx], Vk.a[idx])
                end
        else
                if idx==k
                        return uk(p, view(Vk.ξ,idx)[1], view(Vk.σ,idx)[1], view(Vk.a,idx)[1])
                else
                        return ui(p, view(Vk.ξ,idx)[1], view(Vk.σ,idx)[1], view(Vk.a,idx)[1])
                end
        end
end




# Compute the last component uk of Uk
# uk(Vk::Uk{k,p}) where {k, p} = uk(k,p, Vk.ξ[k], Vk.σ[k], Vk.a[k])
#
#
# ∇k(Vk::Uk{k,p}) where {k, p} = ∇k(uk(Vk))


# function construct(Vk::Uk)
#         @get Vk (k, p)
#         f = separablefcn[]
#         if k>1
#                 for i=1:k-1
#                 @inbounds push!(f, deepcopy(ui(p, Vk.ξ[i], Vk.σ[i], Vk.a[i])))
#                 end
#         end
#         push!(f, deepcopy(uk(p, Vk.ξ[k], Vk.σ[k], Vk.a[k])))
#         return f
# end
