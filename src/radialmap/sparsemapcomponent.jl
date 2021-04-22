Nxexport SparseRadialMapComponent, component, construct, evaluate, off_diagonal, set_id

#### Structure for the k-th component SparseRadialMapComponent of the lower triangular map U

# For now RadialMapComponent = Σi=1,k ui(k)

# The last array for ξ, σ and a will be of size p+2, p+2, p+3 versus p, p, p+1
# for the components i=1,...,k-1

struct SparseRadialMapComponent
        Nx::Int64
        p::Array{Int64,1}
        ξ::Array{Array{Float64,1}, 1}
        σ::Array{Array{Float64,1}, 1}
        a::Array{Array{Float64,1}, 1}

        # Cache for evaluation of function, gradient and hessian on basis

        # Inner constructor
        function SparseRadialMapComponent(Nx::Int64, p::Array{Int64,1}, ξ::Array{Array{Float64,1},1}, σ::Array{Array{Float64,1},1}, a::Array{Array{Float64,1},1})

        @assert size(ξ,1)==Nx "Size of ξ doesn't match the order of RadialMapComponent"
        @assert size(σ,1)==Nx "Size of σ doesn't match the order of RadialMapComponent"
        @assert size(a,1)==Nx "Size of a doesn't match the order of RadialMapComponent"
                return new(Nx, p, ξ, σ, a)
        end
end

function SparseRadialMapComponent(Nx::Int64, p::Array{Int64,1})
        @assert Nx>=1 "Nx must be >0"

        if Nx==1
                if p[1]==-1
                return SparseRadialMapComponent(Nx, p, [Float64[]], [Float64[]],  [Float64[]])
                end
                if p[1]==0
                return SparseRadialMapComponent(Nx, p, [Float64[]], [Float64[]],  [zeros(2)])
                end
                if p[1]>0
                return SparseRadialMapComponent(Nx, p, [zeros(p[1]+2)], [ones(p[1]+2)], [zeros(p[1]+3)])
                end

        else # Nx>1
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
                return SparseRadialMapComponent(Nx, p, ξ, σ, a)
        end
end

SparseRadialMapComponent(Nx::Int64, p::Int64) = SparseRadialMapComponent(Nx, fill(p, Nx))



function show(io::IO, C::SparseRadialMapComponent)
if C.Nx==1
println(io,"$(C.Nx)-st component of a Sparse KR rearrangement of order p = $(C.p)")
else
println(io,"$(C.Nx)-th component of a Sparse KR rearrangement of order p = $(C.p)")
end
end

function set_id(C::SparseRadialMapComponent)
        Nx = C.Nx
if Nx==1
        C.p[1] = -1
        C.ξ[1] = Float64[]
        C.σ[1] = Float64[]
        C.a[1] = Float64[]
else
        C.p .= rmul!(ones(Nx), -1.0)

        @inbounds for i=1:Nx
        C.ξ[i] = Float64[]
        C.σ[i] = Float64[]
        C.a[i] = Float64[]
        end
end
end

# Transform vector of coefficients to a form SparseRadialMapComponent.a
function modify_a(A::Array{Float64,1}, C::SparseRadialMapComponent)
        @assert size(A,1)==size(extract_a(C),1) "A doesn't match the size of C"
        @get C (Nx, p)
                count = 0
        if Nx==1
                C.a[1] .= A
                count += size(A,1)
                # C.a[1] .= A
        else
                # count will store the last position of each assignment

                # Off-diagonal components
                @inbounds for idx=1:Nx-1
                pidx = p[idx]
                if pidx==-1
                        #No assignment
                elseif pidx==0
                        C.a[idx] .= [A[count+1]]
                        count += 1
                else
                        tmp_ai  = view(C.a, idx)[1]
                        tmp_ai .= view(A, count+1:count+(pidx+1))
                        count += deepcopy(pidx)+1
                end
                end

                # Diagonal components
                pidx = p[Nx]
                if pidx==-1
                        #No assignment, set to the identity without coefficient
                elseif pidx==0
                        # Affine function with two coefficients
                        tmp_ak  = view(C.a,Nx)[1]
                        tmp_ak .= view(A, count+1:count+2)

                        count +=2
                else
                        # Constant + p+2 ψ functions
                        tmp_ak  = view(C.a,Nx)[1]
                        tmp_ak .= view(A, count+1:count+pidx+3)
                        count += deepcopy(pidx)+3
                        # C.a[Nx] .= view(A, (Nx-1)*(p+1)+1:Nx*(p+1)+2)
                        # C.a[Nx] .= A[(Nx-1)*(p+1)+1:Nx*(p+1)+2]
                end
        end
        @assert count == size(A,1) "Error in the assignment of the coefficients"
end

extract_a(C::SparseRadialMapComponent) = vcat(C.a...)


size(C::SparseRadialMapComponent) = (C.Nx, C.p)

# Evaluate the map SparseRadialMapComponent at z = (x,...,x)
function (C::SparseRadialMapComponent)(z::T) where {T<:Real}
        @get C (Nx, p)

        out = uk(p[Nx], C.ξ[Nx], C.σ[Nx], C.a[Nx])(z)
        if Nx>1
                for i=1:Nx-1
                @inbounds out += ui(p[i], C.ξ[i], C.σ[i], C.a[i])(z)
                end
        end
        return out
end

# Evaluate the map SparseRadialMapComponent at z = (z1,...,zNx)
# Optimized with views
function (C::SparseRadialMapComponent)(z)
        @get C (Nx, p)
        @assert size(z,1)<=Nx  "The vector z has more components than RadialMapComponent"
        if p[Nx]<1
        out = uk(p[Nx], C.ξ[Nx], C.σ[Nx], C.a[Nx])(z[Nx])
        else
        out = uk(p[Nx], view(C.ξ,Nx)[1], view(C.σ,Nx)[1], view(C.a,Nx)[1])(z[Nx])
        end

        if Nx>1
                @inbounds for idx=1:Nx-1
                pidx = p[idx]

                if pidx==-1
                # ui(z) = 0.0 can skip this
                elseif pidx==0
                out += ui(pidx, C.ξ[idx], C.σ[idx], C.a[idx])(z[idx])
                else
                out += ui(pidx, view(C.ξ,idx)[1], view(C.σ,idx)[1], view(C.a,idx)[1])(z[idx])
                end
                end
        end
        return out
end


# Evaluate the map C at z = (z1,...,zk)
# Optimized with views
function off_diagonal(C::SparseRadialMapComponent, z)
        @get C (Nx, p)
        @assert size(z,1)<=Nx  "The vector z has more components than RadialMapComponent"
        out = 0.0
        if Nx>1
        @inbounds for idx=1:Nx-1
                        pidx = p[idx]
                        if pidx==-1
                                # No computation
                        elseif pidx==0
                        out += ui(pidx, C.ξ[idx], C.σ[idx], C.a[idx])(z[idx])
                        else
                        out += ui(pidx, view(C.ξ,idx)[1], view(C.σ,idx)[1], view(C.a,idx)[1])(z[idx])
                        end
                end
        end
        return out
end

function component(C::SparseRadialMapComponent, idx::Int64)
        @get C (Nx, p)
        @assert Nx>= idx "Can't access this component idx>Nx"
        if p[idx]<1
                if idx==Nx
                        return uk(p[idx], C.ξ[idx], C.σ[idx], C.a[idx])
                else
                        return ui(p[idx], C.ξ[idx], C.σ[idx], C.a[idx])
                end
        else
                if idx==Nx
                        return uk(p[idx], C.ξ[idx], C.σ[idx], C.a[idx])
                else
                        return ui(p[idx], C.ξ[idx], C.σ[idx], C.a[idx])
                end
        end
end
