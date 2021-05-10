export SparseRadialMapComponent, component, construct, clearcoeff!, evaluate, negative_likelihood, off_diagonal, set_id

#### Structure for the Nx-th component SparseRadialMapComponent of the lower triangular map U

# For now RadialMapComponent = Σi=1,k ui(k)

# The last array for ξ, σ and a will be of size p+2, p+2, p+3 versus p, p, p+1
# for the components i=1,...,Nx-1

struct SparseRadialMapComponent
        Nx::Int64
        p::Array{Int64,1}
        activedim::Array{Int64, 1}
        ξ::Array{Array{Float64,1}, 1}
        σ::Array{Array{Float64,1}, 1}
        coeff::Array{Array{Float64,1}, 1}

        # Cache for evaluation of function, gradient and hessian on basis

        # Inner constructor
        function SparseRadialMapComponent(Nx::Int64, p::Array{Int64,1}, activedim::Array{Int64,1}, ξ::Array{Array{Float64,1},1}, σ::Array{Array{Float64,1},1}, coeff::Array{Array{Float64,1},1})

        @assert size(activedim,1) <= Nx "Size of activedim should be smaller than the order of SparseRadialMapComponent"
        @assert size(ξ,1)==Nx "Size of ξ doesn't match the order of SparseRadialMapComponent"
        @assert size(σ,1)==Nx "Size of σ doesn't match the order of SparseRadialMapComponent"
        @assert size(coeff,1)==Nx "Size of coeff doesn't match the order of SparseRadialMapComponent"

        # activedim = filter(x-> x != -1, p)
                return new(Nx, p, activedim, ξ, σ, coeff)
        end
end

function SparseRadialMapComponent(Nx::Int64, p::Array{Int64,1})
        @assert Nx>=1 "Nx must be >0"
        activedim = Int64[]
        @inbounds for i=1:Nx
                if p[i] != -1
                        push!(activedim, copy(i))
                end
        end

        if Nx==1
                if p[1]==-1
                return SparseRadialMapComponent(Nx, p, activedim, [Float64[]], [Float64[]],  [Float64[]])
                end
                if p[1]==0
                return SparseRadialMapComponent(Nx, p, activedim, [Float64[]], [Float64[]],  [zeros(2)])
                end
                if p[1]>0
                return SparseRadialMapComponent(Nx, p, activedim, [zeros(p[1]+2)], [ones(p[1]+2)], [zeros(p[1]+3)])
                end

        else # Nx>1
                ξ = Array{Float64,1}[]
                σ = Array{Float64,1}[]
                coeff =  Array{Float64,1}[]
                # Fill Off Diagonal components
                for pj in p[1:end-1]
                if pj==-1
                        # Null component
                        push!(ξ, Float64[])
                        push!(σ, Float64[])
                        push!(coeff,Float64[])
                elseif pj==0
                        # Linear function
                        push!(ξ, Float64[])
                        push!(σ, Float64[])
                        push!(coeff,zeros(1))
                else    # Linear function + p rbf functions
                        push!(ξ, zeros(pj))
                        push!(σ, ones(pj))
                        push!(coeff,zeros(pj+1))
                end
                end

                # Fill Diagonal Component
                if p[end]==-1
                        # Null component
                        push!(ξ, Float64[])
                        push!(σ, Float64[])
                        push!(coeff,Float64[])
                elseif p[end]==0
                        # Affine function
                        push!(ξ, Float64[])
                        push!(σ, Float64[])
                        push!(coeff,zeros(2))
                else
                        #Constant + p+2 ψ functions
                        push!(ξ, zeros(p[end]+2))
                        push!(σ, ones(p[end]+2))
                        push!(coeff,zeros(p[end]+3))
                end
                return SparseRadialMapComponent(Nx, p, activedim, ξ, σ, coeff)
        end
end

SparseRadialMapComponent(Nx::Int64, p::Int64) = SparseRadialMapComponent(Nx, fill(p, Nx))

function show(io::IO, C::SparseRadialMapComponent)
    println(io,"Sparse radial map component of dimension "*string(C.Nx)*" and order p = $(C.p)")
end

function set_id(C::SparseRadialMapComponent)
        Nx = C.Nx
if Nx==1
        C.p[1] = -1
        C.ξ[1] = Float64[]
        C.σ[1] = Float64[]
        C.coeff[1] = Float64[]
else
        C.p .= rmul!(ones(Nx), -1.0)

        @inbounds for i=1:Nx
        C.ξ[i] = Float64[]
        C.σ[i] = Float64[]
        C.coeff[i] = Float64[]
        end
end
end

# Transform vector of coefficients to a form SparseRadialMapComponent.coeff
function modifycoeff!(C::SparseRadialMapComponent, A::Array{Float64,1})
        @assert size(A,1)==size(extractcoeff(C),1) "A doesn't match the size of C"
        @get C (Nx, p)
                count = 0
        if Nx==1
                C.coeff[1] .= A
                count += size(A,1)
                # C.coeff[1] .= A
        else
                # count will store the last position of each assignment

                # Off-diagonal components
                @inbounds for idx=1:Nx-1
                pidx = p[idx]
                if pidx==-1
                        #No assignment
                elseif pidx==0
                        C.coeff[idx] .= [A[count+1]]
                        count += 1
                else
                        tmp_coeffi  = view(C.coeff,idx)[1]
                        tmp_coeffi .= view(A, count+1:count+(pidx+1))
                        count += deepcopy(pidx)+1
                end
                end

                # Diagonal components
                pidx = p[Nx]
                if pidx==-1
                        #No assignment, set to the identity without coefficient
                elseif pidx==0
                        # Affine function with two coefficients
                        tmp_coeffk  = view(C.coeff,Nx)[1]
                        tmp_coeffk .= view(A, count+1:count+2)

                        count +=2
                else
                        # Constant + p+2 ψ functions
                        tmp_coeffk  = view(C.coeff,Nx)[1]
                        tmp_coeffk .= view(A, count+1:count+pidx+3)
                        count += deepcopy(pidx)+3
                        # C.coeff[Nx] .= view(A, (Nx-1)*(p+1)+1:Nx*(p+1)+2)
                        # C.coeff[Nx] .= A[(Nx-1)*(p+1)+1:Nx*(p+1)+2]
                end
        end
        @assert count == size(A,1) "Error in the assignment of the coefficients"
end

function clearcoeff!(C::SparseRadialMapComponent)
        @inbounds for i in C.activedim
                fill!(C.coeff[i], 0.0)
        end
end

extractcoeff(C::SparseRadialMapComponent) = vcat(C.coeff...)


size(C::SparseRadialMapComponent) = (C.Nx, C.p)

# Evaluate the map SparseRadialMapComponent at z = (x,...,x)
function (C::SparseRadialMapComponent)(z::T) where {T<:Real}
        @get C (Nx, p)

        out = uk(p[Nx], C.ξ[Nx], C.σ[Nx], C.coeff[Nx])(z)
        if Nx>1
                @inbounds for i in intersect(1:Nx-1, C.activedim)
                        out += ui(p[i], C.ξ[i], C.σ[i], C.coeff[i])(z)
                end
        end
        return out
end

# Evaluate the map SparseRadialMapComponent at z = (z1,...,zNx)
# Optimized with views
function (C::SparseRadialMapComponent)(z::AbstractVector{Float64})
        @get C (Nx, p)
        @assert size(z,1)<=Nx  "The vector z has more components than RadialMapComponent"
        if p[Nx]<1
                out = uk(p[Nx], C.ξ[Nx], C.σ[Nx], C.coeff[Nx])(z[Nx])
        else
                out = uk(p[Nx], view(C.ξ,Nx)[1], view(C.σ,Nx)[1], view(C.coeff,Nx)[1])(z[Nx])
        end

        if Nx>1
                @inbounds for idx in intersect(1:Nx-1, C.activedim)
                        pidx = p[idx]

                        if pidx==-1
                        # ui(z) = 0.0 can skip this
                        elseif pidx==0
                        out += ui(pidx, C.ξ[idx], C.σ[idx], C.coeff[idx])(z[idx])
                        else
                        out += ui(pidx, view(C.ξ,idx)[1], view(C.σ,idx)[1], view(C.coeff,idx)[1])(z[idx])
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
        @inbounds for idx in intersect(1:Nx-1, C.activedim)
                        pidx = p[idx]
                        if pidx==-1
                                # No computation
                        elseif pidx==0
                        out += ui(pidx, C.ξ[idx], C.σ[idx], C.coeff[idx])(z[idx])
                        else
                        out += ui(pidx, view(C.ξ,idx)[1], view(C.σ,idx)[1], view(C.coeff,idx)[1])(z[idx])
                        end
                end
        end
        return out
end

function D!(C::SparseRadialMapComponent, z::AbstractVector{Float64})
        @get C (Nx, p)
        @assert size(z,1) == Nx  "The vector z has more components than the SparseRadialMapComponent"
        return D!(uk(p[Nx], C.ξ[Nx], C.σ[Nx], C.coeff[Nx]), z[Nx])
end

D(C::SparseRadialMapComponent) =  z-> D!(C, z)

function negative_likelihood(C::SparseRadialMapComponent, X::AbstractMatrix{Float64})
        @get C (Nx, p)
        J = 0.0
        NxX, Ne = size(X)

        @assert NxX == Nx "Wrong dimension of the ensemble matrix `X`"
        ∂kC = D(C)
        @inbounds for i=1:Ne
                col = view(X,:,i)
                # Quadratic term
                J += 0.5*C(col)^2
                # Log barrier term
                J -= log(∂kC(col))
        end
        J *= 1/Ne
        return J
end

function component(C::SparseRadialMapComponent, idx::Int64)
        @get C (Nx, p)
        @assert Nx>= idx "Can't access this component idx>Nx"
        if p[idx]<1
                if idx==Nx
                        return uk(p[idx], C.ξ[idx], C.σ[idx], C.coeff[idx])
                else
                        return ui(p[idx], C.ξ[idx], C.σ[idx], C.coeff[idx])
                end
        else
                if idx==Nx
                        return uk(p[idx], C.ξ[idx], C.σ[idx], C.coeff[idx])
                else
                        return ui(p[idx], C.ξ[idx], C.σ[idx], C.coeff[idx])
                end
        end
end
