export RadialMapComponent, component, construct, evaluate, modify_a!, off_diagonal

#### Structure for the k-th component RadialMapComponent of the lower triangular map U

# For now RadialMapComponent = Σi=1,k ui(k)

# The last array for ξ, σ and a will be of size p+2, p+2, p+3 versus p, p, p+1
# for the components i=1,...,k-1

struct RadialMapComponent
        Nx::Int64
        p::Int64
        ξ::Array{Array{Float64,1}, 1}#{Union{Array{Float64,1}, 1}
        σ::Array{Array{Float64,1}, 1}
        a::Array{Array{Float64,1}, 1}

        # Cache for evaluation of function, gradient and hessian on basis

        # Inner constructor
        function RadialMapComponent(Nx::Int64, p::Int64, ξ::Array{Array{Float64,1},1}, σ::Array{Array{Float64,1},1}, a::Array{Array{Float64,1},1})

        @assert size(ξ,1)==Nx "Size of ξ doesn't match the order of RadialMapComponent"
        @assert size(σ,1)==Nx "Size of σ doesn't match the order of RadialMapComponent"
        @assert size(a,1)==Nx "Size of a doesn't match the order of RadialMapComponent"
                return new(Nx, p, ξ, σ, a)
        end
end

function RadialMapComponent(Nx::Int64, p::Int64)
        @assert Nx>=1 "Nx must be >0"
        @assert p>=0 "p must be >=0"

        if Nx==1
                if p>0
                return RadialMapComponent(Nx, p, [zeros(p+2)], [ones(p+2)], [zeros(p+3)])
                end

                if p==0
                return RadialMapComponent(Nx, p, [Float64[]], [Float64[]],  [zeros(2)])
                end
        end
        if Nx>1 && p>0
                return RadialMapComponent(Nx, p, push!([zeros(p) for i=1:Nx-1], zeros(p+2)),
                                push!([ones(p) for i=1:Nx-1], ones(p+2)),
                                push!([zeros(p+1) for i=1:Nx-1], zeros(p+3)))
        end

        if Nx>1 && p==0
                return RadialMapComponent(Nx, p, [Float64[] for i=1:Nx],
                                [Float64[] for i=1:Nx],
                                push!([zeros(1) for i=1:Nx-1], zeros(2)))
        end
end

# Function to construct
function construct(Nx::Int64, p::Int64, ξ::Array{Array{Float64,1},1}, σ::Array{Array{Float64,1},1})

if Nx==1
        f = [construct_uk(p, ξ[1], σ[1])]
else
        f = [construct_ui(p, ξ[i], σ[i]) for i=1:Nx-1]
        push!(f, construct_uk(p, ξ[Nx], σ[Nx]))
end
return f
end


function Base.show(io::IO, C::RadialMapComponent)
        println(io,"Radial map component of dimension "*string(C.Nx)*" and order p = "*string(C.p))
end




# Transform vector of coefficients to a form RadialMapComponent.a
function modify_a!(C::RadialMapComponent, A::Array{Float64,1})
        @get C (Nx, p)
        if Nx==1
                C.a[1] .= A
                # C.a[1] .= A
        else
                if p==0
                        for idx=1:Nx-1
                        @inbounds C.a[idx] .= [A[idx]]
                        end
                        tmp_ak = view(C.a,Nx)[1]
                        tmp_ak .= view(A, Nx:Nx+1)
                        # C.a[k] .= A[end-1:end]
                        # C.a[k] .= A[end-1:end]
                else
                        @inbounds  for idx=1:Nx-1
                        tmp_ai = view(C.a, idx)[1]
                        tmp_ai .= view(A, (idx-1)*(p+1)+1:idx*(p+1))
                        # @inbounds C.a[idx] .= view(A, (idx-1)*(p+1)+1:idx*(p+1))
                        # @inbounds C.a[idx] .= A[(idx-1)*(p+1)+1:idx*(p+1)]
                        end
                        tmp_ak  = view(C.a,Nx)[1]
                        tmp_ak .= view(A, (Nx-1)*(p+1)+1:Nx*(p+1)+2)
                        # C.a[k] .= view(A, (k-1)*(p+1)+1:k*(p+1)+2)
                        # C.a[Nx] .= A[(Nx-1)*(p+1)+1:Nx*(p+1)+2]
                end
        end
        nothing
end

extract_a(C::RadialMapComponent) = vcat(C.a...)


function evaluate(C::RadialMapComponent, z::Array{T,1}) where {T<:Real}
        @get C (Nx, p)
        @assert Nx==size(z,1) "Wrong dimension of z"
        A = extract_a(C)

        n = size(A,1)
        basis = zeros(n)

        if p==0
                if Nx==1
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
                for i=1:Nx-1
                        compute_weights(component(C, i), z[i], wo)
                        basis[(i-1)*(p+1)+1:i*(p+1)] .= deepcopy(wo)
                end
                # Diagonal component
                compute_weights(component(C, Nx), z[Nx], wd)
                basis[(Nx-1)*(p+1)+1:Nx*(p+1)+2] .= deepcopy(wd)
        end
        return basis
end


size(C::RadialMapComponent) = (C.Nx, C.p)

# Evaluate the map RadialMapComponent at z = (x,...,x)
function (C::RadialMapComponent)(z::T) where {T<:Real}
        @get C (Nx, p)
        out = uk(p, C.ξ[Nx], C.σ[Nx], C.a[Nx])(z)
        if Nx>1
                for i=1:Nx-1
                @inbounds out += ui(p, C.ξ[i], C.σ[i], C.a[i])(z)
                end
        end
        return out
end

# Evaluate the map RadialMapComponent at z = (z1,...,zNx)
# Optimized with views
function (C::RadialMapComponent)(z)
        @get C (Nx, p)
        @assert size(z,1)<=Nx  "The vector z has more components than RadialMapComponent"
        if p==0
        out = uk(p, C.ξ[Nx], C.σ[Nx], C.a[Nx])(z[end])
        else
        out = uk(p, view(C.ξ,Nx)[1], view(C.σ,Nx)[1], view(C.a,Nx)[1])(z[end])
        end

        if Nx>1
                if p==0
                for i=1:Nx-1
                @inbounds out += ui(p, C.ξ[i], C.σ[i], C.a[i])(z[i])
                end
                else
                for i=1:Nx-1
                @inbounds out += ui(p, view(C.ξ,i)[1], view(C.σ,i)[1], view(C.a,i)[1])(z[i])
                end
                end
        end
        return out
end


# Evaluate the map RadialMapComponent at z = (z1,...,zNx)
# Optimized with views
function off_diagonal(C::RadialMapComponent, z)
        @get C (Nx, p)
        @assert size(z,1)<=Nx  "The vector z has more components than RadialMapComponent"
        out = 0.0
        if Nx>1
                if p==0
                for i=1:Nx-1
                @inbounds out += ui(p, C.ξ[i], C.σ[i], C.a[i])(z[i])
                end
                else
                for i=1:Nx-1
                @inbounds out += ui(p, view(C.ξ,i)[1], view(C.σ,i)[1], view(C.a,i)[1])(z[i])
                end
                end
        end
        return out
end

function component(C::RadialMapComponent, idx::Int64)
        @get C (Nx, p)
        @assert Nx>= idx "Can't access this component idx>Nx"
        if p==0
                if idx==Nx
                        return uk(p, C.ξ[idx], C.σ[idx], C.a[idx])
                else
                        return ui(p, C.ξ[idx], C.σ[idx], C.a[idx])
                end
        else
                if idx==Nx
                        return uk(p, view(C.ξ,idx)[1], view(C.σ,idx)[1], view(C.a,idx)[1])
                else
                        return ui(p, view(C.ξ,idx)[1], view(C.σ,idx)[1], view(C.a,idx)[1])
                end
        end
end




# Compute the last component uk of RadialMapComponent
# uk(C::RadialMapComponent{k,p}) where {k, p} = uk(k,p, C.ξ[k], C.σ[k], C.a[k])
#
#
# ∇k(C::RadialMapComponent{k,p}) where {k, p} = ∇k(uk(C))


# function construct(C::RadialMapComponent)
#         @get C (k, p)
#         f = separablefcn[]
#         if k>1
#                 for i=1:k-1
#                 @inbounds push!(f, deepcopy(ui(p, C.ξ[i], C.σ[i], C.a[i])))
#                 end
#         end
#         push!(f, deepcopy(uk(p, C.ξ[k], C.σ[k], C.a[k])))
#         return f
# end
