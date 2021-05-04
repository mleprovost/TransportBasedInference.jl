export costfunction, LHD

# "Create tools to compute cost function"

# Type to hold cost functions
abstract type costfunction end

# Likelihood function
struct LHD <:costfunction
	A::Array{Float64,2}
	dψ::Array{Float64,2}
	b::Array{Float64,1}
	λ::Float64
	δ::Float64
	nb::Int64
	Ne::Int64

	# Cache variables
	dS::Array{Float64,1}
	logdS::Array{Float64,1}
	dψ_dS::Array{Float64,2}

	# Store gradient and hessian
	J::Array{Float64,1}
	G::Array{Float64,1}
	H::Array{Float64,2}
end

## Create constructor for likelihood function

function LHD(A::Array{Float64,2}, dψ::Array{Float64,2}, λ::Float64, δ::Float64)
	# Number of basis = p+2
	nb = size(A,1)
	@assert size(A,1)==size(A,2) "A must be a square matrix"

	Ne = size(dψ, 1)

	# Add L-2 regularization
	A .+= Matrix((λ/Ne)*I, nb, nb)

	b = zeros(nb)
	b .= sum(A, dims=2)[:,1]
	rmul!(b, δ)

	return LHD(A, dψ, b, λ, δ, nb, Ne, zeros(Ne), zeros(Ne), zeros(Ne,nb),zeros(1), zeros(nb), zeros(nb,nb))
end


function Base.show(io::IO, L::LHD)
	if L.λ==0.0 && L.δ==0.0
	println(io,"Cost function with Ne = $(L.Ne) samples")
	elseif L.λ==0.0 && L.δ!=0.0
	println(io,"Cost function with δ = $(L.δ) - regularized log barrier and Ne = $(L.Ne) samples")
	elseif L.λ!=0.0 && L.δ==0.0
	println(io,"Cost function with λ = $(L.λ) L-2 regularization and Ne = $(L.Ne) samples")
	else
	println(io,"Cost function with λ = $(L.λ) L-2 regularization,  δ = $(L.δ) - regularized log barrier and Ne = $(L.Ne) samples")
	end
end

# Function to compute the cost, gradient and loss
function (Lhd::LHD)(x::Array{Float64,1}, inplace::Bool; noutput::Int64=3)
	Lhd.dS .= Lhd.dψ*x
	Lhd.dS .+= Lhd.δ*sum(Lhd.dψ, dims = 2)[:,1]

	@inbounds for i=1:Lhd.Ne
		Lhd.logdS[i] = log(Lhd.dS[i])
	end

	if inplace
		# Compute gradient
		@inbounds for i=1:Lhd.Ne
			Lhd.dψ_dS[i,:] .= Lhd.dψ[i,:]/Lhd.dS[i]
		end

		# Return objective
		Lhd.J .= 0.5*dot(x, Lhd.A*x) - sum(Lhd.logdS,dims = 1)[1]*(1/Lhd.Ne) + dot(x,Lhd.b)

		# Return gradient
		Lhd.G .= Lhd.A*x - sum(Lhd.dψ_dS, dims = 1)[1,:]*(1/Lhd.Ne) + Lhd.b

		# Return hessian
		Lhd.H .= Lhd.A + BLAS.gemm('T', 'N', 1/Lhd.Ne, Lhd.dψ_dS, Lhd.dψ_dS)

	else
		if noutput == 1
			# Return objective
			J = 0.5*dot(x, Lhd.A*x) - sum(Lhd.logdS,dims = 1)[1]*(1/Lhd.Ne) + dot(x,Lhd.b)
		return J

		elseif noutput == 2
			# Compute gradient
			@inbounds for i=1:Lhd.Ne
				Lhd.dψ_dS[i,:] .= Lhd.dψ[i,:]/Lhd.dS[i]
			end

			# Return objective
			J = 0.5*dot(x, Lhd.A*x) - sum(Lhd.logdS,dims = 1)[1]*(1/Lhd.Ne) + dot(x,Lhd.b)

			# Return gradient
			G = Lhd.A*x - sum(Lhd.dψ_dS, dims = 1)[1,:]*(1/Lhd.Ne) + Lhd.b
			return J,G

		elseif noutput == 3
			# Compute gradient
			@inbounds for i=1:Lhd.Ne
				Lhd.dψ_dS[i,:] .= Lhd.dψ[i,:]/Lhd.dS[i]
			end

			# Return objective
			J = 0.5*dot(x, Lhd.A*x) - sum(Lhd.logdS,dims = 1)[1]*(1/Lhd.Ne) + dot(x,Lhd.b)

			# Return gradient
			G = Lhd.A*x - sum(Lhd.dψ_dS, dims = 1)[1,:]*(1/Lhd.Ne) + Lhd.b

			# Return hessian
			H = Lhd.A + BLAS.gemm('T', 'N', 1/Lhd.Ne, Lhd.dψ_dS, Lhd.dψ_dS)

			return J,G,H
		end
	end
end
