export fast_mul, optimize_ricardo, run_optimization, run_optimization_parallel,  optimize_coeffs,
				solve_nonlinear


# Function to speed-up ψ_mono*Q1'*Q1
fast_mul(ψbis::Array{Float64,2}, Q, N::Int64, nx::Int64) = ([([ψbis zeros(size(ψbis,1), nx)] * Q)[axes(ψbis,1), 1:nx] zeros(size(ψbis,1), N)] * Q')[axes(ψbis,1), 1:N]



# Code to identify the coefficients
function optimize_ricardo(Vk::Uk, W::Weights, λ, δ)
	# Vk and W can have different dimension
	@assert Vk.p == W.p "Mismatch order p of the map"
	k = Vk.k
	@get W (p, Ne)
	# Compute weights
    ψ_off, ψ_mono, dψ_mono = rearrange_ricardo(W,k)

    no = size(ψ_off,1)
	nd = size(ψ_mono,1)
    nx = size(ψ_off,1)+size(ψ_mono,1)+1
    nlog = size(dψ_mono,1)

	@assert nd==nlog "Error size of the diag and ∂k weights"
	@assert nx==no+nlog+1 "Error size of the weights"

	# Cache for the solution
	A = zeros(nd,nd)
	x = zeros(nx)

    # Normalize monotone basis functions
    μψ = deepcopy(mean(ψ_mono, dims=2))
    σψ = deepcopy(std(ψ_mono, dims=2, corrected=false))

    ψ_mono .-= μψ
    ψ_mono ./= σψ
    dψ_mono ./= σψ

    if k==1
		BLAS.gemm!('N', 'T', 1/Ne, ψ_mono, ψ_mono, 1.0, A)
		# A .= BLAS.gemm('N', 'T', 1/Ne, ψ_mono, ψ_mono)
    else

        #Normalize off-diagonal covariates
        μψ_off = deepcopy(mean(ψ_off, dims = 2))
        σψ_off = deepcopy(std(ψ_off, dims = 2, corrected = false))
        ψ_off .-= μψ_off
        ψ_off ./= σψ_off

		#Assemble reduced QR to solve least square problem
		Asqrt = zero(ψ_mono)

		ψ_aug = zeros(Ne+no,no)
		ψ_X = view(ψ_aug,1:Ne,1:no)
		ψ_X .= ψ_off'

		ψ_λ = view(ψ_aug,Ne+1:Ne+no,1:no)
		ψ_λ .= Matrix(√λ*I, no, no)

		# ψ_aug .= vcat(ψ_off', Matrix(√λ*I, no, no))

		# F = zeros(Ne+no,no)
		# F .= vcat(ψ_off', Matrix(√λ*I, no, no))
		# The slowest line of the code
		# @time qr!(F)
		# @show F
	    F = qr(ψ_aug)
		# ψ_aug = 0.0
		# @show size(F.R)

	    # @time Q1 .= Matrix(F.Q)[1:Ne,1:no]
		# @show size(F.Q)
		# @time Q1 = view(F.Q, 1:Ne, 1:no)
		# @time Q1 .= (F.Q*Matrix(I,Ne+no,Ne+no))[1:Ne, 1:no]
		# Speed-up proposed in https://discourse.julialang.org/t/extract-submatrix-from-qr-factor-is-slow-julia-1-4/36582/4
	    # @time Asqrt .= ψ_mono - (ψ_mono*Q1)*Q1'
		Asqrt .= ψ_mono - fast_mul(ψ_mono, F.Q, Ne, no)
		BLAS.gemm!('N', 'T', 1/Ne, Asqrt, Asqrt, 1.0, A)
		# Asqrt = 0.0
	end

    #Assemble reduced QR to solve least square problem
    #Approximate diagonal component
	# for linear S_{k}^{2} (i.e., order 0 function), use closed
	# form solution for linear monotone component
	if p == 0
        @assert size(A)==(1,1) "Quadratic matrix should be a scalar."
		g_mono = zeros(1,1)
		# This equation is equation (A.9) Couplings for nonlinear ensemble filtering
		# uk(z) = c + α z so α = 1/√κ*
        g_mono[1,1] = sqrt(1/A[1,1])

	# for nonlinear diagonal, use Newton solver for coefficients
	else
		g_mono = ones(nd)
		# Construct loss function, the convention is flipped inside the solver
		Lhd = LHD(A, Matrix(dψ_mono'), λ, δ)
		g_mono,_,_,_ = projected_newton(g_mono, Lhd, "TrueHessian")
		g_mono .+= δ
	end

	if p==0
		if k==1
			g_mono /= σψ[1,1]
			# Compute constant term
	        x[no+1] = deepcopy(-μψ*g_mono)[1,1]
		elseif k==2
			g_off = zeros(no)
			ψ = zeros(Ne+no)
			BLAS.gemm!('T', 'N', 1.0, ψ_mono, g_mono, 1.0, view(ψ,1:Ne))
		    # g_off = -F.R\((Q1'*ψ_mono'*g_mono)[:,1])
			g_off.= F.R\((F.Q'*ψ)[1:no])
			rmul!(g_off, -1.0)
			# F = 0.0
		    # g_off = -F.R\(Q1'*ψ_mono'*g_mono)
		    # Rescale coefficients
		    g_mono /= σψ
		    g_off /= σψ_off

		    # Compute and store the constant term (expectation of selected terms)
		    x[no+1] = -dot(μψ[:,1], g_mono[:,1]) - dot(μψ_off[:,1], g_off[:,1])
		else
			g_off = zeros(no)
			ψ = zeros(Ne+no)
			BLAS.gemm!('T', 'N', 1.0, ψ_mono, g_mono, 1.0, view(ψ,1:Ne))
		    # g_off = -F.R\((Q1'*ψ_mono'*g_mono)[:,1])
			g_off.= F.R\((F.Q'*ψ)[1:no])
			rmul!(g_off, -1.0)
			# F = 0.0
		    # g_off = -F.R\((Q1'*ψ_mono'*g_mono)[:,1])
		    # Rescale coefficients
		    g_mono ./= σψ[:,1]
		    g_off ./= σψ_off[:,1]

		    # Compute and store the constant term (expectation of selected terms)
			x[no+1] = -dot(μψ[:,1], g_mono[:,1])-dot(μψ_off[:,1], g_off[:,1])
		end
	else
		if k==1
			g_mono ./= σψ[:,1]
			# Compute constant term
	        x[no+1] = deepcopy(-dot(μψ,g_mono))
		elseif k==2
			g_off = zeros(no)
			ψ = zeros(Ne+no)
			BLAS.gemm!('T', 'N', 1.0, ψ_mono, g_mono, 1.0, view(ψ,1:Ne))
		    # g_off = -F.R\((Q1'*ψ_mono'*g_mono)[:,1])
			g_off.= F.R\((F.Q'*ψ)[1:no])
			rmul!(g_off, -1.0)
			# F = 0.0
		    # Rescale coefficients
		    g_mono ./= σψ[:,1]
		    g_off ./= σψ_off[:,1]
		    # Compute and store the constant term (expectation of selected terms)
			x[no+1] = -dot(μψ[:,1], g_mono[:,1])-dot(μψ_off[:,1], g_off[:,1])
		else
			g_off = zeros(no)
			ψ = zeros(Ne+no)
			BLAS.gemm!('T', 'N', 1.0, ψ_mono, g_mono, 1.0, view(ψ,1:Ne))

		    # @time g_off = -F.R\((Q1'*ψ_mono'*g_mono)[:,1])
			# This speciific form saves a lot of computational time
			# @time g_off.= F.R\((F.Q'*[ψ_mono'*g_mono; zeros(no)])[1:no])
			g_off.= F.R\((F.Q'*ψ)[1:no])
			rmul!(g_off, -1.0)
			# F = 0.0
		    # Rescale coefficients
		    g_mono ./= σψ[:,1]
		    g_off ./= σψ_off[:,1]
		    # Compute and store the constant term (expectation of selected terms)
			x[no+1] = -dot(μψ[:,1], g_mono[:,1])-dot(μψ_off[:,1], g_off[:,1])
		end
	end


	# Coefficients of the diagonal except the constant
	x[no+2:nx] .= deepcopy(g_mono[:,1])


	if k>1
		x[1:no] .= deepcopy(g_off[:,1])
	end
	return x

end

function run_optimization(S::KRmap, ens::EnsembleState{Nx, Ne}; start::Int64=1, P::Parallel=serial) where {Nx,Ne}
	@get S (k, p, γ, λ, δ, κ)
	# Compute centers and widths
	center_std(S, ens)

	# Create weights
	W = create_weights(S, ens)

	# Compute weights
	weights(S, ens, W)

	# Optimize coefficients with multi-threading
	if typeof(P)==Serial
		@inbounds for i=start:k
				xopt = optimize_ricardo(S.U[i], W, λ, δ)
		    	modify_a(xopt, S.U[i])
		end
	else
		@inbounds Threads.@threads for i=start:k
				xopt = optimize_ricardo(S.U[i], W, λ, δ)
				modify_a(xopt, S.U[i])
		end
	end
end


# This code is written for k>1
function run_optimization_parallel(S::KRmap, ens::EnsembleState{Nx, Ne}; start::Int64=1) where {Nx,Ne}
	@get S (k, p, γ, λ, δ, κ)
	@assert k>1 "This code is not written for k=1"
	# Compute centers and widths
	center_std(S, ens)
	# Create weights
	W = create_weights(S, ens)
	# Compute weights
	weights(S, ens, W)
	# Optimize coefficients with Distributed
	# X = SharedArray{Float64}(k*(p+1)+2,k)
	# scoeffs = SharedArray{Int64}(k)
	X = SharedArray{Float64}(k*(p+1)+2,k-start+1)
	scoeffs = SharedArray{Int64}(k-start+1)
	@sync @distributed for i=start:k
		@inbounds begin
		xopt = optimize_ricardo(S.U[i], W, λ, δ)
		scoeffs[i-start+1] = deepcopy(size(xopt,1))
	    X[1:size(xopt,1),i-start+1] .= deepcopy(xopt)
		end
	end

	# Run this part in serial
	@inbounds for i=start:k
		modify_a(X[1:scoeffs[i-start+1],i-start+1], S.U[i])
	# @inbounds modify_a(X[1:scoeffs[i],i], S.U[i])
    end
end


## Optimize for the weights with SparseUkMap

# Code to identify the coefficients
function optimize_ricardo(Vk::SparseUk, ens::EnsembleState{Nx, Ne}, λ, δ) where {Nx, Ne}
	@get Vk (k,p)

	# Compute weights
    ψ_off, ψ_mono, dψ_mono = weights(Vk, ens)

    no = size(ψ_off,1)
	nd = size(ψ_mono,1)
    nx = size(ψ_off,1)+size(ψ_mono,1)+1
    nlog = size(dψ_mono,1)

	@assert nd==nlog "Error size of the diag and ∂k weights"
	@assert nx==no+nlog+1 "Error size of the weights"

	# Cache for the solution
	A = zeros(nd,nd)
	x = zeros(nx)

    # Normalize monotone basis functions
    μψ = deepcopy(mean(ψ_mono, dims=2))
    σψ = deepcopy(std(ψ_mono, dims=2, corrected=false))

    ψ_mono .-= μψ
    ψ_mono ./= σψ
    dψ_mono ./= σψ

    if k==1
		BLAS.gemm!('N', 'T', 1/Ne, ψ_mono, ψ_mono, 1.0, A)
		# A .= BLAS.gemm('N', 'T', 1/Ne, ψ_mono, ψ_mono)
    else

        #Normalize off-diagonal covariates
        μψ_off = deepcopy(mean(ψ_off, dims = 2))
        σψ_off = deepcopy(std(ψ_off, dims = 2, corrected = false))
        ψ_off .-= μψ_off
        ψ_off ./= σψ_off

		#Assemble reduced QR to solve least square problem
		Asqrt = zero(ψ_mono)

		ψ_aug = zeros(Ne+no,no)
		ψ_X = view(ψ_aug,1:Ne,1:no)
		ψ_X .= ψ_off'

		ψ_λ = view(ψ_aug,Ne+1:Ne+no,1:no)
		ψ_λ .= Matrix(√λ*I, no, no)

		# ψ_aug .= vcat(ψ_off', Matrix(√λ*I, no, no))

		# F = zeros(Ne+no,no)
		# F .= vcat(ψ_off', Matrix(√λ*I, no, no))
		# The slowest line of the code
		# @time qr!(F)
		# @show F
	    F = qr(ψ_aug)
		# ψ_aug = 0.0
		# @show size(F.R)

	    # @time Q1 .= Matrix(F.Q)[1:Ne,1:no]
		# @show size(F.Q)
		# @time Q1 = view(F.Q, 1:Ne, 1:no)
		# @time Q1 .= (F.Q*Matrix(I,Ne+no,Ne+no))[1:Ne, 1:no]
		# Speed-up proposed in https://discourse.julialang.org/t/extract-submatrix-from-qr-factor-is-slow-julia-1-4/36582/4
	    # @time Asqrt .= ψ_mono - (ψ_mono*Q1)*Q1'
		Asqrt .= ψ_mono - fast_mul(ψ_mono, F.Q, Ne, no)
		BLAS.gemm!('N', 'T', 1/Ne, Asqrt, Asqrt, 1.0, A)
		# Asqrt = 0.0
	end

    #Assemble reduced QR to solve least square problem
    #Approximate diagonal component
	# for linear S_{k}^{2} (i.e., order 0 function), use closed
	# form solution for linear monotone component
	if p[k] == 0
        @assert size(A)==(1,1) "Quadratic matrix should be a scalar."
		g_mono = zeros(1,1)
		# This equation is equation (A.9) Couplings for nonlinear ensemble filtering
		# uk(z) = c + α z so α = 1/√κ*
        g_mono[1,1] = sqrt(1/A[1,1])
	# for nonlinear diagonal, use Newton solver for coefficients
	else
		g_mono = ones(nd)
		# Construct loss function, the convention is flipped inside the solver
		Lhd = LHD(A, Matrix(dψ_mono'), λ, δ)
		g_mono,_,_,_ = projected_newton(g_mono, Lhd, "TrueHessian")
		g_mono .+= δ
	end

	if p[k]==0
		if k==1
			g_mono /= σψ[1,1]
			# Compute constant term
	        x[no+1] = deepcopy(-μψ*g_mono)[1,1]
		else
			g_off = zeros(no)
			ψ = zeros(Ne+no)
			BLAS.gemm!('T', 'N', 1.0, ψ_mono, g_mono, 1.0, view(ψ,1:Ne))
		    # g_off = -F.R\((Q1'*ψ_mono'*g_mono)[:,1])
			g_off.= F.R\((F.Q'*ψ)[1:no])
			rmul!(g_off, -1.0)
			# F = 0.0
		    # g_off = -F.R\((Q1'*ψ_mono'*g_mono)[:,1])
		    # Rescale coefficients
		    g_mono ./= σψ[:,1]

		    g_off ./= σψ_off[:,1]
		    # Compute and store the constant term (expectation of selected terms)
			x[no+1] = -dot(μψ[:,1], g_mono[:,1])-dot(μψ_off[:,1], g_off[:,1])
		end
	else
		if k==1
			g_mono ./= σψ[:,1]
			# Compute constant term
	        x[no+1] = deepcopy(-dot(μψ,g_mono))
		else
			g_off = zeros(no)
			ψ = zeros(Ne+no)
			BLAS.gemm!('T', 'N', 1.0, ψ_mono, g_mono, 1.0, view(ψ,1:Ne))

		    # @time g_off = -F.R\((Q1'*ψ_mono'*g_mono)[:,1])
			# This speciific form saves a lot of computational time
			# @time g_off.= F.R\((F.Q'*[ψ_mono'*g_mono; zeros(no)])[1:no])
			g_off.= F.R\((F.Q'*ψ)[1:no])
			rmul!(g_off, -1.0)
			# F = 0.0
		    # Rescale coefficients
		    g_mono ./= σψ[:,1]
		    g_off ./= σψ_off[:,1]
		    # Compute and store the constant term (expectation of selected terms)
			x[no+1] = -dot(μψ[:,1], g_mono[:,1])-dot(μψ_off[:,1], g_off[:,1])
		end
	end


	# Coefficients of the diagonal except the constant
	x[no+2:nx] .= deepcopy(g_mono[:,1])


	if k>1
		x[1:no] .= deepcopy(g_off[:,1])
	end
	return x

end


function run_optimization(S::SparseKRmap, ens::EnsembleState{Nx, Ne}; start::Int64=1, P::Parallel=serial) where {Nx,Ne}
	@get S (k, p, γ, λ, δ, κ)
	# Compute centers and widths
	center_std(S, ens)

	# Optimize coefficients
	# Skip the identity components of the map
	if typeof(P)==Serial
		@inbounds for i=start:k
			if !allequal(p[i], -1)
					xopt = optimize_ricardo(S.U[i], EnsembleState(ens.S[1:i,:]), λ, δ)
					modify_a(xopt, S.U[i])
			end
		end
	else
		@inbounds Threads.@threads for i=start:k
			if !allequal(p[i], -1)
					xopt = optimize_ricardo(S.U[i], EnsembleState(ens.S[1:i,:]), λ, δ)
					modify_a(xopt, S.U[i])
			end
		end
	end
end








#
#
#
# function solve_nonlinear(dψ::Array{Float64,2}, A::Array{Float64,2}, λ, δ)
#
# 	n∂, Ne = size(dψ)
#
# 	# Create the model
# 	model = Model(Ipopt.Optimizer)
#
# 	# Create variable x
# 	@variable(model, x[1:n∂])#, start = 0.0)
#
# 	@NLparameter(model, b[i = 1:n∂] == δ*sum(A[i,j] for j=1:n∂))
#
# 	@NLparameter(model, Ã[i = 1:n∂, j = 1:n∂] == A[i,j]+(λ/Ne)*I[i,j])
#
# 	@NLparameter(model, dψopt[i = 1:n∂, j = 1:Ne] == dψ[i,j] + δ*sum(dψ[i,j] for j=1:Ne))
#
#
# 	# Ã is a symmetric matrix
# 	# Set lower bound on x
# 	@inbounds for i in 1:n∂
# 	    set_lower_bound(x[i], 0.0)
# 	end
#
# 	@NLobjective(model, Min, 0.5*sum(sum(x[i]*Ã[i,j]*x[j] for i=1:n∂) for j=1:n∂)-
# 	                  sum(log(sum(dψopt[i,j]*x[i] for i=1:n∂)) for j=1:Ne)*(1/Ne)+
# 	                  sum(x[i]*b[i] for i=1:n∂))
#
# 	optimize!(model)
#
# 	return value.(x)
# end
#
#
#
#
# # Code to identify the coefficients
# function optimize_coeffs(Vk::Uk{k,p}, W::Weights{n, p, Ne}) where {k,p,n,Ne}
#     # Compute weights
#     wq, w∂ = rearrange_weights(W, k)
#
#     nx = size(wq,1)
#     nlog = size(w∂,1)
#
#
#     # Create the model
#     model = Model(Ipopt.Optimizer)
#
#     # Create variable x
#     @variable(model, x[1:nx])#, start = 0.001)
#
#     @NLparameter(model, z[i = 1:nx, j = 1:Ne] == wq[i,j])
#     @NLparameter(model, b[i = 1:nlog, j = 1:Ne] == w∂[i,j])
#
#
#     # Set lower bound on the coefficients of the derivatives
#     @inbounds for i in nx-nlog+1:nx
#         set_lower_bound(x[i], 0.0)
#     end
#
#     if p==0
#     @NLobjective(model, Min, (1/Ne)*sum(0.5*sum(z[j,i]*x[j] for j=1:nx)^2 -(w∂[1,i]*x[end]) for i=1:Ne))
#     else
#     @NLobjective(model, Min, (1/Ne)*sum(0.5*sum(z[j,i]*x[j] for j=1:nx)^2 -
#                 log(sum(w∂[j,i]*x[end-(p+2)+j] for j=1:p+2)) for i=1:Ne))
#     end
#     optimize!(model)
#
#     # Now replace identifed coefficients into a
#     modify_a(value.(x), Vk)
# end
#
#
#
#
#




# function _cost(x, wquad, w∂,p)
#     J = 0.0
#     nx = length(x)
#     Ne = size(wquad,2)
#     @inbounds for i=1:Ne
#         J += 0.5*sum(wquad[j,i]*x[j] for j=1:nx)^2
#
#         if p==0
#         J += -log(w∂[1,i]*x[end])
#         else
#         J += -log(sum(w∂[j,i]*x[end-(p+2)+j] for j=1:p+2))
#         end
#     end
#     J *=  (1/Ne)
#     return J
# end
