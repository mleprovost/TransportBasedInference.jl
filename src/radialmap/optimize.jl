export fast_mul, optimize, optimize_coeffs, solve_nonlinear


# Function to speed-up ψ_mono*Q1'*Q1
fast_mul(ψbis::Array{Float64,2}, Q, N::Int64, nx::Int64) = ([([ψbis zeros(size(ψbis,1), nx)] * Q)[axes(ψbis,1), 1:nx] zeros(size(ψbis,1), N)] * Q')[axes(ψbis,1), 1:N]



# Code to identify the coefficients
function optimize(C::RadialMapComponent, W::Weights, λ, δ)
	# C and W can have different dimension
	@assert C.p == W.p "Mismatch order p of the map"
	Nx = C.Nx
	@get W (p, Ne)
	# Compute weights
    ψ_off, ψ_mono, dψ_mono = rearrange(W,Nx)

    no = size(ψ_off,2)
	nd = size(ψ_mono,2)
    nx = size(ψ_off,2)+size(ψ_mono,2)+1
    nlog = size(dψ_mono,2)

	@assert nd==nlog "Error size of the diag and ∂Nx weights"
	@assert nx==no+nlog+1 "Error size of the weights"

	# Cache for the solution
	A = zeros(nd,nd)
	x = zeros(nx)

    # Normalize monotone basis functions
    μψ = copy(mean(ψ_mono, dims=1))[1,:]
    σψ = copy(std(ψ_mono, dims=1, corrected=false)[1,:])

    ψ_mono .-= μψ'
    ψ_mono ./= σψ'
    dψ_mono ./= σψ'

    if Nx==1
		BLAS.gemm!('T', 'N', 1/Ne, ψ_mono, ψ_mono, 1.0, A)
		# A .= BLAS.gemm('N', 'T', 1/Ne, ψ_mono, ψ_mono)
    else
        #Normalize off-diagonal covariates
        μψ_off = copy(mean(ψ_off, dims = 1)[1,:])
        σψ_off = copy(std(ψ_off, dims = 1, corrected = false)[1,:])
        ψ_off .-= μψ_off'
        ψ_off ./= σψ_off'

		#Assemble reduced QR to solve least square problem
		Asqrt = zero(ψ_mono)

		ψ_aug = zeros(Ne+no,no)
		ψ_X = view(ψ_aug,1:Ne,1:no)
		ψ_X .= ψ_off

		# ψ_λ = view(ψ_aug,Ne+1:Ne+no,1:no)
		@inbounds for i=1:no
			ψ_aug[Ne+i,i] = √λ
		end
	    F = qr(ψ_aug)
		# Speed-up proposed in https://discourse.julialang.org/t/extract-submatrix-from-qr-factor-is-slow-julia-1-4/36582/4
		Q1 = Matrix(F.Q)[1:Ne,:]
		Asqrt .= ψ_mono - Q1*Q1'*ψ_mono
		# Asqrt .= ψ_mono - fast_mul(ψ_mono, F.Q, Ne, no)

		BLAS.gemm!('T', 'N', 1/Ne, Asqrt, Asqrt, 1.0, A)
	end

    #Assemble reduced QR to solve least square problem
    #Approximate diagonal component
	# for linear S_{Nx}^{2} (i.e., order 0 function), use closed
	# form solution for linear monotone component
	if p == 0
        @assert size(A)==(1,1) "Quadratic matrix should be a scalar."
		g_mono = zeros(1)
		# This equation is equation (A.9) Couplings for nonlinear ensemble filtering
		# uNx(z) = c + α z so α = 1/√κ*
        g_mono[1] = sqrt(1/A[1,1])

	# for nonlinear diagonal, use Newton solver for coefficients
	else
		g_mono = ones(nd)
		# Construct loss function, the convention is flipped inside the solver
		lhd = LHD(A, dψ_mono, λ, δ)
		g_mono,_,_,_ = projected_newton(g_mono, lhd, "TrueHessian")
		g_mono .+= δ
	end

	if Nx==1
		g_mono ./= σψ
		x[no+1] = copy(-dot(μψ,g_mono))
	else
		g_off = zeros(no)
		ψ = zeros(Ne+no)
		view(ψ, 1:Ne) .= ψ_mono*g_mono

		g_off.= F.R\((F.Q'*ψ)[1:no])
		rmul!(g_off, -1.0)

		# Rescale coefficients
		g_mono ./= σψ

		g_off ./= σψ_off
		# Compute and store the constant term (expectation of selected terms)
		x[no+1] = -dot(μψ, g_mono)-dot(μψ_off, g_off)
	end

	# Coefficients of the diagonal except the constant
	x[no+2:nx] .= copy(g_mono)


	if Nx>1
		x[1:no] .= copy(g_off)
	end
	return x

end

function optimize(S::RadialMap, X; start::Int64=1, P::Parallel=serial)
	NxX, Ne = size(X)
	@get S (Nx, p, γ, λ, δ, κ)
	@assert NxX == Nx "Wrong dimension of the ensemble matrix X"
	# Compute centers and widths
	center_std!(S, X)

	# Create weights
	W = create_weights(S, X)

	# Compute weights
	compute_weights(S, X, W)

	# Optimize coefficients with multi-threading
	if typeof(P)==Serial
		@inbounds for i=start:Nx
				xopt = optimize(S.C[i], W, λ, δ)
		    	modify_a!(S.C[i], xopt)
		end
	else
		@inbounds Threads.@threads for i=start:Nx
				xopt = optimize(S.C[i], W, λ, δ)
				modify_a!(S.C[i], xopt)
		end
	end
end

## Optimize for the weights with SparseRadialMapComponentMap

# Code to identify the coefficients

function optimize(C::SparseRadialMapComponent, X, λ, δ)
	NxX, Ne = size(X)
	@get C (Nx,p)

	@assert NxX == Nx "Wrong dimension of the ensemble matrix X"

	# Compute weights
    ψ_off, ψ_mono, dψ_mono = compute_weights(C, X)

    no = size(ψ_off,2)
	nd = size(ψ_mono,2)
    nx = size(ψ_off,2)+size(ψ_mono,2)+1
    nlog = size(dψ_mono,2)

	@assert nd==nlog "Error size of the diag and ∂Nx weights"
	@assert nx==no+nlog+1 "Error size of the weights"

	# Cache for the solution
	A = zeros(nd,nd)
	x = zeros(nx)

    # Normalize monotone basis functions
	μψ = copy(mean(ψ_mono, dims=1))[1,:]
    σψ = copy(std(ψ_mono, dims=1, corrected=false))[1,:]

	ψ_mono .-= μψ'
    ψ_mono ./= σψ'
    dψ_mono ./= σψ'

	# @show ψ_mono[2,:]
	# @show dψ_mono[2,:]

    if Nx==1 || no == 0
		BLAS.gemm!('T', 'N', 1/Ne, ψ_mono, ψ_mono, 1.0, A)
		# A .= BLAS.gemm('N', 'T', 1/Ne, ψ_mono, ψ_mono)
    else
        #Normalize off-diagonal covariates
        μψ_off = copy(mean(ψ_off, dims = 1))[1,:]
        σψ_off = copy(std(ψ_off, dims = 1, corrected = false))[1,:]
        ψ_off .-= μψ_off'
        ψ_off ./= σψ_off'

		#Assemble reduced QR to solve least square problem
		Asqrt = zero(ψ_mono)

		ψ_aug = zeros(Ne+no,no)
		ψ_X = view(ψ_aug,1:Ne,1:no)
		ψ_X .= ψ_off

		@inbounds for i=1:no
			ψ_aug[Ne+i,i] = √λ
		end

		F = qr(ψ_aug)

		Q1 = Matrix(F.Q)[1:Ne,:]
		Asqrt .= ψ_mono - Q1*Q1'*ψ_mono
		# Asqrt .= ψ_mono - fast_mul(ψ_mono, F.Q, Ne, no)

		BLAS.gemm!('T', 'N', 1/Ne, Asqrt, Asqrt, 1.0, A)
	end

    #Assemble reduced QR to solve least square problem
    #Approximate diagonal component
	# for linear S_{Nx}^{2} (i.e., order 0 function), use closed
	# form solution for linear monotone component
	if p[Nx] == 0
        @assert size(A)==(1,1) "Quadratic matrix should be a scalar."
		g_mono = zeros(1)
		# This equation is equation (A.9) Couplings for nonlinear ensemble filtering
		# uNx(z) = c + α z so α = 1/√κ*
        g_mono[1] = sqrt(1/A[1,1])
	# for nonlinear diagonal, use Newton solver for coefficients
	else
		g_mono = ones(nd)
		# Construct loss function, the convention is flipped inside the solver
		lhd = LHD(A, dψ_mono, λ, δ)
		g_mono,_,_,_ = projected_newton(g_mono, lhd, "TrueHessian")
		g_mono .+= δ
	end

	if Nx == 1 || no == 0
		g_mono ./= σψ
		x[no+1] = copy(-dot(μψ,g_mono))
	else
		g_off = zeros(no)
		ψ = zeros(Ne+no)
		view(ψ, 1:Ne) .= ψ_mono*g_mono

		g_off.= F.R\((F.Q'*ψ)[1:no])
		rmul!(g_off, -1.0)

	    # Rescale coefficients
	    g_mono ./= σψ

	    g_off ./= σψ_off
	    # Compute and store the constant term (expectation of selected terms)
		x[no+1] = -dot(μψ, g_mono)-dot(μψ_off, g_off)
	end

	# Coefficients of the diagonal except the constant
	x[no+2:nx] .= copy(g_mono)


	if Nx>1 && no != 0
		x[1:no] .= copy(g_off)
	end
	return x

end

function optimize(S::SparseRadialMap, X; start::Int64=1, P::Parallel=serial)
	NxX, Ne = size(X)
	@get S (Nx, p, γ, λ, δ, κ)

	@assert NxX == Nx "Wrong dimension of the ensemble matrix X"

	# Compute centers and widths
	center_std!(S, X)

	# Optimize coefficients
	# Skip the identity components of the map
	if typeof(P)==Serial
		@inbounds for i=start:Nx
			if !allequal(p[i], -1)
					xopt = optimize(S.C[i], X[1:i,:], λ, δ)
					modify_a!(S.C[i], xopt)
			end
		end
	else
		@inbounds Threads.@threads for i=start:Nx
			if !allequal(p[i], -1)
					xopt = optimize(S.C[i], X[1:i,:], λ, δ)
					modify_a!(S.C[i], xopt)
			end
		end
	end
end


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
# function optimize_coeffs(C::RadialMapComponent{Nx,p}, W::Weights{n, p, Ne}) where {Nx,p,n,Ne}
#     # Compute weights
#     wq, w∂ = rearrange_weights(W, Nx)
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
#     modify_a!(C, xopt)
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




# This code is written for Nx>1
# function run_optimization_parallel(S::RadialMap, X; start::Int64=1)
# 	Nx, Ne = size(X)
# 	@get S (Nx, p, γ, λ, δ, κ)
# 	@assert Nx>1 "This code is not written for Nx=1"
# 	# Compute centers and widths
# 	center_std!(S, X)
# 	# Create weights
# 	W = create_weights(S, X)
# 	# Compute weights
# 	weights(S, X, W)
# 	# Optimize coefficients with Distributed
# 	# X = SharedArray{Float64}(Nx*(p+1)+2,Nx)
# 	# scoeffs = SharedArray{Int64}(Nx)
# 	X = SharedArray{Float64}(Nx*(p+1)+2,Nx-start+1)
# 	scoeffs = SharedArray{Int64}(Nx-start+1)
# 	@sync @distributed for i=start:Nx
# 		@inbounds begin
# 		xopt = optimize(S.C[i], W, λ, δ)
# 		scoeffs[i-start+1] = deepcopy(size(xopt,1))
# 	    X[1:size(xopt,1),i-start+1] .= deepcopy(xopt)
# 		end
# 	end
#
# 	# Run this part in serial
# 	@inbounds for i=start:Nx
# 		modify_a!(S.C[i], X[1:scoeffs[i-start+1],i-start+1])
# 	# @inbounds modify_a!(S.C[i], X[1:scoeffs[i],i])
#     end
# end
