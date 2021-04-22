export Weights, create_weights, weights, component,
        extract_a, modify_a, rearrange_weights,
        rearrange_ricardo, ncoeff

import Base: +, *, size

struct Weights
        k::Int64
        p::Union{Int64, Array{Int64,1}}
        Ne::Int64
        woff::Array{Float64,2}
        wdiag::Array{Float64,2}
        w∂k::Array{Float64,2}
end

function Base.show(io::IO, W::Weights)
  println(io,"Weights for a transport map with k = $(W.k), p = $(W.p) and Ne = $(W.Ne)")
end


function create_weights(T::KRmap)
        k = T.k
        p = T.p
        if p==0
                woff  = zeros(k-1)
                wdiag = zeros(2*k)
                w∂k   = zeros(k)
        else
                woff  = zeros((k-1)*(p+1))
                wdiag = zeros(k*(p+3))
                w∂k   = zeros(k*(p+2))
        end
        return woff, wdiag, w∂k
end

function create_weights(T::KRmap, X::AbstractMatrix{Float64})
        Nx, Ne = size(X)
        k = T.k
        p = T.p
        if p==0
                woff  = zeros(k-1, Ne)
                wdiag = zeros(2*k, Ne)
                w∂k   = zeros(k, Ne)
        else
                woff  = zeros((k-1)*(p+1), Ne)
                wdiag = zeros(k*(p+3), Ne)
                w∂k   = zeros(k*(p+2), Ne)
        end
        return Weights(k, p, Ne, woff, wdiag, w∂k)
end

function create_weights(T::KRmap, X::EnsembleState)
        Nx, Ne = size(X)
        k = T.k
        p = T.p
        if p==0
                woff  = zeros(k-1, Ne)
                wdiag = zeros(2*k, Ne)
                w∂k   = zeros(k, Ne)
        else
                woff  = zeros((k-1)*(p+1), Ne)
                wdiag = zeros(k*(p+3), Ne)
                w∂k   = zeros(k*(p+2), Ne)
        end
        return Weights(k, p, Ne, woff, wdiag, w∂k)
end

function weights(u::ui, z::Float64, woff)
        p = u.p
        # Check that woff has the right size
        if p==-1
                #No weights
        elseif p==0
                @assert size(woff, 1) == 1
                woff[1] = z
        else
                @assert size(woff, 1) == p+1

                woff[1] = z
                for j=1:p
                woff[j+1] = rbf(u.ξi[j], u.σi[j])(z)
                end
        end
        # woff
end


function weights(u::uk, z::Float64, wdiag, w∂k; withconstant::Bool = true)
        p = u.p
        # Check that wdiag and w∂k have the right size
        if withconstant == true
        if p==-1
                # No weights
        elseif p==0
                @assert size(wdiag, 1) == 2
                @assert size(w∂k, 1) == 1

                wdiag[1:2] .= [1.0; z]
                w∂k[1] = 1.0
        else
                @assert size(wdiag, 1) == p+3
                @assert size(w∂k, 1) == p+2

                wdiag[1] = 1.0
                # ψ₀ and ψ₀′
                wdiag[2] = ψ₀(u.ξk[1], u.σk[1])(z)
                w∂k[1] = ψ₀′(u.ξk[1], u.σk[1])(z)

                # ψₚ₊₁ and ψₚ₊₁′
                wdiag[p+3] = ψpp1(u.ξk[p+2], u.σk[p+2])(z)
                w∂k[p+2] = ψpp1′(u.ξk[p+2], u.σk[p+2])(z)

                # ψⱼ and ψⱼ′
                @inbounds for j=2:p+1
                wdiag[j+1] = ψj(u.ξk[j], u.σk[j])(z)
                w∂k[j] = rbf(u.ξk[j], u.σk[j])(z)
                end
        end
        else
        if p==-1
                # No weights
        elseif p==0
                @assert size(wdiag, 1) == 1
                @assert size(w∂k, 1) == 1

                wdiag[1] = z
                w∂k[1] = 1.0
        else
                @assert size(wdiag, 1) == p+2
                @assert size(w∂k, 1) == p+2

                # ψ₀ and ψ₀′
                wdiag[1] = ψ₀(u.ξk[1], u.σk[1])(z)
                w∂k[1] = ψ₀′(u.ξk[1], u.σk[1])(z)

                # ψₚ₊₁ and ψₚ₊₁′
                wdiag[p+2] = ψpp1(u.ξk[p+2], u.σk[p+2])(z)
                w∂k[p+2] = ψpp1′(u.ξk[p+2], u.σk[p+2])(z)

                # ψⱼ and ψⱼ′
                @inbounds for j=2:p+1
                wdiag[j] = ψj(u.ξk[j], u.σk[j])(z)
                w∂k[j] = rbf(u.ξk[j], u.σk[j])(z)
                end
        end
        end

        # wdiag, w∂k
end

function weights(u::uk, z::Float64, wdiag; withconstant::Bool = true)
        p = u.p
        # Check that wdiag and w∂k have the right size
        if withconstant ==true
        if p==-1
                # No weights
        elseif p==0
                @assert size(wdiag, 1) == 2
                wdiag[1:2] .= [1.0; z]
        else
                @assert size(wdiag, 1) == p+3
                wdiag[1] = 1.0
                # ψ₀ and ψ₀′
                wdiag[2] = ψ₀(u.ξk[1], u.σk[1])(z)

                # ψₚ₊₁ and ψₚ₊₁′
                wdiag[p+3] = ψpp1(u.ξk[p+2], u.σk[p+2])(z)

                # ψⱼ and ψⱼ′
                @inbounds for j=2:p+1
                wdiag[j+1] = ψj(u.ξk[j], u.σk[j])(z)
                end
        end
        else
                if p==-1
                        # No weights
                elseif p==0
                        @assert size(wdiag, 1) == 1
                        wdiag[1] = z
                else
                        @assert size(wdiag, 1) == p+2
                        # ψ₀ and ψ₀′
                        wdiag[1] = ψ₀(u.ξk[1], u.σk[1])(z)

                        # ψₚ₊₁ and ψₚ₊₁′
                        wdiag[p+2] = ψpp1(u.ξk[p+2], u.σk[p+2])(z)

                        # ψⱼ and ψⱼ′
                        @inbounds for j=2:p+1
                        wdiag[j] = ψj(u.ξk[j], u.σk[j])(z)
                        end
                end
        end
end

function weights(T::KRmap, z::Array{Float64,1}, woff::Array{Float64,1}, wdiag::Array{Float64,1}, w∂k::Array{Float64,1})
        @get T (k, p)
        if p==0
                if k==1
                wd = view(wdiag,:)
                w∂ = view(w∂k,k:k)
                weights(component(T.U[1],1), z[1], wd, w∂)
                else
                # Fill wdiag and w∂k
                @inbounds for i=1:k
                        wd = view(wdiag,(i-1)*2+1:i*2)
                        w∂ = view(w∂k,i:i)
                        weights(component(T.U[i],i), z[i], wd, w∂)
                end
                # Fill woff
                @inbounds for i=1:k-1
                        wo = view(woff,i:i)
                        weights(component(T.U[end],i), z[i], wo)
                end
                end
        else
                if k==1
                wd = view(wdiag,:)
                w∂ = view(w∂k,:)
                weights(component(T.U[1],1), z[1], wd, w∂)
                else
                        # Fill wdiag and w∂k
                        @inbounds for i=1:k
                                wd = view(wdiag, (i-1)*(p+3)+1:i*(p+3))
                                w∂ = view(w∂k, (i-1)*(p+2)+1:i*(p+2))
                                weights(component(T.U[i],i), z[i], wd, w∂)
                        end
                        # Fill woff
                        @inbounds for i=1:k-1
                                wo = view(woff, (i-1)*(p+1)+1:i*(p+1))
                                weights(component(T.U[end],i), z[i], wo)
                        end
                end
        end
end


function weights(T::KRmap, X::AbstractMatrix{Float64}, woff::Array{Float64,2}, wdiag::Array{Float64,2}, w∂k::Array{Float64,2}) where {k, Ne}
@assert T.k == k "Error in dimension of the ensemble and size of the KR map"
p = T.p
if p==0
        if k==1
        utmp = component(T.U[1],1)
        @inbounds for l=1:Ne
                wd = view(wdiag,:,l)
                w∂ = view(w∂k,1:1,l)
                weights(utmp, ens.S[1,l], wd, w∂)
        end
        else
        # Fill wdiag and w∂k
        @inbounds for i=1:k
        utmp = component(T.U[i],i)
                for l=1:Ne
                wd = view(wdiag, (i-1)*2+1:i*2, l)
                w∂ = view(w∂k,i:i,l)
                weights(utmp, ens.S[i,l], wd, w∂)
                end
        end
        # Fill woff
        @inbounds  for i=1:k-1
                utmp = component(T.U[end],i)
                for l=1:Ne
                wo = view(woff,i:i,l)
                weights(utmp, ens.S[i,l], wo)
                end
        end
        end
else
        if k==1
        utmp = component(T.U[1],1)
        @inbounds for l=1:Ne
                wd = view(wdiag,:,l)
                w∂ = view(w∂k,:,l)
                weights(utmp, ens.S[1,l], wd, w∂)
        end
        else
                # Fill wdiag and w∂k
                @inbounds for i=1:k
                        utmp = component(T.U[i],i)
                        for l=1:Ne
                        wd = view(wdiag, (i-1)*(p+3)+1:i*(p+3), l)
                        w∂ = view(w∂k, (i-1)*(p+2)+1:i*(p+2), l)
                        weights(utmp, ens.S[i,l], wd, w∂)
                        end
                end
                # Fill woff
                @inbounds for i=1:k-1
                        utmp = component(T.U[end],i)
                        for l=1:Ne
                        wo = view(woff, (i-1)*(p+1)+1:i*(p+1),l)
                        weights(utmp, ens.S[i,l], wo)
                        end
                end
        end
end
end

function weights(T::KRmap, ens::EnsembleState{k,Ne}, woff::Array{Float64,2}, wdiag::Array{Float64,2}, w∂k::Array{Float64,2}) where {k, Ne}
@assert T.k == k "Error in dimension of the ensemble and size of the KR map"
p = T.p
if p==0
        if k==1
        utmp = component(T.U[1],1)
        @inbounds for l=1:Ne
                wd = view(wdiag,:,l)
                w∂ = view(w∂k,1:1,l)
                weights(utmp, ens.S[1,l], wd, w∂)
        end
        else
        # Fill wdiag and w∂k
        @inbounds for i=1:k
        utmp = component(T.U[i],i)
                for l=1:Ne
                wd = view(wdiag, (i-1)*2+1:i*2, l)
                w∂ = view(w∂k,i:i,l)
                weights(utmp, ens.S[i,l], wd, w∂)
                end
        end
        # Fill woff
        @inbounds  for i=1:k-1
                utmp = component(T.U[end],i)
                for l=1:Ne
                wo = view(woff,i:i,l)
                weights(utmp, ens.S[i,l], wo)
                end
        end
        end
else
        if k==1
        utmp = component(T.U[1],1)
        @inbounds for l=1:Ne
                wd = view(wdiag,:,l)
                w∂ = view(w∂k,:,l)
                weights(utmp, ens.S[1,l], wd, w∂)
        end
        else
                # Fill wdiag and w∂k
                @inbounds for i=1:k
                        utmp = component(T.U[i],i)
                        for l=1:Ne
                        wd = view(wdiag, (i-1)*(p+3)+1:i*(p+3), l)
                        w∂ = view(w∂k, (i-1)*(p+2)+1:i*(p+2), l)
                        weights(utmp, ens.S[i,l], wd, w∂)
                        end
                end
                # Fill woff
                @inbounds for i=1:k-1
                        utmp = component(T.U[end],i)
                        for l=1:Ne
                        wo = view(woff, (i-1)*(p+1)+1:i*(p+1),l)
                        weights(utmp, ens.S[i,l], wo)
                        end
                end
        end
end
end

function weights(T::KRmap, ens::EnsembleState{k, Ne}, W::Weights) where {k, Ne}
        @assert T.p == W.p "Error value of p, can't return weights"
        @assert T.k == W.k "Error value of k, can't return weights"
        @assert T.k == k "Error value of k, can't return weights"

        return weights(T, ens, W.woff, W.wdiag, W.w∂k)
end

# Extract the weights on
function rearrange_weights(W::Weights, l::Int64)
        @get W (k, p, Ne, woff, wdiag)
        if p==0
                if l==1
                        wquad = wdiag[1:2,:]
                else
                        wquad = vcat(woff[1:l-1,:], wdiag[2*(l-1)+1:2*l,:])
                end
        else
                if l==1
                        wquad = wdiag[1:p+3,:]
                else
                        wquad = vcat(woff[1:(l-1)*(p+1),:], wdiag[(l-1)*(p+3)+1:l*(p+3),:])
                end
        end
        return  wquad#, w∂
end

# This function extracts the right weights for the cost function, wquad will be the part of the weights
# associated with the quadratic function, while w∂ will be associated with the log barrier function
# rearrange_weights does not use the line of 1 in wdiag
function rearrange_ricardo(W::Weights, l::Int64)
        @get W (k, p, Ne, woff, wdiag, w∂k)
        if p==0
                if l==1
                        wo = Float64[]
                        wd = wdiag[2:2,:]
                        w∂ = w∂k[1:1,:]
                else
                        wo = woff[1:l-1,:]
                        wd = wdiag[2*l:2*l,:]
                        w∂ = w∂k[l:l,:]
                end
        else
                if l==1
                        wo = Float64[]
                        wd = wdiag[2:p+3,:]
                        w∂ = w∂k[1:p+2,:]
                else
                        wo = woff[1:(l-1)*(p+1),:]
                        wd = wdiag[(l-1)*(p+3)+2:l*(p+3),:]
                        w∂ = w∂k[(l-1)*(p+2)+1:l*(p+2),:]
                end
        end
        return  wo, wd, w∂
end


## Procedure for a sparse map, weights cannot be reused from one line to the other,
#  need to treat each k-th component individually


function ncoeff(k::Int64, p::Array{Int64})
        noff = 0
        ndiag = 0
        n∂k = 0
        for i=1:k-1
                if p[i]==-1
                elseif p[i]==0
                        noff += 1
                else
                        noff += deepcopy(p[i])+1
                end
        end

        if p[k]==-1
        elseif p[k]==0
                ndiag = 1
                n∂k = 1
        else
                ndiag = p[k]+2
                n∂k = p[k]+2
        end

        return noff, ndiag, n∂k
end



# This function can be called directly into optimize_ricardo to get the right weights

function weights(Vk::SparseUk, z::Array{Float64,1})
        @get Vk (k, p)
        # Determine the number of coefficients for each ( ncoeff)
        noff, ndiag, n∂k = ncoeff(k, p)

        woff  = zeros(noff)
        # The constant of the diagonal term is omitted
        wdiag = zeros(ndiag)
        w∂k   = zeros(n∂k)
        count = 0
        # Fill off-diagonal components
        @inbounds for i=1:k-1
                pidx = p[i]
                if pidx == -1
                        # No weights to compute
                elseif pidx == 0
                        # Linear function
                        wo = view(woff,count+1)
                        weights(component(Vk,i), z[i], wo)
                        count +=1
                else
                        # Linear term + pidx rbf functions
                        wo = view(woff,count+1:count+pidx+1)
                        weights(component(Vk,i), z[i], wo)
                        count += deepcopy(pidx) + 1
                end
        end
        @assert count == noff "The size is correct for the off-diagonal weights"

        # Fill diagonal components
        if p[k] ==-1
                # No weights to compute
        else
                # Affine function for p[k]=0
                # Constant plus (p[k]+2) ψ functions for p[k]>0
                wd = view(wdiag,:)
                w∂ = view(w∂k,:)
                weights(component(Vk,k), z[k], wd, w∂,  withconstant = false)
        end
        return woff, wdiag, w∂k
end

function weights(Vk::SparseUk, ens::EnsembleState{Nx, Ne}) where {Nx, Ne}
        @get Vk (k, p)
        # Determine the number of coefficients for each ( ncoeff)
        noff, ndiag, n∂k = ncoeff(k, p)
        if noff ==0
        woff  = Float64[]
        else
        woff  = zeros(noff, Ne)
        end
        # The constant of the diagonal term is omitted
        if ndiag ==0
        wdiag = Float64[]
        else
        wdiag = zeros(ndiag, Ne)
        end
        if n∂k==0
        w∂k = Float64[]
        else
        w∂k   = zeros(n∂k, Ne)
        end

        count = 0
        # Fill off-diagonal components
        @inbounds for i=1:k-1
                pidx = p[i]
                if pidx == -1
                        # No weights to compute
                elseif pidx == 0
                        # Linear function
                        for l=1:Ne
                        wo = view(woff,count+1,l:l)
                        weights(component(Vk,i), ens.S[i,l], wo)
                        end
                        count +=1
                else
                        for l=1:Ne
                        # Linear term + pidx rbf functions
                        wo = view(woff,count+1:count+pidx+1,l:l)
                        weights(component(Vk,i), ens.S[i,l], wo)
                        end
                        count += deepcopy(pidx) + 1
                end
        end
        @assert count == noff "The size is correct for the off-diagonal weights"

        # Fill diagonal components
        if p[k] ==-1
                # No weights to compute
        else
                for l=1:Ne
                # Affine function for p[k]=0
                # Constant plus (p[k]+2) ψ functions for p[k]>0
                wd = view(wdiag,:,l:l)
                w∂ = view(w∂k,:,l:l)
                weights(component(Vk,k), ens.S[k,l], wd, w∂, withconstant = false)
                end
        end
        return woff, wdiag, w∂k
end

function weights(Vk::SparseUk, X)
        Nx, Ne = size(X)
        @get Vk (k, p)
        # Determine the number of coefficients for each ( ncoeff)
        noff, ndiag, n∂k = ncoeff(k, p)
        if noff ==0
        woff  = Float64[]
        else
        woff  = zeros(noff, Ne)
        end
        # The constant of the diagonal term is omitted
        if ndiag ==0
        wdiag = Float64[]
        else
        wdiag = zeros(ndiag, Ne)
        end
        if n∂k==0
        w∂k = Float64[]
        else
        w∂k   = zeros(n∂k, Ne)
        end

        count = 0
        # Fill off-diagonal components
        @inbounds for i=1:k-1
                pidx = p[i]
                if pidx == -1
                        # No weights to compute
                elseif pidx == 0
                        # Linear function
                        for l=1:Ne
                        wo = view(woff,count+1,l:l)
                        weights(component(Vk,i), ens.S[i,l], wo)
                        end
                        count +=1
                else
                        for l=1:Ne
                        # Linear term + pidx rbf functions
                        wo = view(woff,count+1:count+pidx+1,l:l)
                        weights(component(Vk,i), ens.S[i,l], wo)
                        end
                        count += deepcopy(pidx) + 1
                end
        end
        @assert count == noff "The size is correct for the off-diagonal weights"

        # Fill diagonal components
        if p[k] ==-1
                # No weights to compute
        else
                for l=1:Ne
                # Affine function for p[k]=0
                # Constant plus (p[k]+2) ψ functions for p[k]>0
                wd = view(wdiag,:,l:l)
                w∂ = view(w∂k,:,l:l)
                weights(component(Vk,k), ens.S[k,l], wd, w∂, withconstant = false)
                end
        end
        return woff, wdiag, w∂k
end
