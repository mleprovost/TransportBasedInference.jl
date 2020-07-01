

function scale(ens::EnsembleState{Nx,Ne}; diag::Bool=true) where {Nx, Ne}

    x̄ = mean(ens)
    ens.S .-= x̄

    if diag ==true
        σ = std(state.S, dims = 2)[:,1]
        @inbounds for i=1:Nx
            ens.S[i,:] ./= σ[i]
        end
    else
        L = cholesky(cov(ens)).L
        ldiv!(ens.S, L, ens.S)
    end
    return ens
end
