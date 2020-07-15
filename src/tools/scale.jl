export scale!, scale

function scale!(ens::EnsembleState{Nx,Ne}; diag::Bool=true) where {Nx, Ne}
    @assert Ne>1 "Need at leat two samples"
    x̄ = mean(ens)
    ens.S .-= x̄

    if diag ==true
        σ = std(ens.S, dims = 2)[:,1]
        @inbounds for i=1:Nx
            ens.S[i,:] ./= σ[i]
        end
    else
        L = cholesky(cov(ens)).L
        ldiv!(ens.S, L, ens.S)
    end
end


function scale!(result::EnsembleState{Nx, Ne}, ens::EnsembleState{Nx,Ne}; diag::Bool=true) where {Nx, Ne}
    @assert Ne>1 "Need at leat two samples"
    result.S .= deepcopy(ens.S)
    x̄ = mean(ens)
    result.S .-= x̄

    if diag ==true
        σ = std(ens.S, dims = 2)[:,1]
        @inbounds for i=1:Nx
            result.S[i,:] ./= σ[i]
        end
    else
        L = cholesky(cov(ens)).L
        ldiv!(result.S, L, ens.S)
    end
    return result
end


scale(ens::EnsembleState{Nx, Ne}; diag::Bool=true) where {Nx, Ne} = scale!(EnsembleState(Nx, Ne), ens; diag = diag)
