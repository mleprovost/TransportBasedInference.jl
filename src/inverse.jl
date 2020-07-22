export functionalf!, functionalg!, functionalfg!

function functionalf!(F, xk, cacheF, ψoff, output::Array{Float64,1}, R::IntegratedFunction{m, Nψ, Nx}) where {m, Nψ, Nx}
    function integrand!(v::Vector{Float64}, t::Float64)
        # repeated_grad_xk_basis(cacheF, R.f.f,  t*xk)
        cacheF .= repeated_grad_xk_basis(R.f.f,  t*xk)
        @avx @. v = (cacheF .* ψoff) *ˡ R.f.f.coeff
        evaluate!(v, R.g, v)
        # v .= R.g((repeated_grad_xk_basis(R.f.f,  t*xk) .* ψoff)*R.f.f.coeff)
    end

    quadgk!(integrand!, F, 0, 1; rtol = 1e-3)
    F .*= xk
    F .-= output
    nothing
end

functionalf!(cacheF, ψoff, output::Array{Float64,1}, R::IntegratedFunction{m, Nψ, Nx}) where {m, Nψ, Nx} =
    (F, xk) -> functionalf!(F, xk, cacheF, ψoff, output, R)

function functionalg!(J, xk, cache, cacheJ, ψoff, output::Array{Float64,1}, R::IntegratedFunction{m, Nψ, Nx}) where {m, Nψ, Nx}
    cache .= repeated_grad_xk_basis(R.f.f,  xk)
    # epeated_grad_xk_basis!(cache, R.f.f,  xk)

    @avx @. cacheJ = (cache .* ψoff) *ˡ R.f.f.coeff
    evaluate!(cacheJ, R.g, cacheJ)
    J .= Diagonal(cacheJ)
    nothing
end


functionalg!(cache, cacheJ, ψoff, output::Array{Float64,1}, R::IntegratedFunction{m, Nψ, Nx}) where {m, Nψ, Nx} =
    (J, xk) -> functionalg!(J, xk, cache, cacheJ, ψoff, output, R)


function functionalfg!(F, J, xk, cache, ψoff, output::Array{Float64,1}, R::IntegratedFunction{m, Nψ, Nx}) where {m, Nψ, Nx}

    if !(F == nothing)
        function integrand!(v::Vector{Float64}, t::Float64)
            v .= R.g((repeated_grad_xk_basis(R.f.f,  t*xk) .* ψoff)*R.f.f.coeff)
        end
        F .= xk .* quadgk!(integrand!, cache, 0, 1)[1]
        F .-= output
    end

    if !(J == nothing)
        J .= Diagonal(R.g((repeated_grad_xk_basis(R.f.f,  xk) .* ψoff)*R.f.f.coeff))
    end
end


functionalfg!(cache, ψoff, output::Array{Float64,1}, R::IntegratedFunction{m, Nψ, Nx}) where {m, Nψ, Nx} =
    (F, J, xk) -> functionalfg!(F, J, xk, cache, ψoff, output, R)
