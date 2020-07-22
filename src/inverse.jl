export functional!

function functional!(F, J, xk, cache, ψoff, output::Array{Float64,1}, R::IntegratedFunction{m, Nψ, Nx}) where {m, Nψ, Nx}

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


functional!(cache, ψoff, output::Array{Float64,1}, R::IntegratedFunction{m, Nψ, Nx}) where {m, Nψ, Nx} =
    (F, J, xk) -> functional!(F, J, xk, cache, ψoff, output, R)
