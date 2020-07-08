

export IntegratedFunction


struct IntegratedFunction{Nψ, m, Nx}
    g::Rectifier
    f::ExpandedFunction{Nψ, m, Nx}
end


function (R::IntegratedFunction{Nψ, m, Nx})(x::Array{T,1}) where {Nψ, m, Nx, T<:Real}
    return f() + quadgk(R.g())

end
