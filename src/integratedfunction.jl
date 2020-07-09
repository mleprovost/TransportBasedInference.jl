

export IntegratedFunction


struct IntegratedFunction{Nψ, m, Nx}
    g::Rectifier
    f::ExpandedFunction{m, Nx}
end


function (R::IntegratedFunction{m, Nx})(x::Array{T,1}) where {m, Nx, T<:Real}
    return f() + quadgk(R.g())

end
