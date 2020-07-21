export  HermiteMap,
        evaluate!,
        evaluate,
        inverse




struct HermiteMap{m}
    Lk::Array{LinHermiteMapk,1}

    # Regularization parameter
    α::Float64

end
