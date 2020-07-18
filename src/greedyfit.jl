export greedyfit, update_component, update_coeffs



function greedyfit(Hk::HermiteMapk{m, Nψ, k}, X::Array{Float64,2}, maxterms::Int64; maxpatience::Int64 = 10^5) where {m, Nψ, k}

    best_valid_error = Inf
    patience = 0

    # Initialize map Hk to identity
    Hk = HermiteMapk(k, m; α = 1e-6);

    # Compute storage # Add later storage for validation S_valid
    S = Storage(Hk, X)

    # Compute initial loss on training set
    train_err = negative_log_likelihood!(S, Hk, X)(0.0, nothing, getcoeff(Hk))


    # Compute the reduced margin
    reduced_margin = getreducedmargin(getidx(Hk))

    while ncoeff(Hk) < maxterms




    end
end


function update_component(Hk::HermiteMapk{m, Nψ, k}, X::Array{Float64,2}, reduced_margin::Array{Int64,2}, S::Storage{m, Nψ, k}) where {m, Nψ, k}

    idxnew = vcat(getidx(Hk), reduced_margin)

    # Define updated map
    fnew = ExpandedFunction(Hk.I.f.f.B, idxnew, vcat(getcoeff(Hk), zeros(size(reduced_margin,1))))
    Hknew = HermiteMapk(fnew; α = 1e-6)



end

function update_coeffs(Hkold::HermiteMapk{m, Nψ, k}, Hknew::HermiteMapk{m, Nψnew, k}) where {m, Nψ, Nψnew, k}




end
