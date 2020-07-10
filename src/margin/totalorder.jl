
export totalorder


function totalorder(order::Array{Int64,1})
    # totalorder: Compute indices of total degree
    # polynomial expansion. Output is an (ncoeff x d) matrix
    # where 0 corresponds to the constant function

    if isempty(order)
        return zeros(Int64,0,0)
    end

    # Determine dimension and max_order
    d = size(order, 1)
    max_order = maximum(order)

    # Initialize multi_idx with zeros
    midxs_new = zeros(d)
    multi_idxs = deepcopy(midxs_new)

    @show multi_idxs

    # Initialize midxs_old_set
    if sum(midxs_new) < max_order
        midxs_old_set = deepcopy(midxs_new)
    else
        midxs_old_set = zeros(Int64, 1, d)
    end

    # Generate higher order multi-indices
    for i = 1:max_order

        # Initialize empty set of starting multi-indices
        midxs_new_set = zeros(Int64, 1, d)

        # Extract each multi_idx in midxs_old_set
        for j = 1:size(midxs_old_set,1)
            @show j, midxs_new_set
            midxs_old_j = midxs_new_set[j]

            # Expand index set along each direction
            for k = 1:d

                # If allowable, add new multi_idx
                if midxs_old_k[k] < order_list[k]
                    midx_new = midxs_old_j
                    midx_new[k] += 1
                    multi_idxs = vcat(multi_idxs, midx_new)

                    # If boundary of orders isn't added, expand
                    # in the next iteration by adding to set
                    if sum(midx_new) < max_order
                        midxs_new_set = vcat(midxs_new_set, midx_new)
                    end
                end
            end
        end

        # Overwrite midxs_old_set for next iteration
        midxs_old_set = deepcopy(midxs_new_set)
    end

    # Remove duplicates in multi_idxs
    multi_idxstmp = unique(eachslice(order; dims = 1))
    cardinal = size(multi_idxstmp,1)
    multi_idxstmp = zeros(cardinal)

    @inbounds for i=1:cardinal

    end


end
