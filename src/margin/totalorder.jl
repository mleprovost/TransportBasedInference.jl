
export totalorder


function totalorder(order_list::Array{Int64,1})
    # totalorder: Compute indices of total degree
    # polynomial expansion. Output is an (ncoeff x d) matrix
    # where 0 corresponds to the constant function

    if isempty(order_list)
        return zeros(Int64,0,0)
    end

    # Determine dimension and max_order
    d = size(order_list, 1)
    max_order = maximum(order_list)

    # Initialize multi_idx with zeros
    midxs_new = zeros(Int64, 1, d)
    multi_idxs = deepcopy(midxs_new)

    # Initialize midxs_old_set
    if sum(midxs_new) < max_order
        midxs_old_set = deepcopy(midxs_new)
    else
        midxs_old_set = zeros(Int64, 0, d)
    end

    # Generate higher order multi-indices
    for i = 1:max_order
        # Initialize empty set of starting multi-indices
        midxs_new_set = zeros(Int64, 0, d)

        # Extract each multi_idx in midxs_old_set
        for j = 1:size(midxs_old_set,1)
            midxs_old_j = deepcopy(midxs_old_set[j,:])
            # Expand index set along each direction
            for k = 1:d

                # If allowable, add new multi_idx
                if midxs_old_j[k] < order_list[k]
                    midx_new = deepcopy(midxs_old_j)
                    midx_new[k] += 1
                    multi_idxs = vcat(multi_idxs, reshape(midx_new,(1,d)))

                    # If boundary of orders isn't added, expand
                    # in the next iteration by adding to set
                    if sum(midx_new) < max_order
                        midxs_new_set = vcat(midxs_new_set, reshape(midx_new,(1,d)))
                    end
                end
            end
        end

        # Overwrite midxs_old_set for next iteration
        midxs_old_set = deepcopy(midxs_new_set)
    end

    # Remove duplicates in multi_idxs
    multi_idxstmp = unique(eachslice(multi_idxs; dims = 1))
    cardinal = size(multi_idxstmp,1)
    multi_idxs = zeros(Int64, cardinal,d)

    @inbounds for i=1:cardinal
        multi_idxs[i,:] = multi_idxstmp[i]
    end

    multi_idxs = sortslices(multi_idxs; dims = 1)


    return multi_idxs

end
