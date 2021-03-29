export Metrics, rmse, spread, mean_hist, metric_hist

#Structure that hold the different metrics for each case
struct Metrics
Ne::Int64

# RMSE
"Rmse"
rmse::Array{Float64,1}

"Median of the rmse"
rmse_med::Float64

"Mean of the rmse"
rmse_mean::Float64

"Standard deviation of the rmse"
rmse_std::Float64

# SPREAD
"Spread"
spread::Array{Float64,1}

"Median of the spread"
spread_med::Float64

"Mean of the rmse"
spread_mean::Float64

"Standard deviation of the rmse"
spread_std::Float64

# COVERAGE PROBABILTY
"Coverage probability"
covprob::Array{Float64,1}

"Median of the coverage probability"
covprob_med::Float64

"Mean of the coverage probability"
covprob_mean::Float64

"Standard deviation of the coverage probability"
covprob_std::Float64
end


# Definition from Spantini et al. 2019
"""
    rmse(X)

Compute the root-mean square error of the ensemble matrix `X`    
"""
rmse(x::Array{Float64,1}, X::Array{Float64,2}) = norm(mean(X; dims = 2)[:,1]-x)/sqrt(size(X,1))

"""
    spread(X)

Compute the spread of the ensemble matrix `X`
"""
spread(X::Array{Float64,2}) = sqrt(tr(cov(X'))/size(X,1))

"""
    mean_hist(hist)

Stack together the mean of the different ensembles.
"""
function mean_hist(hist::Array{Array{Float64,2},1})
    l = length(hist)
    Nx = size(hist[1])[1]
    x̂ = zeros(Nx, l)

    @inbounds for i=1:l
        x̂[:,i] .= mean(hist[i]; dims = 2)[:,1]
    end
    return x̂
end

function quant(hist::Array{Array{Float64,2},1})
    l = length(hist)
    Nx = size(hist[1])[1]
    qinf = zeros(Nx, l)
    qsup = zeros(Nx, l)


    for j=1:Nx
        for i=1:l
            q₋, q₊ = quantile(hist[i][j,:],[0.025, 0.975])
            qinf[j,i] = deepcopy(q₋)
            qsup[j,i] = deepcopy(q₊)
        end
    end

    return qinf, qsup
end

# Create function to construct the median, mean and std a metric of two arrays
function metric_hist(metric::Function, hist::Array{Array{Float64,2},1})
    l = length(hist)
    Metric = map(i->metric(hist[i]),1:l)
    return Metric, median(Metric), mean(Metric), std(Metric)
end


# Create function to construct the median, mean and std a metric of two arrays
function metric_hist(metric::Function, xstar::Array{Float64,2}, hist::Array{Array{Float64,2},1})
    l = length(hist)
    Metric = map(i->metric(xstar[:,i], hist[i]),1:l)
    return Metric, median(Metric), mean(Metric), std(Metric)
end
