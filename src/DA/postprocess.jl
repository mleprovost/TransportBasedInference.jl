export post_process

function post_process(data::SyntheticData, model::Model, J::Int64, statehist::Array{Array{Float64,2},1})
    Ne = size(statehist[1],2)
    @show Ne
    # Post_process compute the statistics (mean, median, and
    # standard deviation) of the RMSE, spread and coverage
    # probability over (J-T_BurnIn) assimilation steps.

    # enshist contains the initial condition, so one more element
    idx_xt = model.Tstep+1:model.Tspinup+model.Tstep
    idx_ens = model.Tstep-model.Tburn+2:model.Tstep+1
    # Compute root mean square error statistics
    Rmse, Rmse_med, Rmse_mean, Rmse_std = metric_hist(rmse, data.xt[:,idx_xt], statehist[idx_ens])
    # Compute ensemble spread statistics
    Spread, Spread_med, Spread_mean, Spread_std = metric_hist(spread, statehist[idx_ens])
    # Compute quantile information
    qinf, qsup = quant(statehist[idx_ens])

    # Compute coverage probability statistics

    Covprob = zeros(length(idx_xt))
    b = zeros(Bool, model.Nx)
    for (i,idx) in enumerate(idx_xt)
        for j=1:model.Nx
        b[j] = (qsup[j,i] >= data.xt[j,idx] >= qinf[j,i])
        end
        Covprob[i] = deepcopy(mean(b))
    end

    Covprob_med  = median(Covprob)
    Covprob_mean = mean(Covprob)
    Covprob_std  = std(Covprob)

    Metric = Metrics(Ne, Rmse, Rmse_med, Rmse_mean, Rmse_std, Spread, Spread_med,
                    Spread_mean, Spread_std, Covprob, Covprob_med,
                    Covprob_mean, Covprob_std)
    return Metric
end
