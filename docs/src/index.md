# AdaptiveTransportMap.jl

*A framework for Bayesian inference with transport maps*

The objective of this package is to allow for easy and fast resolution of Bayesian inference problems using transport maps. The package provides tools for
- joint and conditional density estimation from limited samples of the target distribution.
- sequential ensemble data assimilation with the stochastic ensemble Kalman filter, the ensemble transform Kalman filter and a nonlinear generalization of the stochastic ensemble Kalman filter.

This package has been designed to address the following question:

For a random variable $\mathsf{X} \in \mathbb{R}^n$ with distribution $\pi$
Given $\{ \boldsymbol{x}^i \}, \; \text{for} \; i = 1, \ldots, N_e$

For a random variable $\mathsf{X} \sim \pi$ where $\pi$ a complex distribution of interest, can we identify a transformation $\boldsymbol{S}$ such that the target distribution is pushed forward to a reference density $\eta$ (e.g. the standard Gaussian distribution).


Given two probability measuresνπandνηonRd, assumed to have densitiesπandηrespectively,acouplingis a pair of random variablesXandZwhich admitπandηas marginals. One specialkind of coupling is given by a deterministic transformationS:Rd→Rdsuch thatS(X) =Zindistribution. The transformationSthat satisfies this property is called atransport map[39].


## References

[^1]: Baptista, R., Zahm, O., & Marzouk, Y. (2020). An adaptive transport framework for joint and conditional density estimation. arXiv preprint arXiv:2009.10303.

[^2]: Spantini, A., Baptista, R., & Marzouk, Y. (2019). Coupling techniques for nonlinear ensemble filtering. arXiv preprint arXiv:1907.00389.

[^3]: Le Provost, M., Baptista, R., Marzouk, Y., & Eldredge, J. (2021). A low-rank nonlinear ensemble filter for vortex models of aerodynamic flows. In AIAA Scitech 2021 Forum (p. 1937).
