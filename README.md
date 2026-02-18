## Codes used in: *[Strategy optimization for Bayesian quantum parameter estimation with finite copies: Adaptive greedy, parallel, sequential and general strategies](https://arxiv.org/abs/2602.09655)*
#### Erik L. André, Jessica Bavaresco and Mohammad Mehboudi

This repository contains all the code files used to obtain the data in the article "Strategy optimization for Bayesian quantum parameter estimation with finite copies: Adaptive greedy, parallel, sequential and general strategies"

To be run in MATLAB, [QETLAB](http://www.qetlab.com/), [SeDuMi](https://github.com/sqlp/sedumi) and [Mosek](https://docs.mosek.com/11.0/toolbox/index.html), as well as the [Parallel Computing Toolbox](https://www.mathworks.com/products/parallel-computing.html) are required. The following files are included in the repository:

- [testeroptimization_sdp_kcopy_seesaw.m](https://github.com/eriklandre/Bayesian-quantum-parameter-estimation/blob/main/testeroptimization_sdp_kcopy_seesaw.m):
**Optimizes the tester given $k$ copies of the channel encoding the unknown parameter(s)**
- [phaseestimation_and_noise_kcopy.m](https://github.com/eriklandre/Bayesian-quantum-parameter-estimation/blob/main/phaseestimation_and_noise_kcopy.m):
**Solves the $k$-copy noisy phase estimation problem (set $p=0$ for the usual phase estimation problem)**
- [thermometry_greedy.m](https://github.com/eriklandre/Bayesian-quantum-parameter-estimation/blob/main/thermometry_greedy.m):
**Solves the adaptive greedy thermometry problem**
  - [thermometry_greedy_cache.m](https://github.com/eriklandre/Bayesian-quantum-parameter-estimation/blob/main/thermometry_greedy_cache.m):
**Solves the adaptive greedy thermometry problem but loading the data for each outcome, such that the code preloads the necessary testers and estimators, yielding a smaller running time**
- [thermometry_kcopy.m](https://github.com/eriklandre/Bayesian-quantum-parameter-estimation/blob/main/thermometry_kcopy.m):
**Solves the $k$-copy thermometry problem**
- [unitary_and_noise_greedy.m](https://github.com/eriklandre/Bayesian-quantum-parameter-estimation/blob/main/unitary_and_noise_greedy.m):
**Solves the adaptive greedy noisy SU(2) estimation problem (set $p=0$ for the usual SU(2) estimation problem)**
- [unitary_and_noise_kcopy.m](https://github.com/eriklandre/Bayesian-quantum-parameter-estimation/blob/main/unitary_and_noise_kcopy.m):
**Solves the $k$-copy noisy SU(2) estimation problem (set $p=0$ for the usual SU(2) estimation problem)**
