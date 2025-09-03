# MAGNUS: Modeling, Analysis & desiGN of Uncertain Systems #

MAGNUS provides a collection of classes to support the development and analysis of mathematical models. It is written in C++ to promote execution speed and includes Python binders through [pybind11](https://pybind11.readthedocs.io/en/stable/).

MAGNUS builds on [MC++](https://github.com/omega-icl/mcpp) for expression tree manipulation, differentiation and evaluation, along with [CRONOS](https://github.com/omega-icl/cronos) for numerical integration and sensitivity analysis of dynamic systems and [CANON](https://github.com/omega-icl/canon) for local and global numerical optimization.

The present version 1.1 of MAGNUS has capability for:

* Parameter estimation in mathematical models
* Model-based feasibility analysis using nested sampling
* Model-based design of experiments

A range of python notebooks are provided in `src/interface` to illustrate these capabilities.

---
### Setting up MAGNUS ###

Refer to [INSTALL.md](./INSTALL.md) for instructions.

### Contacts ###

* Repo owner: [Benoit C. Chachuat](https://profiles.imperial.ac.uk/b.chachuat)
* OMEGA Research Group

---
### References ###

* Bernardi, A., Gomoescu, L., Wang, J., Pantelides, C.C., Chadwick, D. & Chachuat, B., [Kinetic Model Discrimination for Methanol and DME Synthesis using Bayesian Estimation](https://doi.org/10.1016/j.ifacol.2019.06.084), _IFAC-PapersOnLine_ **52**(1), 335-340, 2019
* Chachuat, B., Sandrin, M. & Pantelides, C.C., [Optimal Experiment Campaigns under Uncertainty Minimizing Bayes Risk](https://doi.org/10.1016/j.ifacol.2025.07.196), _IFAC-PapersOnLine_ **59**(6), 504-509, 2025
* Gomoescu, L., [Bayesian Uncertainty Quantification in Process Modelling: Applications in Parameter Estimation and Feasibility Analysis](https://doi.org/10.25560/109494), PhD Thesis, Department of Chemical Engineering, Imperial College London, 2022
* Kusumo, K.P., [Algorithms and Tools for Feasibility Analysis and Optimal Experiment Design in Pharmaceutical Manufacturing](https://doi.org/10.25560/96978), PhD Thesis, Department of Chemical Engineering, Imperial College London, 2022
* Kusumo, K.P., Kuriyan, K., Vaidyaraman, S., Garcia Muñoz, S. Shah, N. & Chachuat, B., [Probabilistic Framework for Optimal Experimental Campaigns in the Presence of Operational Constraints](https://doi.org/10.1039/D1RE00465D), _Reaction Chemistry & Engineering_ **7**(11), 2359-2374, 2022
* Kusumo, K.P., Kuriyan, K., Vaidyaraman, S., Garcia Muñoz, S. Shah, N. & Chachuat, B., [Risk Mitigation in Model-Based Experiment Design: A Continuous-Effort Approach to Optimal Campaigns](https://doi.org/10.1016/j.compchemeng.2022.107680), _Computers & Chemical Engineering_ **159**,107680, 2022
* Kusumo, K.P., Gomoescu, L., Paulen, R., Garcia Muñoz, S., Pantelides, C.C., Shah, N. & Chachuat, B. [Bayesian Approach to Probabilistic Design Space Characterization: A Nested Sampling Strategy](https://doi.org/10.1021/acs.iecr.9b05006), _Industrial & Engineering Chemistry Research_ **59**(6), 2396-2408, 2020.
* Kusumo, K.P., Morrissey, R., Mitchell, H., Shah, N. & Chachuat, B., [A Design Centering Methodology for Probabilistic Design Space](https://doi.org/10.1016/j.ifacol.2021.08.222), _IFAC-PapersOnLine_ **54**(3), 79-84, 2021
* Mowbray, M., Kontoravdi, C., Shah, N. & Chachuat, B., [A Decomposition Approach to Characterizing Feasibility in Acyclic Multi-Unit Processes](https://doi.org/10.1016/j.ifacol.2024.08.339), _IFAC-PapersOnLine_ **58**(14), 216-221, 2024
* Pitt, J.A., Gomoescu, L., Pantelides, C.C., Chachuat, B. & Banga, J.R., [Critical Assessment of Parameter Estimation Methods in Models of Biological Oscillators](https://doi.org/10.1016/j.ifacol.2018.09.040), _IFAC-PapersOnLine_ **51**(19), 72-75, 2018
* Sandrin, M., Pantelides, C.C. & Chachuat, B., [Methodological and Computational Framework for Model-based Design of Parallel Experimental Campaigns under Uncertainty](https://doi.org/10.1016/j.jprocont.2025.103465), _Journal of Process Control_ **152**, 103465, 2025
* Sandrin, M., Kusumo, K.P., Pantelides, C.C. & Chachuat, B., [Solving for Exact Designs in Optimal Experiment Campaigns under Uncertainty](https://doi.org/10.1016/j.ifacol.2024.08.412), _IFAC-PapersOnLine_ **58**(14), 658-663, 2024
