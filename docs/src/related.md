# Related packages

There are many established Monte Carlo frameworks, each with different focuses. Arianna is designed to offer full flexibility in defining Monte Carlo moves and system-specific updates. Unlike black-box MCMC samplers, it allows users to implement custom moves (think of cluster updates in spin models) or domain-specific sampling strategies. Additionally, Arianna includes an adaptive Monte Carlo framework (via the `PolicyGuided` module) that dynamically adapt Monte Carlo moves to maximise sampling efficiency.

For MCMC sampling, some related packages include:
- [Turing.jl](https://github.com/TuringLang/Turing.jl) – A probabilistic programming library for Bayesian inference, built on MCMC methods like Metropolis-Hastings and Hamiltonian Monte Carlo.
- [AdvancedMH.jl](https://github.com/TuringLang/AdvancedMH.jl) – A package that provides enhanced Metropolis-Hastings algorithms with flexible proposal distributions.

For more general Monte Carlo and physics-based simulations:
- [AbstractMCMC.jl](https://github.com/TuringLang/AbstractMCMC.jl) - Abstract types and interfaces for Markov chain Monte Carlo methods for `TuringLang` MC modules.
- [MonteCarloX.jl](https://github.com/zierenberg/MonteCarloX.jl) – A Monte Carlo package package similar in spirit to Arianna that separates the algorithmic part from the system.
- [Molly.jl](https://github.com/JuliaMolSim/Molly.jl) – A molecular dynamics package exposing internals for customization that can also perform basic Monte Carlo simulations.
- [MonteCarlo.jl](https://carstenbauer.github.io/MonteCarlo.jl/dev/) – A framework for Monte Carlo simulations with support for various sampling techniques, mostly oriented to spin systems and Quantum Monte Carlo.

Arianna differentiates itself by prioritising flexibility in move design and adaptive sampling, making it particularly useful for physics-inspired Monte Carlo methods beyond standard statistical MCMC applications.