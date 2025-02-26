<h1 align="center">
  <img src="https://raw.githubusercontent.com/TheDisorderedOrganization/Arianna/main/logo.png" width="500"/>
</h1>

<p align="center"><i>A system-agnostic approach to Monte Carlo simulations</i></p>

<p align="center">
  <a href="https://github.com/TheDisorderedOrganization/MCMC/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-GPL%203.0-red.svg" alt="License">
  </a>
  <a href="https://github.com/TheDisorderedOrganization/Arianna/actions/workflows/ci.yml">
    <img src="https://github.com/TheDisorderedOrganization/Arianna/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
  <a href="https://codecov.io/gh/TheDisorderedOrganization/Arianna">
    <img src="https://codecov.io/gh/TheDisorderedOrganization/Arianna/graph/badge.svg?token=URGL1HJOOI" alt="codecov">
  </a>
</p>

<p align="center">
Arianna is a flexible and extensible framework for Monte Carlo simulations. Instead of acting as a black-box simulator, it provides a modular structure where users define their own system and Monte Carlo "moves". The package includes some simple predefined systems for example purposes, and more complex systems are defined in other repos like <a href="https://github.com/TheDisorderedOrganization/ParticlesMC">ParticlesMC</a>.
</p>

## Features

- **General-Purpose Monte Carlo Engine**: A lightweight framework that provides the core algorithms for Monte Carlo sampling, allowing users to define their own systems, moves, and proposal distributions.
- **Extensible Algorithms**: Built-in support for Metropolis-Hastings with the flexibility to implement advanced techniques like event-chain Monte Carlo.
- **Policy-Guided Monte Carlo**: Integrates adaptive sampling using policy gradient methods to optimise move parameters dynamically.
- **Predefined Systems**: Includes simple examples to help users get started quickly, with additional system implementations available through companion repositories like [ParticlesMC](https://github.com/TheDisorderedOrganization/ParticlesMC).

## Installation

To install the Arianna module, you can clone the repository and use the Julia package manager to add the module path to your environment.

```sh
git clone https://github.com/TheDisorderedOrganization/Arianna.git
cd Arianna
julia -e 'using Pkg; Pkg.activate("."); Pkg.instantiate()'
```

## Usage

Arianna is designed to work with user-defined systems rather than providing predefined ones. However, to help users get started, we provide example cases such as [particle_1D.jl](https://github.com/TheDisorderedOrganization/Arianna/blob/main/example/particle_1d/particle_1d.jl) in the [example](https://github.com/TheDisorderedOrganization/Arianna/tree/main/example) folder. Once you have defined your system and the associated moves, Arianna allows you to run Monte Carlo simulations and store relevant data. The following Julia script illustrates how to set up and execute a general simulation in the [particle_1D.jl](https://github.com/TheDisorderedOrganization/Arianna/blob/main/example/particle_1d/particle_1d.jl) example.

```julia
include("example/particle_1D/particle_1d.jl")

x₀ = 0.0
β = 2.0
M = 10
chains = [System(x₀, β) for _ in 1:M]
pool = (Move(Displacement(0.0), StandardGaussian(), ComponentArray(σ=0.1), 1.0),)
steps = 10^5
burn = 1000
sampletimes = build_schedule(steps, burn, 10)
path = "data/MC/particle_1d/Harmonic/beta$β/M$M/seed$seed"

algorithm_list = (
    (algorithm=Metropolis, pool=pool, seed=seed, parallel=false),
    (algorithm=StoreCallbacks, callbacks=(callback_energy, callback_acceptance), scheduler=sampletimes),
    (algorithm=StoreTrajectories, scheduler=sampletimes),
) 

simulation = Simulation(chains, algorithm_list, steps; path=path, verbose=true)
run!(simulation)
```
This implementation employs the **Metropolis algorithm** for Monte Carlo sampling of multiple independent chains, using Gaussian-distributed displacements as proposed moves. The simulation records energy and acceptance statistics while storing particle trajectories for analysis. The resulting data is saved in the specified output directory for further evaluation.

## Contributing

We welcome contributions from the community. If you have a new system or feature to add, please fork the repository, make your changes, and submit a pull request.

## Citing

If you use Arianna in your research, please cite it! You can find the citation information in the [CITATION](https://github.com/TheDisorderedOrganization/Arianna/blob/main/CITATION.bib) file or directly through GitHub’s "Cite this repository" button.

## License

This project is licensed under the GNU General Public License v3.0.  License. See the [LICENSE](https://github.com/TheDisorderedOrganization/Arianna/blob/main/LICENSE) file for details.

## Contact

For any questions or issues, please open an issue on the GitHub repository or contact the maintainers.
