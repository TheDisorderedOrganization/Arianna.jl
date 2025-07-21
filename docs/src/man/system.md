# Adding Your Own System

Now that you understand how to run a Monte Carlo (MC) simulation, you may want to extend the framework by defining your own system. The [particle_1d.jl](https://github.com/TheDisorderedOrganization/Arianna.jl/examples/particle_1d/particle_1d.jl)  file provides a minimal example of a system, which you can use as a reference when creating a new one.

To define a new system, you need to specify its state variables, Monte Carlo moves and how to perform them. These components determine how the system evolves during the simulation. A system consists of:

## System

**Specify the state representation and the target probability density.**

- State representation: Defines the key quantities describing the system (e.g., position, energy, temperature). Your system has to be a struct where each element is a state variable. Example:
```julia
mutable struct Particle{T<:AbstractFloat}
    x::T
    β::T
    e::T
end
```
- Target density: This is the (unnormalised) log-density of the system. In this case it's simply the Boltzmann distribution at inverse temperature $\beta$. Note that it takes a generic state as input instead of the whole system object. This is better for performance, as generally the density only depends on a few properties of the system. In this case `state` is a tuple defined as `(e, β)`.
```julia
function Arianna.unnormalised_log_target_density(state, ::Particle)
    return -state[1] * state[2]
end
```

## Monte Carlo move

**Specify how the system state changes during the simulation**

- Define an action. In the example, the action is a displacement.
```julia
mutable struct Displacement{T<:AbstractFloat} <: Action
    δ::T
end
```
- Define how this action is sampled in the `sample_action!` function. In the example, the displacement length is sampled from a normal distribution.
```julia
function Arianna.sample_action!(action::Displacement,::StandardGaussian, parameters, system::Particle, rng)
    action.δ = rand(rng, Normal(zero(action.δ), parameters.σ))
    return nothing
end
```
- Specify the probablity of proposing the action in `log_proposal_density`. Note that this function must give the exact probability of sampling the action with `sample_action!`. In this case, we just need the density of the normal distribution.
```julia
function Arianna.log_proposal_density(action::Displacement, ::StandardGaussian, parameters, system::Particle)
    return -(action.δ)^2 / (2parameters.σ^2) - log(2π * parameters.σ^2) / 2
end
```
- Finally, provide how this action changes the state of the system in the `perform_action!` function. In the example, performing the displacement updates the position and the energy of the particle. Note that `perform_action!` returns information on the state of the system before and after the action. This doesn't have to be the whole system, but just the part relevant for evaluating the density ratio in `delta_log_target_density`.
```julia
function Arianna.perform_action!(system::Particle, action::Displacement)
    e₁ = system.e
    system.x += action.δ
    system.e = potential(system.x)
    return (e₁, system.β), (system.e, system.β)
end
```

By modifying and extending the existing particle_1D.jl example, you can create a variety of physical and mathematical models suitable for MC simulations.

## Checking if everything has been implemented

To check if all required functions have been implemented, you can run `Arianna.Linter.run_linter("path/to/your/folder/or/file")`. This will output a tree of functions and structs indicating what systems / actions / policies / functions are implemented.

**Example**

```julia
julia> Arianna.Linter.run_linter("examples/particle_1d/particle_1D.jl")

Linting : examples/particle_1d/particle_1d.jl

└─ System: Particle
   ├─ Function: unnormalised_log_target_density    ✅ Defined
   └─ Action: Displacement
      ├─ Function: perform_action!                    ✅ Defined
      ├─ Function: invert_action!                     ✅ Defined
      ├─ Function: revert_action!                     ❕ Not defined
      ├─ Function: reward                             ✅ Defined
      ├─ Policy: StandardGaussian
      │  ├─ Function: sample_action!                     ✅ Defined
      │  └─ Function: log_proposal_density               ✅ Defined
      └─ Policy: StandardUniform
         ├─ Function: sample_action!                     ✅ Defined
         └─ Function: log_proposal_density               ✅ Defined
````
