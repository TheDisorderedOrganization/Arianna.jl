using Arianna
using Arianna.PolicyGuided
using Random
using Distributions
using ComponentArrays

###############################################################################
## SYSTEM
mutable struct Ising1D{T<:AbstractFloat} <: AriannaSystem
    spin::Vector{Int}
    j::T
    h::T
    β::T
    e::T
end

Arianna.length(system::Ising1D) = length(system.spin)

function Ising1D(N::Int, j::T, h, β::T) where {T<:AbstractFloat}
    random_spin = rand([-1, 1], N)  # Example size of 100 spins
    system = Ising1D(random_spin, j, h, β, 0.0)
    e = compute_energy(system)
    system.e = e
    return system
end

function compute_energy(system::Ising1D)
    return sum(compute_energy_spin(system, i) for i in 1:length(system.spin)) / 2
end

function compute_energy_spin(system::Ising1D, i::Int)
    # Compute the energy contribution of the spin at index i
    N = length(system.spin)
    left = system.spin[mod1(i-1, N)]  # Previous spin (wraps around)
    right = system.spin[mod1(i+1, N)]  # Next spin (wraps around)
    interaction = - system.j * (left + right) * system.spin[i]
    externamagnetic = - system.h * system.spin[i]
    return interaction + externamagnetic
end

function Arianna.unnormalised_log_target_density(state, ::Ising1D)
    return -state[1] * state[2]
end

###############################################################################
## ACTIONS
mutable struct Flip <: Action
    i::Int
end

function Arianna.perform_action!(system::Ising1D, action::Flip)
    e₁ = system.e
    ei = compute_energy_spin(system, action.i)
    system.spin[action.i] *= -1  # Flip the spin at index i
    ef = compute_energy_spin(system, action.i)
    system.e = e₁ - ei + ef  # Update the energy
    return (e₁, system.β), (system.e, system.β)
end

function Arianna.invert_action!(::Flip, ::Ising1D)
    return nothing
end


###############################################################################
## POLICIES
struct StandardUniform <: Policy end


function Arianna.sample_action!(action::Flip, ::StandardUniform, parameters, system::Ising1D, rng)
    action.i = rand(rng, 1:length(system.spin))  # Randomly select a spin to flip
    return nothing
end

function Arianna.log_proposal_density(action::Flip, ::StandardUniform, parameters, system::Ising1D)
    # Uniform proposal density for flipping a spin
    return 1.0
end

struct EnergyBias <: Policy end


function Arianna.sample_action!(action::Flip, ::EnergyBias, parameters, system::Ising1D, rng)
    local_energy = [compute_energy_spin(system, i) for i in 1:length(system.spin)]
    partition = sum(exp.(parameters.θ .* local_energy))  # Compute partition function
    weights = exp.(parameters.θ .* local_energy) ./ partition # Compute probabilities based on local energy
    if !isapprox(sum(weights), 1)
        println(parameters.θ)
    end
    #println("Weights: ", weights)
    id1 = rand(rng, Categorical(weights))
    action.i = id1 # Randomly select a spin to flip
    return nothing
end

function Arianna.log_proposal_density(action::Flip, ::EnergyBias, parameters, system::Ising1D)
    local_energy = [compute_energy_spin(system, i) for i in 1:length(system.spin)]
    partition = sum(exp.(parameters.θ * local_energy))  # Compute partition function
    # Uniform proposal density for flipping a spin
    return parameters.θ * local_energy[action.i] - log(partition)
end

function Arianna.PolicyGuided.reward(::Flip, ::Ising1D)
    # Reward is the change in energy due to flipping the spin
    return 1.0
end


###############################################################################
## UTILS
function Arianna.store_trajectory(io, system::Ising1D, t::Int, ::DAT)
    println(io, "$t $(system.spin)")
    return nothing
end

function callback_energy(simulation)
    return mean(system.e for system in simulation.chains)
end

###############################################################################

nothing
