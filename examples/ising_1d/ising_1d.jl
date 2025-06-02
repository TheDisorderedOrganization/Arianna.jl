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
    sum_neighbours = sum(system.spin[i] * system.spin[i+1] for i in 1:length(system.spin)-1)
    pbs_neighbor = system.spin[end] * system.spin[1]
    interaction = -system.j * (sum_neighbours + pbs_neighbor)
    externamagnetic = -system.h * sum(system.spin)
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
    system.spin[action.i] *= -1  # Flip the spin at index i
    system.e = compute_energy(system)
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