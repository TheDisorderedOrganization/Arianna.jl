using Arianna
using Arianna.PolicyGuided
using Random
using Distributions
using ComponentArrays

###############################################################################
## SYSTEM
mutable struct Particle{T<:AbstractFloat} <: AriannaSystem
    x::T
    β::T
    e::T
    function Particle(x::T, β::T) where {T<:AbstractFloat}
        return new{T}(x, β, potential(x))
    end
end

System(x, β) = Particle(x, β)

function Arianna.unnormalised_log_target_density(state, ::Particle)
    return -state[1] * state[2]
end

###############################################################################
## ACTIONS
mutable struct Displacement{T<:AbstractFloat} <: Action
    δ::T
end

function Arianna.perform_action!(system::Particle, action::Displacement)
    e₁ = system.e
    system.x += action.δ
    system.e = potential(system.x)
    return (e₁, system.β), (system.e, system.β)
end

function Arianna.invert_action!(action::Displacement, system::Particle)
    action.δ = -action.δ
    return nothing
end

function Arianna.PolicyGuided.reward(action::Displacement, system::Particle)
    return (action.δ)^2
end

###############################################################################
## POLICIES
struct StandardGaussian <: Policy end

setup_parameters(::StandardGaussian) = ComponentArray(σ=1.0)

function Arianna.log_proposal_density(action::Displacement, ::StandardGaussian, parameters, system::Particle)
    return -(action.δ)^2 / (2parameters.σ^2) - log(2π * parameters.σ^2) / 2
end

function Arianna.sample_action!(action::Displacement, ::StandardGaussian, parameters, system::Particle, rng)
    action.δ = rand(rng, Normal(zero(action.δ), parameters.σ))
    return nothing
end

###############################################################################
## UTILS
function Arianna.store_trajectory(io, system::Particle, t::Int, format::DAT)
    println(io, "$t $(system.x)")
    return nothing
end

function callback_energy(simulation)
    return mean(system.e for system in simulation.chains)
end

###############################################################################

nothing