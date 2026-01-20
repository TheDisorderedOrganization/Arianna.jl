"""
    module Arianna

Arianna is a flexible and extensible framework for Monte Carlo simulations. Instead of acting as a black-box simulator, it
  provides a modular structure where users define their own system and Monte Carlo "moves". 
"""
module Arianna

using Random
using Distributions
using Statistics
using LinearAlgebra
using Transducers
using Dates
using Printf

"""
    abstract type AriannaSystem

Abstract type representing a system that can be simulated using methods defined in the `Arianna` module.
"""
abstract type AriannaSystem end

Base.length(::AriannaSystem) = 1 # Default length for systems

export AriannaSystem

include("simulation.jl")
export Simulation, build_schedule, run!

include("algorithms.jl")
export AriannaAlgorithm, StoreCallbacks, StoreTrajectories, StoreLastFrames, StoreBackups, PrintTimeSteps
export TXT, DAT
export @callback

include("metropolis.jl")
export Action, Policy, Move, MetropolisRule, BarkerRule
export sample_action!, perform_action!, revert_action!, invert_action!
export log_proposal_density, delta_log_target_density
export mc_step!, mc_sweep!
export Metropolis, StoreParameters, StoreAcceptance

include("PolicyGuided/PolicyGuided.jl")
include("Linter/Linter.jl")

end
