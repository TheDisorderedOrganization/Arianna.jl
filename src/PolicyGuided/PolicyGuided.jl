"""
    PolicyGuided

Module for policy-guided Monte Carlo algorithms.
"""
module PolicyGuided

using ..Arianna: AriannaSystem, Action, Policy, AriannaAlgorithm, Simulation, Metropolis
import ..Arianna: make_step!, write_algorithm, sample_action!, perform_action!, delta_log_target_density, log_proposal_density, invert_action!, revert_action!, raise_error
using Random
using LinearAlgebra
using Transducers
using ForwardDiff

include("gradients.jl")
include("learning.jl")
include("estimator.jl")
include("update.jl")

export Static, VPG, BLPG, BLAPG, NPG, ANPG, BLANPG, reward
export PolicyGradientEstimator, PolicyGradientUpdate

end