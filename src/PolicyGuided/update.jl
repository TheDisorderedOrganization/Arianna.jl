"""
    PolicyGradientUpdate{P,O,VPR<:AbstractArray,VG<:AbstractArray} <: AriannaAlgorithm

Algorithm for updating policy parameters in Monte Carlo simulations.

# Fields
- `pools::Vector{P}`: Vector of independent pools (one for each system)
- `optimisers::O`: List of optimisers (one for each move)
- `learn_ids::Vector{Int}`: List of learnable moves
- `parameters_list::VPR`: List of current parameters values (one array for each move)
- `gradients_data::VG`: Gradient information (one for each move)
"""

struct PolicyGradientUpdate{P,O,VPR<:AbstractArray,VG<:AbstractArray} <: AriannaAlgorithm
    pools::Vector{P}            # Vector of independent pools (one for each system)
    optimisers::O               # List of optimisers (one for each move)
    learn_ids::Vector{Int}      # List of learnable moves
    parameters_list::VPR        # List of current parameters values (one array for each move)
    gradients_data::VG          # Gradient information (one for each move)

    function PolicyGradientUpdate(chains::Vector{S}, pools::Vector{P}, optimisers::O, gradients_data::VG) where {S,P,O,VG}
        # Safety checks
        @assert length(chains) == length(pools)
        @assert all(k -> all(move -> move.parameters == getindex.(pools, k)[1].parameters, getindex.(pools, k)), eachindex(pools[1]))
        @assert all(k -> all(move -> move.weight == getindex.(pools, k)[1].weight, getindex.(pools, k)), eachindex(pools[1]))
        @assert length(optimisers) == length(pools[1])
        # Find learnable actions
        learn_ids = [k for k in eachindex(optimisers) if !isa(optimisers[k], Static)]
        #Make sure that all policies and parameters across chains refer to the same objects
        policy_list = [move.policy for move in pools[1]]
        parameters_list = [move.parameters for move in pools[1]]
        for pool in pools
            for k in eachindex(policy_list)
                pool[k].policy = policy_list[k]
                pool[k].parameters = parameters_list[k]
            end
        end
        return new{P,O,typeof(parameters_list),VG}(pools, optimisers, learn_ids, parameters_list,  gradients_data)
    end

end

function PolicyGradientUpdate(chains; dependencies=missing, extras...)
    @assert length(dependencies) == 1
    @assert isa(dependencies[1], PolicyGradientEstimator)
    pge = dependencies[1]
    return PolicyGradientUpdate(chains, pge.pools, pge.optimisers, pge.gradients_data)
end

function make_step!(::Simulation, algorithm::PolicyGradientUpdate)
    for (k, lid) in enumerate(algorithm.learn_ids)
        gd = average(algorithm.gradients_data[k])
        learning_step!(algorithm.parameters_list[lid], gd, algorithm.optimisers[lid])
        algorithm.gradients_data[k] = initialise_gradient_data(algorithm.parameters_list[lid])
    end
    return nothing
end

function write_algorithm(io, algorithm::PolicyGradientUpdate, scheduler)
    println(io, "\tPolicyGradientUpdate")
    println(io, "\t\tCalls: $(length(filter(x -> 0 < x ≤ scheduler[end], scheduler)))")
    println(io, "\t\tLearnable moves: $(algorithm.learn_ids)")
    println(io, "\t\tOptimisers:")
    for (k, opt) in enumerate(algorithm.optimisers)
        println(io, "\t\t\tMove $(k): " * replace(string(opt), r"\{[^\{\}]*\}" => ""))
    end
end

nothing