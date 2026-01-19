"""
    abstract type Action

Abstract type representing Monte Carlo actions/moves that can be performed on a system.

Concrete subtypes must implement:
- `sample_action!(action, policy, parameters, system, rng)`: Sample a new action from the policy
- `perform_action!(system, action)`: Apply the action to modify the system state
- `invert_action!(action, system)`: Invert/reverse the action
- `log_proposal_density(action, policy, parameters, system)`: Log probability density of proposing this action

Optional methods:
- `revert_action!(system, action)`: Optimized version of perform_action! that can reuse cached values
"""
abstract type Action end

"""
    abstract type Policy

Abstract type representing proposal policies for Monte Carlo actions.

A Policy defines how actions are sampled and their proposal probabilities are calculated.
Concrete subtypes work together with specific Action types to implement the proposal mechanism.
"""
abstract type Policy end

struct AriannaBaseFunctionMissing <: Exception
    message::String
end

Base.showerror(io::IO, e::AriannaBaseFunctionMissing) = print(io, "AriannaBaseFunctionMissing: $(e.message)")

"""
    raise_missingfunction(s)

Helper function to raise errors for unimplemented required methods.

# Arguments
- `s`: String describing the missing method implementation
"""
raise_missingfunction(s) = throw(AriannaBaseFunctionMissing("$(s) is not defined. Please implement this required method."))

"""
    sample_action!(action::Action, policy::Policy, parameters, system::AriannaSystem, rng)

Sample a new action from the policy.

# Arguments
- `action`: Action to be sampled
- `policy`: Policy to sample from
- `parameters`: Parameters of the policy
- `system`: System the action will be applied to
- `rng`: Random number generator
"""
sample_action!(::Action, ::Policy, parameters, ::AriannaSystem, rng) = raise_missingfunction("sample_action!")

"""
    log_proposal_density(action, policy, parameters, system::AriannaSystem)

Calculate the log probability density of proposing the given action.

# Arguments
- `action`: Proposed action
- `policy`: Policy used for proposal
- `parameters`: Parameters of the policy
- `system`: System the action is applied to
"""
log_proposal_density(action, policy, parameters, ::AriannaSystem) = raise_missingfunction("log_proposal_density")

"""
    perform_action!(system::AriannaSystem, action::Action)

Apply the action to modify the system state.

# Arguments
- `system`: System to modify
- `action`: Action to perform

# Returns
A tuple of (x₁, x₂) containing the old and new states
"""
perform_action!(::AriannaSystem, ::Action) = raise_missingfunction("perform_action!")

"""
    unnormalised_log_target_density(x, system)

Calculate the unnormalized log probability density of a system state.

# Arguments
- `x`: System state
- `system`: System object
"""
unnormalised_log_target_density(x, ::AriannaSystem) = raise_missingfunction("unnormalised_log_target_density")
"""
    delta_log_target_density(x₁, x₂, system::AriannaSystem)

Calculate the change in log target density between two states.

# Arguments
- `x₁`: Initial state
- `x₂`: Final state
- `system`: System object
"""
delta_log_target_density(x₁, x₂, system::AriannaSystem) = unnormalised_log_target_density(x₂, system) - unnormalised_log_target_density(x₁, system)
"""
    invert_action!(action::Action, system::AriannaSystem)

Invert/reverse an action.

# Arguments
- `action`: Action to invert
- `system`: System the action was applied to
"""
invert_action!(::Action, ::AriannaSystem) = raise_missingfunction("invert_action!")

"""
    revert_action!(system, action::Action)

Optimized version of perform_action! that can reuse cached values.

# Arguments
- `system`: System to modify
- `action`: Action to perform
"""
revert_action!(system::AriannaSystem, action::Action) = perform_action!(system, action)

"""
    Move{A<:Action,P<:Policy,V<:AbstractArray,T<:AbstractFloat}

A struct representing a Monte Carlo move with an associated action, policy, and parameters.

# Fields
- `action::A`: The action to be performed in the move
- `policy::P`: The policy used to propose actions
- `parameters::V`: Parameters of the policy
- `weight::T`: Weight/probability of selecting this move in a sweep
- `total_calls::Int`: Total number of times this move has been attempted
- `accepted_calls::Int`: Number of times this move has been accepted

# Type Parameters
- `A`: Type of the action
- `P`: Type of the policy
- `V`: Type of the parameter array
- `T`: Type of the weight (floating point)
"""
mutable struct Move{A<:Action,P<:Policy,V<:AbstractArray,T<:AbstractFloat}
    action::A
    policy::P
    parameters::V
    weight::T
    total_calls::Int
    accepted_calls::Int
end

"""
    Move(action, policy, parameters, weight)

Create a new Move instance.

# Arguments
- `action`: The action to be performed in the move
- `policy`: The policy used to propose actions
- `parameters`: Parameters of the policy
- `weight`: Weight/probability of selecting this move in a sweep
"""
function Move(action, policy, parameters, weight)
    return Move(action, policy, parameters, weight, 0, 0)
end

"""
    abstract type AriannaRule

Abstract type representing acceptance rules for Monte Carlo algorithms.
"""
abstract type AriannaRule end


"""
    MetropolisRule <: AriannaRule

Acceptance rule for the classical Metropolis Monte Carlo algorithm.
Accepts a proposal with probability min(1, exp(log_acceptance_ratio)).
"""
struct MetropolisRule <: AriannaRule end

"""
    BarkerRule <: AriannaRule

Acceptance rule for the Barker Monte Carlo algorithm.
Accepts a proposal with probability 1 / (1 + exp(-log_acceptance_ratio)).
"""
struct BarkerRule <: AriannaRule end


"""
    acceptance_rule(::MetropolisRule, log_acceptance_ratio)

Compute the acceptance probability according to the Metropolis rule.

# Arguments
- `log_acceptance_ratio`: The logarithm of the acceptance ratio

# Returns
- Acceptance probability (between 0 and 1)
"""
function acceptance_rule(::MetropolisRule, log_acceptance_ratio::T) where {T<:AbstractFloat}
    return min(one(T), exp(log_acceptance_ratio))
end

"""
    acceptance_rule(::BarkerRule, log_acceptance_ratio)

Compute the acceptance probability according to the Barker rule.

# Arguments
- `log_acceptance_ratio`: The logarithm of the acceptance ratio

# Returns
- Acceptance probability (between 0 and 1)
"""
function acceptance_rule(::BarkerRule, log_acceptance_ratio::T) where {T<:AbstractFloat}
    return one(T) / (one(T) + exp(-log_acceptance_ratio))
end

"""
    mc_step!(system::AriannaSystem, action::Action, policy::Policy, parameters::AbstractArray{T}, rng) where {T<:AbstractFloat}

Perform a single Monte Carlo step.

# Arguments
- `system`: System to modify
- `action`: Action to perform
- `policy`: Policy used for proposal
- `parameters`: Parameters of the policy
- `rng`: Random number generator
# Keyword Arguments
- `rule`: Acceptance rule used for the algorithm (default: `MetropolisRule()`)
"""
function mc_step!(system::AriannaSystem, action::Action, policy::Policy, parameters::AbstractArray{T}, rng; rule::AriannaRule=MetropolisRule()) where {T<:AbstractFloat}
    sample_action!(action, policy, parameters, system, rng)
    logq_forward = log_proposal_density(action, policy, parameters, system)
    x₁, x₂ = perform_action!(system, action)
    Δlogp = delta_log_target_density(x₁, x₂, system)
    invert_action!(action, system)
    logq_backward = log_proposal_density(action, policy, parameters, system)
    log_acceptance_ratio = Δlogp + logq_backward - logq_forward
    α = acceptance_rule(rule, log_acceptance_ratio)
    if α > rand(rng)
        return 1
    else
        revert_action!(system, action)
        return 0
    end
end

"""
    mc_sweep!(system::AriannaSystem, pool, rng; mc_steps=1)

Perform a Monte Carlo sweep over a pool of moves.

# Arguments
- `system`: System to modify
- `pool`: Pool of moves to perform sweeps over
- `rng`: Random number generator
# Keyword Arguments
- `mc_steps`: Number of Monte Carlo steps per sweep
- `rule`: Acceptance rule used for the algorithm
"""
function mc_sweep!(system::AriannaSystem, pool, rng; mc_steps=1, rule::AriannaRule=MetropolisRule())
    weights = [move.weight for move in pool]
    for _ in 1:mc_steps
        id = rand(rng, Categorical(weights))
        move = pool[id]
        move.accepted_calls += mc_step!(system, move.action, move.policy, move.parameters, rng; rule=rule)
        move.total_calls += 1
    end
    return nothing
end

"""
    Metropolis{P,R<:AbstractRNG,C<:Function} <: AriannaAlgorithm

A struct representing a Metropolis Monte Carlo algorithm.

# Fields
- `pools::Vector{P}`: Vector of independent pools (one for each system)
- `sweepstep::Int`: Number of Monte Carlo steps per sweep
- `seed::Int`: Random number seed
- `rngs::Vector{R}`: Vector of random number generators
- `parallel::Bool`: Flag to parallelise over different threads
- `collecter::C`: Transducer to collect results from parallelised loops
- `rule::Ru`: Acceptance rule used for the algorithm

# Type Parameters
- `P`: Type of the pool
- `R`: Type of the random number generator
- `C`: Type of the transducer
- 'Ru`: Type of the Arianna rule used for acceptance
"""
struct Metropolis{P,R<:AbstractRNG,C<:Function,Ru<:AriannaRule} <: AriannaAlgorithm
    pools::Vector{P}            # Vector of independent pools (one for each system)
    sweepstep::Int              # Number of mc steps per mc sweep
    seed::Int                   # Random number seed
    rngs::Vector{R}             # Vector of random number generators
    parallel::Bool              # Flag to parallelise over different threads
    collecter::C                # Transducer to collect results from parallelised loops
    rule::Ru
    function Metropolis(
        chains::Vector{S},
        pools::Vector{P};
        sweepstep::Int=1,
        seed::Int=1,
        R::DataType=Xoshiro,
        parallel::Bool=false,
        rule::Ru=MetropolisRule()
    ) where {S<:AriannaSystem,P,Ru<:AriannaRule}
        # Safety checks
        @assert length(chains) == length(pools)
        @assert all(k -> all(move -> move.parameters == getindex.(pools, k)[1].parameters, getindex.(pools, k)), eachindex(pools[1]))
        @assert all(k -> all(move -> move.weight == getindex.(pools, k)[1].weight, getindex.(pools, k)), eachindex(pools[1]))
        @assert all(x -> length(x) == length(chains[1]), chains)
        #Make sure that all policies and parameters across chains refer to the same objects
        policy_list = [move.policy for move in pools[1]]
        parameters_list = [move.parameters for move in pools[1]]
        for pool in pools
            for k in eachindex(policy_list)
                pool[k].policy = policy_list[k]
                pool[k].parameters = parameters_list[k]
            end
        end
        # Handle randomness
        seeds = [seed + c - 1 for c in eachindex(chains)]
        rngs = [R(s) for s in seeds]
        # Handle parallelism
        collecter = parallel ? Transducers.tcollect : collect
        return new{P,R,typeof(collecter),Ru}(pools, sweepstep, seed, rngs, parallel, collecter, rule)
    end

end

"""
    Metropolis(chains; pool=missing, sweepstep=1, seed=1, R=Xoshiro, parallel=false, extras...)

Create a new Metropolis instance.

# Arguments
- `chains`: Vector of systems to run the algorithm on
- `pool`: Pool of moves to perform sweeps over
- `sweepstep`: Number of Monte Carlo steps per sweep
- `seed`: Random number seed
- `R`: Type of the random number generator
- `parallel`: Flag to parallelise over different threads
- `extras`: Additional keyword arguments

# Returns
- `algorithm`: Metropolis instance
"""
function Metropolis(chains; pool=missing, sweepstep=length(chains[1]), seed=1, R=Xoshiro, parallel=false, rule=MetropolisRule(), extras...)
    pools = [deepcopy(pool) for _ in chains]
    return Metropolis(chains, pools; sweepstep=sweepstep, seed=seed, R=R, parallel=parallel, rule=rule)
end

"""
    make_step!(simulation::Simulation, algorithm::Metropolis)

Perform a single step of the Metropolis algorithm.

# Arguments
- `simulation`: Simulation to perform the step on
- `algorithm`: Metropolis instance
"""
function make_step!(simulation::Simulation, algorithm::Metropolis)
    algorithm.collecter(
        eachindex(simulation.chains) |> Map(c -> begin
            mc_sweep!(simulation.chains[c], algorithm.pools[c], algorithm.rngs[c]; mc_steps=algorithm.sweepstep, algorithm.rule)
        end)
    )
    return nothing
end


"""
    write_parameters(::Policy, parameters)

Write the parameters of a policy to a string.

# Arguments
- `policy`: Policy to write the parameters of
- `parameters`: Parameters of the policy
"""
function write_parameters(::Policy, parameters)
    return "$(collect(vec(parameters)))"
end

"""
    write_algorithm(io, algorithm::Metropolis, scheduler)

Write the algorithm to a string.

# Arguments
- `io`: IO stream to write to
- `algorithm`: Algorithm to write
- `scheduler`: Scheduler to write
"""
function write_algorithm(io, algorithm::Metropolis, scheduler)
    println(io, "\tMetropolis")
    println(io, "\t\tCalls: $(length(filter(x -> 0 < x ≤ scheduler[end], scheduler)))")
    println(io, "\t\tMC steps per simulation step: $(algorithm.sweepstep)")
    println(io, "\t\tSeed: $(algorithm.seed)")
    println(io, "\t\tParallel: $(algorithm.parallel)")
    if algorithm.parallel
        println(io, "\t\tThreads: $(Threads.nthreads())")
    end
    println(io, "\t\tMoves:")
    for (k, move) in enumerate(algorithm.pools[1])
        println(io, "\t\t\tMove $(k):")
        println(io, "\t\t\t\tAction: " * replace(string(typeof(move.action)), r"\{.*" => ""))
        println(io, "\t\t\t\tPolicy: " * replace(string(typeof(move.policy)), r"\{.*" => ""))
        println(io, "\t\t\t\tParameters: " * write_parameters(move.policy, move.parameters))
        println(io, "\t\t\t\tWeight: $(move.weight)")
    end
end

"""
    StoreParameters{V<:AbstractArray} <: AriannaAlgorithm

A struct representing a parameter store.

# Fields
- `paths::Vector{String}`: Vector of paths to the parameter files
- `files::Vector{IOStream}`: Vector of open file streams to the parameter files
- `parameters_list::V`: List of parameters to store
- `store_first::Bool`: Flag to store the parameters at the first step
- `store_last::Bool`: Flag to store the parameters at the last step

# Type Parameters
- `V`: Type of the parameter array
"""
struct StoreParameters{V<:AbstractArray} <: AriannaAlgorithm
    paths::Vector{String}
    files::Vector{IOStream}
    parameters_list::V
    store_first::Bool
    store_last::Bool

    function StoreParameters(pool, path; ids=collect(eachindex(pool)), store_first::Bool=true, store_last::Bool=false)
        parameters_list = [move.parameters for move in pool[ids]]
        dirs = joinpath.(path, "parameters", ["$(k)" for k in ids])
        mkpath.(dirs)
        paths = joinpath.(dirs, "parameters.dat")
        files = Vector{IOStream}(undef, length(paths))
        try
            files = open.(paths, "w")
        finally
            close.(files)
        end
        return new{typeof(parameters_list)}(paths, files, parameters_list, store_first, store_last)
    end

end

"""
    StoreParameters(chains; dependencies=missing, path=missing, ids=missing, store_first=true, store_last=false, extras...)

Create a new StoreParameters instance.

# Arguments
- `chains`: Vector of chains to store the parameters of
- `dependencies`: Dependencies of the parameter store
- `path`: Path to the parameter files
- `ids`: IDs of the parameters to store
- `store_first`: Flag to store the parameters at the first step
- `store_last`: Flag to store the parameters at the last step
- `extras`: Additional keyword arguments

# Returns
- `algorithm`: StoreParameters instance
"""
function StoreParameters(chains; dependencies=missing, path=missing, ids=missing, store_first=true, store_last=false, extras...)
    @assert length(dependencies) == 1
    @assert length(dependencies) == 1
    @assert isa(dependencies[1], Metropolis)
    pool = dependencies[1].pools[1]
    if ismissing(ids)
        ids = collect(eachindex(pool))
    end
    return StoreParameters(pool, path; ids=ids, store_first=store_first, store_last=store_last)
end

function initialise(algorithm::StoreParameters, simulation::Simulation)
    simulation.verbose && println("Opening parameter files...")
    algorithm.files .= open.(algorithm.paths, "w")
    algorithm.store_first && make_step!(simulation, algorithm)
    return nothing
end

function make_step!(simulation::Simulation, algorithm::StoreParameters)
    for k in eachindex(algorithm.files)
        println(algorithm.files[k], "$(simulation.t) $(collect(algorithm.parameters_list[k]))")
        flush(algorithm.files[k])
    end
end

function finalise(algorithm::StoreParameters, simulation::Simulation)
    algorithm.store_last && make_step!(simulation, algorithm)
    simulation.verbose && println("Closing parameter files...")
    close.(algorithm.files)
    return nothing
end


"""
    StoreAcceptance <: AriannaAlgorithm

Algorithm to store the moving average acceptance rate of the simulation.
"""
struct StoreAcceptance <: AriannaAlgorithm
    path::String
    file::Vector{IOStream}

    function StoreAcceptance(path::String)
        file_path = joinpath(path, "acceptance.dat")
        return new(file_path, Vector{IOStream}(undef, 1))
    end
end

function StoreAcceptance(chains::AbstractVector; path=missing, extras...)
    return StoreAcceptance(path)
end

function initialise(algorithm::StoreAcceptance, simulation::Simulation)
    simulation.verbose && println("Opening acceptance file...")
    algorithm.file[1] = open(algorithm.path, "w")
    return nothing
end

function make_step!(simulation::Simulation, algorithm::StoreAcceptance)
    metropolis_algos = filter(x -> isa(x, Metropolis), simulation.algorithms)
    if isempty(metropolis_algos)
        return
    end

    all_pools = []
    for algo in metropolis_algos
        append!(all_pools, algo.pools)
    end

    if isempty(all_pools)
        return
    end

    pool_rates = [
        [move.total_calls > 0 ? move.accepted_calls / move.total_calls : 0.0 for move in pool]
        for pool in all_pools
    ]

    avg_rates = mean(pool_rates)

    print(algorithm.file[1], "$(simulation.t)")
    for val in avg_rates
        print(algorithm.file[1], " $val")
    end
    println(algorithm.file[1])
    flush(algorithm.file[1])
end

function finalise(algorithm::StoreAcceptance, simulation::Simulation)
    simulation.verbose && println("Closing acceptance file...")
    close(algorithm.file[1])
    return nothing
end

nothing