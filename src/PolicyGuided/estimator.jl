"""
    PolicyGradientEstimator{P,O,VPL,VPR,VO,VG,VC,VL,ADB,R,C} <: AlgoritAriannaAlgorithm

A struct representing a policy gradient estimator for Monte Carlo algorithms.

# Fields
- `pools::Vector{P}`: Vector of independent pools (one for each system)
- `optimisers::O`: List of optimisers (one for each move)
- `learn_ids::Vector{Int}`: List of learnable moves
- `q_batch_size::Int`: Number of independent samples generated from proposal distributions
- `policy_list::VPL`: List of policies (one for each move)
- `parameters_list::VPR`: List of current parameters values (one array for each move)
- `objectives::VO`: Cache for estimated objective functions Ĵ (one for each move)
- `gradients_data::VG`: Gradient information (one for each move)
- `chains_shadow::VC`: Copy of chains (for Enzyme)
- `∇logqs_forward::VL`: Preallocated forward gradients (one array for each move)
- `∇logqs_backward::VL`: Preallocated backward gradients (one array for each move)
- `ad_backend::ADB`: Backend for automatic differentiation (Enzyme or Zygote)
- `seed::Int`: Random number seed
- `rngs::Vector{R}`: Vector of random number generators
- `parallel::Bool`: Flag to parallelise over different threads
- `reducer::C`: Transducer to reduce results from parallelised loops

# Type Parameters
- `P`: Type of the pool
- `O`: Type of the optimisers
- `VPL`: Type of the policy list
- `VPR`: Type of the parameter list
- `VO`: Type of the objectives array
- `VG`: Type of the gradients data
- `VC`: Type of the chains shadow
- `VL`: Type of the gradient arrays
- `ADB`: Type of the automatic differentiation backend
- `R`: Type of the random number generator
- `C`: Type of the reducer function
"""

struct PolicyGradientEstimator{P,O,VPL<:AbstractArray,VPR<:AbstractArray,VO<:AbstractArray,VG<:AbstractArray,VC<:AbstractArray,VL<:AbstractArray,ADB<:AD_Backend,R<:AbstractRNG,C<:Function} <: AriannaAlgorithm
    pools::Vector{P}            # Vector of independent pools (one for each system)
    optimisers::O               # List of optimisers (one for each move)
    learn_ids::Vector{Int}      # List of learnable moves
    q_batch_size::Int           # Number of independent samples generated from proposal distributions
    policy_list::VPL            # List of policies (one for each move)
    parameters_list::VPR        # List of current parameters values (one array for each move)
    objectives::VO              # Cache for estimated objective functions Ĵ (one for each move)
    gradients_data::VG          # Gradient information (one for each move)
    chains_shadow::VC           # Copy of chains (for Enzyme)
    ∇logqs_forward::VL          # Preallocated forward gradients (one array for each move)
    ∇logqs_backward::VL         # Preallocated backward gradients (one array for each move)
    ad_backend::ADB             # Backend for automatic differentiation (Enzyme or Zygote)
    seed::Int                   # Random number seed
    rngs::Vector{R}             # Vector of random number generators
    parallel::Bool              # Flag to parallelise over different threads
    reducer::C                  # Transducer to reduce results from parallelised loops

    function PolicyGradientEstimator(
        chains::Vector{S},
        pools::Vector{P},
        optimisers::O;
        q_batch_size::Int=1,
        ad_backend::AD_Backend=ForwardDiff_Backend(),
        seed::Int=1,
        R::DataType=Xoshiro,
        parallel::Bool=false
    ) where {S,P,O}
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
        # Create objectives and gradients caches
        objectives = zeros(eltype(pools[1][1].parameters), length(learn_ids))
        gradients_data = map(k -> initialise_gradient_data(parameters_list[k]), learn_ids)
        # Create shadows for Enzyme
        chains_shadow = deepcopy(chains)
        # Initialise gradients
        ∇logqs_forward = map(zero, parameters_list)
        ∇logqs_backward = map(zero, parameters_list)
        # Handle randomness
        seeds = [seed + c - 1 for c in eachindex(chains)]
        rngs = [R(s) for s in seeds]
        # Handle parallelism
        reducer = parallel ? Transducers.foldxt : Transducers.foldxl
        return new{P,O,typeof(policy_list),typeof(parameters_list),typeof(objectives),typeof(gradients_data),typeof(chains_shadow),typeof(∇logqs_forward),typeof(ad_backend),R,typeof(reducer)}(
            pools, optimisers, learn_ids, q_batch_size, policy_list, parameters_list, objectives, gradients_data,
            chains_shadow, ∇logqs_forward, ∇logqs_backward, ad_backend, seed, rngs, parallel, reducer
        )
    end

end

function PolicyGradientEstimator(chains; dependencies=missing, optimisers=missing, q_batch_size=1, ad_backend=ForwardDiff_Backend(), R=Xoshiro, parallel=false, extras...)
    @assert length(dependencies) == 1
    @assert isa(dependencies[1], Metropolis)
    metropolis = dependencies[1]
    pools, seed = metropolis.pools, metropolis.seed
    return PolicyGradientEstimator(chains, pools, optimisers; q_batch_size=q_batch_size, ad_backend=ad_backend, seed=seed, R=R, parallel=parallel)
end

function make_step!(simulation::Simulation, algorithm::PolicyGradientEstimator)
    for (k, lid) in enumerate(algorithm.learn_ids)
        gd = algorithm.reducer(+,
            eachindex(simulation.chains) |> Map(c -> begin
                1:algorithm.q_batch_size |> Map(_ -> begin
                    sample_gradient_data(
                        algorithm.pools[c][lid].action,
                        algorithm.policy_list[lid],
                        algorithm.parameters_list[lid],
                        simulation.chains[c],
                        algorithm.rngs[c];
                        ∇logq_forward=algorithm.∇logqs_forward[lid],
                        ∇logq_backward=algorithm.∇logqs_backward[lid],
                        shadow=algorithm.chains_shadow[c],
                        ad_backend=algorithm.ad_backend
                    )
                end)
            end) |> Cat()
        )
        algorithm.gradients_data[k] = algorithm.gradients_data[k] + gd
        algorithm.objectives[k] = algorithm.gradients_data[k].j / algorithm.gradients_data[k].n
    end
    return nothing
end

function write_algorithm(io, algorithm::PolicyGradientEstimator, scheduler)
    println(io, "\tPolicyGradientEstimator")
    println(io, "\t\tCalls: $(length(filter(x -> 0 < x ≤ scheduler[end], scheduler)))")
    println(io, "\t\tLearnable moves: $(algorithm.learn_ids)")
    println(io, "\t\tQ batch size: $(algorithm.q_batch_size)")
    println(io, "\t\tAD backend: $(last(split(string(typeof(algorithm.ad_backend)), ".")))")
    println(io, "\t\tSeed: $(algorithm.seed)")
    println(io, "\t\tParallel: $(algorithm.parallel)")
    if algorithm.parallel
        println(io, "\t\tThreads: $(Threads.nthreads())")
    end
end



"""
    StoreObjective{V<:AbstractArray} <: AriannaAlgorithm

A struct representing an objective function store.

# Fields
- `paths::Vector{String}`: Vector of paths to the objective files
- `files::Vector{IOStream}`: Vector of open file streams to the objective files
- `objectives::V`: List of objectives to store
- `store_first::Bool`: Flag to store the objectives at the first step
- `store_last::Bool`: Flag to store the objectives at the last step

# Type Parameters
- `V`: Type of the objective array
"""
struct StoreObjective{V<:AbstractArray} <: AriannaAlgorithm
    paths::Vector{String}
    files::Vector{IOStream}
    objectives::V
    store_first::Bool
    store_last::Bool

    function StoreObjective(estimator::PolicyGradientEstimator, path; store_first::Bool=true, store_last::Bool=false)
        objectives = estimator.objectives
        ids = estimator.learn_ids
        dirs = joinpath.(path, "moves", ["$(k)" for k in ids])
        mkpath.(dirs)
        paths = joinpath.(dirs, "objective.dat")
        files = Vector{IOStream}(undef, length(paths))
        try
            files = open.(paths, "w")
        finally
            close.(files)
        end
        return new{typeof(objectives)}(paths, files, objectives, store_first, store_last)
    end
end

"""
    StoreObjective(chains; dependencies=missing, path=missing, store_first=true, store_last=false, extras...)

Create a new StoreObjective instance.

# Arguments
- `chains`: Vector of chains (unused but kept for API consistency)
- `dependencies`: Dependencies of the objective store (must contain PolicyGradientEstimator)
- `path`: Path to the objective files
- `store_first`: Flag to store the objectives at the first step
- `store_last`: Flag to store the objectives at the last step
- `extras`: Additional keyword arguments

# Returns
- `algorithm`: StoreObjective instance
"""
function StoreObjective(chains; dependencies=missing, path=missing, store_first=true, store_last=false, extras...)
    @assert length(dependencies) == 1
    @assert isa(dependencies[1], PolicyGradientEstimator)
    estimator = dependencies[1]
    return StoreObjective(estimator, path; store_first=store_first, store_last=store_last)
end

function initialise(algorithm::StoreObjective, simulation::Simulation)
    simulation.verbose && println("Opening objective files...")
    algorithm.files .= open.(algorithm.paths, "w")
    algorithm.store_first && make_step!(simulation, algorithm)
    return nothing
end

function make_step!(simulation::Simulation, algorithm::StoreObjective)
    for k in eachindex(algorithm.files)
        println(algorithm.files[k], "$(simulation.t) $(algorithm.objectives[k])")
        flush(algorithm.files[k])
    end
end

function finalise(algorithm::StoreObjective, simulation::Simulation)
    algorithm.store_last && make_step!(simulation, algorithm)
    simulation.verbose && println("Closing objective files...")
    close.(algorithm.files)
    return nothing
end

nothing