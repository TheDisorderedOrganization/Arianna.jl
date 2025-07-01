"""
    mutable struct Simulation{S, A, VS}

A structure representing a Monte Carlo simulation.

# Fields
- `chains::Vector{S}`: Vector of independent Arianna systems.
- `algorithms::A`: List of algorithms.
- `steps::Int`: Number of MC sweeps.
- `t::Int`: Current time step.
- `schedulers::VS`: List of schedulers (one for each algorithm).
- `counters::Vector{Int}`: Counters for the schedulers (one for each algorithm).
- `path::String`: Simulation path.
- `verbose::Bool`: Flag for verbose output.
"""
mutable struct Simulation{S,A,VS}
    chains::Vector{S}
    algorithms::A
    steps::Int
    t::Int
    schedulers::VS
    counters::Vector{Int}
    path::String
    verbose::Bool

    """
    Create a new `Simulation` instance.

    # Arguments
    - `chains::Vector{S}`: Vector of independent Arianna systems.
    - `algorithms::A`: List of algorithms.
    - `schedulers::VS`: List of schedulers (one for each algorithm).
    - `steps::Int`: Number of MC sweeps.
    - `path::String="data"`: Simulation path.
    - `verbose::Bool=false`: Flag for verbose output.
    """
    function Simulation(
        chains::Vector{S},
        algorithms::A,
        schedulers::VS,
        steps::Int;
        path::String="data",
        verbose::Bool=false
    ) where {S<:AriannaSystem,A,VS}
        @assert length(schedulers) == length(algorithms)
        @assert all(scheduler -> all(x -> 0 ≤ x ≤ steps, scheduler), schedulers)
        @assert all(scheduler -> issorted(scheduler), schedulers)
        t = 0
        counters = [findfirst(x -> x > 0, scheduler) for scheduler in schedulers]
        mkpath(path)
        return new{S,A,VS}(chains, algorithms, steps, t, schedulers, counters, path, verbose)
    end

end

"""
    Simulation(chains, algorithm_list, steps; path="data", verbose=false)

Create a new `Simulation` instance from a list of algorithm constructors.

# Arguments
- `chains`: Vector of independent Arianna systems.
- `algorithm_list`: List of algorithm constructors.
- `steps`: Number of MC sweeps.
- `path="data"`: Simulation path.
- `verbose=false`: Flag for verbose output.
"""
function Simulation(chains, algorithm_list, steps; path="data", verbose=false)
    schedulers_tmp = []
    algorithms_tmp = []
    algorithm_names = []
    for constructor in algorithm_list
        push!(algorithm_names, constructor.algorithm)
        scheduler = haskey(constructor, :scheduler) ? constructor.scheduler : 1:steps
        push!(schedulers_tmp, scheduler)
        kwargs = Base.structdiff(constructor, (algorithm=nothing, scheduler=nothing, dependencies=nothing))
        if haskey(constructor, :dependencies)
            parent_ids = findall(in(constructor.dependencies), algorithm_names)
            parent_instances = algorithms_tmp[parent_ids]
            kwargs = merge(kwargs, (dependencies=parent_instances,))
        end
        kwargs = merge(kwargs, (path=path, steps=steps, verbose=verbose))
        push!(algorithms_tmp, constructor.algorithm(chains; kwargs...))
    end
    schedulers = ntuple(k -> schedulers_tmp[k], length(schedulers_tmp))
    algorithms = ntuple(k -> algorithms_tmp[k], length(algorithms_tmp))
    return Simulation(chains, algorithms, schedulers, steps; path=path, verbose=verbose)
end

"""
    build_schedule(steps::Int, burn::Int, Δt::Int)

Create a vector of timestep from `burn` to `steps` at intervals `Δt`.
"""
function build_schedule(steps::Int, burn::Int, Δt::Int)
    return collect(burn:Δt:steps) ∪ [steps]
end

"""
    build_schedule(steps::Int, burn::Int, base::AbstractFloat)

Create a vector of timestep from `burn` to `steps` log-spaced with base `base`.
"""
function build_schedule(steps::Int, burn::Int, base::AbstractFloat)
    return unique(vcat([burn], [burn + Int(base^n) for n in 0:floor(Int, log(base, steps - burn))], [steps]))
end

"""
    build_schedule(steps::Int, burn::Int, block::Vector{Int})

Create a vector of timestep from `burn` to `steps` with repeated blocks specified by `block`.
"""
function build_schedule(steps::Int, burn::Int, block::Vector{Int})
    nblock = (steps - burn) ÷ block[end]
    blocks = [block .+ burn .+ (m - 1) * block[end] for m in 1:nblock]
    return filter(x -> x ≤ steps, unique(vcat(blocks..., [steps])))
end

function write_system(io, system::AriannaSystem)
    println(io, "\t" * "$(typeof(system))")
    return nothing
end

function write_summary(simulation)
    open(joinpath(simulation.path, "summary.log"), "w") do file
        println(file, "SIMULATION SUMMARY")
        println(file)
        println(file, "Simulation:")
        println(file, "\tSteps: $(simulation.steps)")
        println(file, "\tNumber of chains: $(length(simulation.chains))")
        println(file, "\tNumber of algorithms: $(length(simulation.algorithms))")
        println(file, "\tVerbose: $(simulation.verbose)")
        println(file, "\tStarted on $(now())")
        println(file)
        println(file, "System:")
        write_system(file, simulation.chains[1])
        println(file)
        println(file, "Algorithms:")
        for (algorithm, scheduler) in zip(simulation.algorithms, simulation.schedulers)
            write_algorithm(file, algorithm, scheduler)
        end
        println(file)
    end
end

function update_summary(simulation, sim_time)
    open(joinpath(simulation.path, "summary.log"), "a") do file
        println(file, "Report:")
        println(file, "\tSimulation time: $(sim_time) s")
    end
end

function finalise_summary(simulation)
    open(joinpath(simulation.path, "summary.log"), "a") do file
        total_size = 0
        for (root, dirs, files) in walkdir(simulation.path)
            for file in files
                total_size += filesize(joinpath(root, file))
            end
        end
        sim_size = total_size / 1024^2
        println(file, "\tSimulation size: $(sim_size) MB")
        println(file, "\tStatus: Completed on $(now())")
    end
end

"""
    run!(simulation::Simulation)

Run the Monte Carlo simulation.

# Arguments
- `simulation::Simulation`: The simulation instance to run.
"""
function run!(simulation::Simulation)
    try
        simulation.verbose && println("\n" * "-"^50)
        simulation.verbose && println("\033[1;32mINITIALISATION\033[0m")
        for algorithm in simulation.algorithms
            initialise(algorithm, simulation)
        end
        write_summary(simulation)
        simulation.verbose && println("\033[1;32m\nRUNNING SIMULATION...\033[0m")
        sim_time = @elapsed for simulation.t in 1:simulation.steps
            for k in eachindex(simulation.algorithms)
                if simulation.t == simulation.schedulers[k][simulation.counters[k]]
                    make_step!(simulation, simulation.algorithms[k])
                    simulation.counters[k] += 1
                end
            end
        end
        simulation.verbose && println("\nSimulation completed in $(sim_time) s")
        update_summary(simulation, sim_time)
    finally
        simulation.verbose && println("\033[1;32m\nFINALISATION\033[0m")
        for algorithm in simulation.algorithms
            finalise(algorithm, simulation)
        end
        finalise_summary(simulation)
        simulation.verbose && println("\033[1;32m\nDONE\033[0m")
        simulation.verbose && println("-"^50 * "\n")
    end
    return nothing
end

nothing
