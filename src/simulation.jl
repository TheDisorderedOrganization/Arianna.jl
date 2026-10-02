"""
    mutable struct Simulation{S, A, VS}

A structure representing a Monte Carlo simulation.

# Fields
- `chains::Vector{S}`: Vector of independent Arianna systems.
- `algorithms::A`: List of algorithms.
- `steps::Int`: Number of MC sweeps.
- `t::Int`: Current time step.
- `t_start::Int`: time step to start the simulation from.
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
    t_start::Int
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
    - `t_start::Int=0`: starting time of the simulation
    - `path::String="data"`: Simulation path.
    - `verbose::Bool=false`: Flag for verbose output.
    """
    function Simulation(
        chains::Vector{S},
        algorithms::A,
        schedulers::VS,
        steps::Int;
        t_start::Int=0,
        path::String="data",
        verbose::Bool=false
    ) where {S<:AriannaSystem,A,VS}
        @assert length(schedulers) == length(algorithms)
        @assert all(scheduler -> all(x -> 0 ≤ x ≤ steps, scheduler), schedulers)
        @assert all(scheduler -> issorted(scheduler), schedulers)
        t = t_start # initial time for the simulation 0 by default
        counters = [something(findfirst(x -> x > t_start, scheduler),1) for scheduler in schedulers]
        mkpath(path)
        return new{S,A,VS}(chains, algorithms, steps, t, t_start, schedulers, counters, path, verbose)
    end

end

"""
    Simulation(chains, algorithm_list, steps; t_start=0 path="data", verbose=false)

Create a new `Simulation` instance from a list of algorithm constructors.

# Arguments
- `chains`: Vector of independent Arianna systems.
- `algorithm_list`: List of algorithm constructors.
- `steps`: Number of MC sweeps.
- `t_start=0`: initial time for the simulation.
- `path="data"`: Simulation path.
- `verbose=false`: Flag for verbose output.
"""
function Simulation(chains, algorithm_list, steps; t_start=0, path="data", verbose=false)
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
    return Simulation(chains, algorithms, schedulers, steps; t_start=t_start, path=path, verbose=verbose)
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

"""
    build_schedule(steps::Int, tw::Int, N::Int; burn::Int=0)

Create a vector of timestep from `burn` to `steps` with a log spaced scheme repeated every tw.
"""

struct MultiOrigins
    tw::Int # decorrelation time
    N::Int # number of point from burn to steps
end

function build_schedule(steps::Int, params::MultiOrigins; burn::Int=0)
    tw = params.tw
    N = params.N

    @assert 0 <= burn < steps "burn=$burn must be >= 0 and < steps=$steps"

    tau_targets = unique([round(Int, exp(i * log(steps) / (N-1))) for i in 0:N-1])
    tau_targets[end] = steps

    ntw     = round(Int, (steps-burn) / tw)
    origins = [burn + k * tw for k in 0:ntw]

    sched_set = Set{Int}()
    for tw_i in origins
        push!(sched_set, tw_i)
        for tau_j in tau_targets
            dum = tw_i + tau_j
            if dum <= steps
                push!(sched_set, dum)
            end
        end
    end

    sched = filter(x -> (x >= burn) && (x <= steps), collect(sched_set))
    return sort!(sched)
end

function write_system(io, system::AriannaSystem)
    println(io, "\t" * "$(typeof(system))")
    return nothing
end

function write_summary(simulation)
    writing_mode = simulation.t_start > 0 ? "a" : "w"
    open(joinpath(simulation.path, "summary.log"), writing_mode) do file
        println(file, "SIMULATION SUMMARY")
        println(file)
        println(file, "Simulation:")
        println(file, "\tSteps: $(simulation.steps)")
        println(file, "\tInitial time: $(simulation.t_start)")
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
    run!(simulation::Simulation; wall_time::Real=Inf)

Run the Monte Carlo simulation.

# Arguments
- `simulation::Simulation`: The simulation instance to run.
- `wall_time::Real=Inf`: to set a wall_time (in seconds) that trigger a restart by default wall_time is Inf
"""
function run!(simulation::Simulation; wall_time::Real=Inf)
    t0 = time()         # initial measure of time in seconds
    status = :running # default 
    try
        simulation.verbose && println("\n" * "-"^50)
        simulation.verbose && println("\033[1;32mINITIALISATION\033[0m")
        for algorithm in simulation.algorithms
            initialise(algorithm, simulation)
        end
        write_summary(simulation)
        simulation.verbose && println("\033[1;32m\nRUNNING SIMULATION...\033[0m")
        sim_time = @elapsed for simulation.t in (simulation.t_start+1):simulation.steps
            for k in eachindex(simulation.algorithms)
                if simulation.t == simulation.schedulers[k][simulation.counters[k]]
                    make_step!(simulation, simulation.algorithms[k])
                    if simulation.counters[k] < length(simulation.schedulers[k]) # pin the counter at the last entry to prevent out of bound situations
                        simulation.counters[k] += 1
                    end
                end
            end
            if time()-t0 ≥ wall_time
                status = :need_restart # switch to need_restart if wall_time is hit by the simulation
                break # exit the `for simulation.t` loop
            end
        end
        if status == :running
            status = :completed
        end
        simulation.verbose && println("\nSimulation $(status == :completed ? "completed" : "stopped for restart") in $(sim_time) s")
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
    return status
end

nothing
