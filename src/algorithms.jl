"""
    abstract type AriannaAlgorithm

Abstract type for Simulation algorithms.
"""
abstract type AriannaAlgorithm end

"""
    initialise(::AriannaAlgorithm, ::Simulation)

Initialise the algorithm for the given simulation.
"""
initialise(::AriannaAlgorithm, ::Simulation) = return nothing

"""
    make_step!(::Simulation, ::AriannaAlgorithm)

Perform a single step of the algorithm in the simulation.
"""
make_step!(::Simulation, ::AriannaAlgorithm) = return nothing

"""
    finalise(::AriannaAlgorithm, ::Simulation)

Finalise the algorithm for the given simulation.
"""
finalise(::AriannaAlgorithm, ::Simulation) = return nothing

"""
    write_algorithm(io, algorithm::AriannaAlgorithm, scheduler)

Write a summary of the algorithm on the given IO stream.
"""
function write_algorithm(io, algorithm::AriannaAlgorithm, scheduler)
    println(io, "\t" * replace(string(typeof(algorithm)), r"\{.*" => ""))
    println(io, "\t\tCalls: $(length(filter(x -> 0 < x ≤ scheduler[end], scheduler)))")
end

"""
    StoreCallbacks{V} <: AriannaAlgorithm

Algorithm to store callback values during simulation.

# Fields
- `callbacks::V`: Vector of callback functions to evaluate
- `paths::Vector{String}`: Paths to output files for each callback
- `files::Vector{IOStream}`: File handles for writing callback values
- `store_first::Bool`: Whether to store callback values at initialization
- `store_last::Bool`: Whether to store callback values at finalization

# Constructor
    StoreCallbacks(callbacks::V, path; store_first::Bool=true, store_last::Bool=false)

Create a new StoreCallbacks instance.

# Arguments
- `callbacks::V`: Vector of callback functions
- `path`: Base path for output files
- `store_first=true`: Store values at initialization
- `store_last=false`: Store values at finalization
"""
struct StoreCallbacks{V} <: AriannaAlgorithm
    callbacks::V
    paths::Vector{String}
    files::Vector{IOStream}
    store_first::Bool
    store_last::Bool

    function StoreCallbacks(callbacks::V, path; store_first::Bool=true, store_last::Bool=false) where {V}
        mkpath(path)
        paths = joinpath.(path, [replace(string(cb), "callback_" => "") * ".dat" for cb in callbacks])
        files = Vector{IOStream}(undef, length(paths))
        try
            files = open.(paths, "w")
        finally
            close.(files)
        end
        return new{V}(callbacks, paths, files, store_first, store_last)
    end

end

function StoreCallbacks(chains; path=missing, callbacks=missing, store_first=true, store_last=false, extras...)
    if ismissing(callbacks)
        callbacks = []
    end
    return StoreCallbacks(callbacks, path; store_first=store_first, store_last=store_last)
end

function initialise(algorithm::StoreCallbacks, simulation::Simulation)
    simulation.verbose && println("Opening callback files...")
    algorithm.files .= open.(algorithm.paths, "w")
    algorithm.store_first && make_step!(simulation, algorithm)
    return nothing
end

function make_step!(simulation::Simulation, algorithm::StoreCallbacks)
    for (callback, file) in zip(algorithm.callbacks, algorithm.files)
        println(file, "$(simulation.t) $(callback(simulation))")
        flush(file)
    end
end

function finalise(algorithm::StoreCallbacks, simulation::Simulation)
    algorithm.store_last && make_step!(simulation, algorithm)
    simulation.verbose && println("Closing callback files...")
    close.(algorithm.files)
    return nothing
end

"""
    Format

Abstract type for output file formats.
"""
abstract type Format end

"""
    TXT <: Format

Format type for text (.txt) file output.
"""
struct TXT <: Format
    extension::String
    function TXT()
        return new(".txt")
    end
end

"""
    DAT <: Format

Format type for data (.dat) file output.
"""
struct DAT <: Format
    extension::String
    function DAT()
        return new(".dat")
    end
end

"""
    StoreTrajectories{F<:Format} <: AriannaAlgorithm

Algorithm to store system trajectories during simulation.

# Fields
- `paths::Vector{String}`: Paths to output files
- `files::Vector{IOStream}`: File handles for writing trajectories
- `fmt::F`: Format type for output files
- `store_first::Bool`: Whether to store trajectories at initialization
- `store_last::Bool`: Whether to store trajectories at finalization
"""
struct StoreTrajectories{F<:Format} <: AriannaAlgorithm
    paths::Vector{String}
    files::Vector{IOStream}
    fmt::F
    store_first::Bool
    store_last::Bool

    function StoreTrajectories(chains, path, fmt; store_first::Bool=true, store_last::Bool=false)
        dirs = joinpath.(path, "trajectories", ["$c" for c in eachindex(chains)])
        mkpath.(dirs)
        ext = fmt.extension
        paths = joinpath.(dirs, "trajectory$ext")
        files = Vector{IOStream}(undef, length(paths))
        try
            files = open.(paths, "w")
        finally
            close.(files)
        end
        return new{typeof(fmt)}(paths, files, fmt, store_first, store_last)
    end

end

function StoreTrajectories(chains; path=missing, fmt=DAT(), store_first=true, store_last=false, extras...)
    return StoreTrajectories(chains, path, fmt, store_first=store_first, store_last=store_last)
end

"""
    store_trajectory(io, system::AriannaSystem, t, fmt::Format)

Store the system trajectory at time t to the given IO stream in the specified format.
"""
function store_trajectory(io, system::AriannaSystem, t, fmt::Format)
    println(io, "$t, $system")
    return nothing
end

function initialise(algorithm::StoreTrajectories, simulation::Simulation)
    simulation.verbose && println("Opening trajectory files...")
    algorithm.files .= open.(algorithm.paths, "w")
    algorithm.store_first && make_step!(simulation, algorithm)
    return nothing
end

function make_step!(simulation::Simulation, algorithm::StoreTrajectories)
    for c in eachindex(simulation.chains)
        store_trajectory(algorithm.files[c], simulation.chains[c], simulation.t, algorithm.fmt)
        flush(algorithm.files[c])
    end
end

function finalise(algorithm::StoreTrajectories, simulation::Simulation)
    algorithm.store_last && make_step!(simulation, algorithm)
    simulation.verbose && println("Closing trajectory files...")
    close.(algorithm.files)
    return nothing
end

"""
    StoreLastFrames <: AriannaAlgorithm

Algorithm to store the final state of each system at the end of simulation.

# Fields
- `paths::Vector{String}`: Paths to output files
- `fmt::Format`: Format type for output files
"""
struct StoreLastFrames <: AriannaAlgorithm
    paths::Vector{String}
    fmt::Format
    function StoreLastFrames(chains, path, fmt)
        dirs = joinpath.(path, "trajectories", ["$c" for c in eachindex(chains)])
        mkpath.(dirs)
        ext = fmt.extension
        paths = joinpath.(dirs, "lastframe$ext")
        return new(paths, fmt)
    end

end

function StoreLastFrames(chains; path=missing, fmt=DAT(), extras...)
    return StoreLastFrames(chains, path, fmt)
end

store_lastframe(io, system::AriannaSystem, t, fmt::Format) = store_trajectory(io, system, t, fmt)
"""
    store_lastframe(io, system, t, fmt::Format)

Store the final state of the system at time t to the given IO stream in the specified format.
"""
function finalise(algorithm::StoreLastFrames, simulation::Simulation)
    for c in eachindex(simulation.chains)
        open(algorithm.paths[c], "w") do file
            store_lastframe(file, simulation.chains[c], simulation.t, algorithm.fmt)
        end
    end
    return nothing
end

"""
    StoreBackups <: AriannaAlgorithm

Algorithm to create backup files of system states during simulation.

# Fields
- `dirs::Vector{String}`: Directories for storing backup files
- `fmt::Format`: Format type for output files
- `store_first::Bool`: Whether to store backups at initialization
- `store_last::Bool`: Whether to store backups at finalization
"""
struct StoreBackups <: AriannaAlgorithm
    dirs::Vector{String}
    fmt::Format
    store_first::Bool
    store_last::Bool
    function StoreBackups(chains, path, fmt; store_first::Bool=false, store_last::Bool=false)
        dirs = joinpath.(path, "trajectories", ["$c" for c in eachindex(chains)])
        mkpath.(dirs)
        return new(dirs, fmt, store_first, store_last)
    end

end

function StoreBackups(chains; path=missing, fmt=DAT(), store_first=false, store_last=false, extras...)
    return StoreBackups(chains, path, fmt, store_first=store_first, store_last=store_last)
end

store_backup(io, system::AriannaSystem, t, fmt::Format) = store_trajectory(io, system, t, fmt)
"""
    store_backup(io, system, t, fmt::Format)

Store a backup of the system state at time t to the given IO stream in the specified format.
"""
function initialise(algorithm::StoreBackups, simulation::Simulation)
    algorithm.store_first && make_step!(simulation, algorithm)
    return nothing
end

function make_step!(simulation::Simulation, algorithm::StoreBackups)
    for c in eachindex(simulation.chains)
        open(joinpath(algorithm.dirs[c], "restart_t$(simulation.t)$(algorithm.fmt.extension)"), "w") do file
            store_backup(file, simulation.chains[c], simulation.t, algorithm.fmt)
        end
    end
end

function finalise(algorithm::StoreBackups, simulation::Simulation)
    algorithm.store_last && make_step!(simulation, algorithm)
    return nothing
end

"""
    PrintTimeSteps <: AriannaAlgorithm

Algorithm to display a progress bar and current timestep during simulation.
"""
struct PrintTimeSteps <: AriannaAlgorithm end

function PrintTimeSteps(chains; extras...)
    return PrintTimeSteps()
end

function make_step!(simulation::Simulation, ::PrintTimeSteps)
    t = simulation.t
    percent = t / simulation.steps
    bar_length = 50
    filled_length = Int(round(percent * bar_length))
    bar = "\033[1;34m" * repeat("■", filled_length) * "\033[0m" * repeat("□", bar_length - filled_length)
    @printf("\rProgress: [%s] %.0f%% t = %d", bar, percent * 100, t)
end

nothing