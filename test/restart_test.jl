using Arianna
using Test
using DelimitedFiles

# particle_1d.jl defines the `Particle` system, `Displacement`, `StandardGaussian`
# and `energy`, but it calls `potential(x)` without defining it — the includer must.
include("../examples/particle_1d/particle_1d.jl")

potential(x) = x^2

# Helper: build a harmonic-oscillator simulation with `steps` sweeps.
# `t_start` lets us build a *resumed* simulation; `sampletimes` is an explicit
# schedule so we can check which points get skipped on resume.
function make_simulation(steps; path, M=2, seed=42, t_start=0, sampletimes=[steps], store_first=true)
    rng = Xoshiro(seed)
    β = 2.0
    chains = [System(4rand(rng) - 2, β) for _ in 1:M]
    pool = (Move(Displacement(0.0), StandardGaussian(), ComponentArray(σ=0.1), 1.0),)
    algorithm_list = (
        (algorithm=Metropolis, pool=pool, seed=seed, parallel=false),
        (algorithm=StoreCallbacks, callbacks=(energy,), scheduler=sampletimes, store_first=store_first),
        (algorithm=StoreLastFrames, scheduler=[steps]),
    )
    return Simulation(chains, algorithm_list, steps; t_start=t_start, path=path, verbose=false)
end

@testset "run! stops early when the wall_time budget is exceeded" begin
    steps = 10^6                     # large: a full run would take a while
    path = "data/restart_test/early_stop"
    simulation = make_simulation(steps; path=path)

    # wall_time = 0.0 makes this deterministic: the check `time() - t0 ≥ wall_time`
    # is already true after the very first *complete* step, so it always breaks at t = 1.
    # (No sleep, no reliance on how fast the machine is → not a flaky test.)
    status = run!(simulation; wall_time=0.0)

    @test status == :need_restart          # it reported *why* it stopped
    @test 1 ≤ simulation.t < steps          # a complete step ran, but not the whole run

    # The `finally` block must still run on the way out → the checkpoint frame exists.
    @test isfile(joinpath(path, "chains", "1", "lastframe.dat"))
end

@testset "run! completes and reports :completed when there is no time limit" begin
    steps = 1000                     # small: runs to the end quickly
    path = "data/restart_test/complete"
    simulation = make_simulation(steps; path=path)

    status = run!(simulation)               # default wall_time = Inf → never times out

    @test status == :completed
    @test simulation.t == steps             # the loop reached the last step
end

@testset "run! resumes from t_start and skips already-done schedule points" begin
    steps = 1000
    t_start = 500
    sampletimes = collect(100:100:1000)     # [100, 200, …, 1000]
    path = "data/restart_test/resume"
    simulation = make_simulation(steps; path=path, t_start=t_start, sampletimes=sampletimes)

    # The constructor should have placed us at t_start.
    @test simulation.t == t_start
    @test simulation.t_start == t_start

    # Each scheduler's counter should point at the FIRST time strictly after
    # t_start, i.e. everything ≤ t_start is skipped (already done before restart).
    for k in eachindex(simulation.schedulers)
        c = simulation.counters[k]
        sched = simulation.schedulers[k]
        @test sched[c] > t_start                 # next scheduled point is after t_start
        c > 1 && @test sched[c-1] ≤ t_start      # and the one before it was ≤ t_start
    end

    # A resumed run still finishes cleanly and reaches the end.
    status = run!(simulation)
    @test status == :completed
    @test simulation.t == steps
end

@testset "restart appends to data files instead of truncating them" begin
    path = "data/restart_test/append"
    steps_full = 1000
    sampletimes_full = collect(100:100:steps_full)          # [100, …, 1000]
    t_stop = 500
    energy_path = joinpath(path, "chains", "1", "energy.dat")

    # Segment A — fresh first job, with store_first=true (the DEFAULT). A fresh
    # run records the initial config at t=0, then samples the schedule ≤ t_stop.
    sampletimes_A = filter(t -> t ≤ t_stop, sampletimes_full)   # [100, …, 500]
    simA = make_simulation(t_stop; path=path, t_start=0, sampletimes=sampletimes_A)
    run!(simA)
    times_A = Int.(readdlm(energy_path)[:, 1])
    @test times_A == vcat(0, sampletimes_A)                 # t=0 initial frame + 100…500

    # Segment B — the restart job: full schedule, resuming at t_stop, SAME path.
    # t_start > 0 ⇒ initialise opens energy.dat in APPEND mode AND must NOT re-fire
    # store_first (that would duplicate the seam). (Reloading the configuration is
    # the downstream package's job; here we only test Arianna's file behaviour.)
    simB = make_simulation(steps_full; path=path, t_start=t_stop, sampletimes=sampletimes_full)
    run!(simB)
    times_all = Int.(readdlm(energy_path)[:, 1])

    @test times_all == vcat(0, sampletimes_full)  # A survived + B appended, single t=0
    @test issorted(times_all)                     # increasing across the seam
    @test allunique(times_all)                    # no duplicate (no second t=0, no second t=500)
end

@testset "run! survives a schedule that ends before steps (no BoundsError)" begin
    steps = 500
    # This callback schedule stops at 300, well before steps=500. After t=300 its
    # counter runs past the last entry; without the guard in run!, indexing
    # schedulers[k][counters[k]] at t=301 would throw a BoundsError.
    sampletimes = [100, 200, 300]
    path = "data/restart_test/short_schedule"
    simulation = make_simulation(steps; path=path, sampletimes=sampletimes)

    status = run!(simulation)        # must run to the end without erroring
    @test status == :completed
    @test simulation.t == steps
end
