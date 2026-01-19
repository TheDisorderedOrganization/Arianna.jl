using Arianna
using Test
using DelimitedFiles

include("../examples/particle_1d/particle_1d.jl")

potential(x) = x^2

@testset "Harmonic oscillator distribution" begin
    seed = 42
    rng = Xoshiro(seed)
    M = 100
    steps = 10^6
    burn = 1000
    block = [0, 10]
    sampletimes = build_schedule(steps, burn, block)
    for β in [2.0, 2.5, 3.0]
        chains = [System(4rand(rng) - 2, β) for _ in 1:M]
        pool = (Move(Displacement(0.0), StandardGaussian(), ComponentArray(σ=0.1), 1.0),)
        path = "data/MC/particle_1d/Harmonic/beta$β/M$M/seed$seed"
        algorithm_list = (
            (algorithm=Metropolis, pool=pool, seed=seed, parallel=false),
            (algorithm=StoreCallbacks, callbacks=(callback_energy,), scheduler=sampletimes),
            (algorithm=StoreAcceptance, dependencies=(Metropolis,), scheduler=sampletimes),
            (algorithm=StoreTrajectories, scheduler=sampletimes),
            (algorithm=StoreBackups, scheduler=build_schedule(steps, burn, steps ÷ 10), store_first=true, store_last=true),
            (algorithm=StoreLastFrames, scheduler=[steps]),
            (algorithm=PrintTimeSteps, scheduler=build_schedule(steps, burn, steps ÷ 10)),
        )
        simulation = Simulation(chains, algorithm_list, steps; path=path, verbose=true)
        run!(simulation)
        μ⁺ = 0.0
        σ⁺ = 1 / sqrt(2β)
        trj_files = [joinpath(dir, "trajectory.dat") for dir in readdir(joinpath(path, "chains"), join=true)]
        trajectories = map(file -> readdlm(file)[:, 2], trj_files)
        positions = vcat(trajectories...)
        @test isapprox(mean(positions), μ⁺, atol=1e-3)
        @test isapprox(std(positions), σ⁺, atol=1e-3)
    end
end