using Arianna
using Arianna.PolicyGuided
using Test
using DelimitedFiles

include("../examples/particle_1d/particle_1d.jl")

potential(x) = x^2

@testset "Displacement optimisation" begin
    seed = 42
    rng = Xoshiro(seed)
    β = 2.0
    M = 10
    chains = [System(4rand(rng) - 2, β) for _ in 1:M]
    σ₀ = 0.2
    pool = (
        Move(Displacement(0.0), StandardGaussian(), ComponentArray(σ=σ₀), 0.4),
        Move(Displacement(0.0), StandardGaussian(), ComponentArray(σ=σ₀), 0.1),
        Move(Displacement(0.0), StandardGaussian(), ComponentArray(σ=σ₀), 0.1),
        Move(Displacement(0.0), StandardGaussian(), ComponentArray(σ=σ₀), 0.1),
        Move(Displacement(0.0), StandardGaussian(), ComponentArray(σ=σ₀), 0.1),
        Move(Displacement(0.0), StandardGaussian(), ComponentArray(σ=σ₀), 0.1),
        Move(Displacement(0.0), StandardGaussian(), ComponentArray(σ=σ₀), 0.1),
    )
    optimisers = (Static(), VPG(0.001), BLPG(0.001), BLAPG(1e-6, 1e-6), NPG(1e-2, 1e-6), ANPG(1e-6, 1e-6), BLANPG(1e-6, 1e-6))
    steps = 10^5
    burn = 1000
    block = [0, 10]
    sampletimes = build_schedule(steps, burn, block)
    path = "data/PGMC/particle_1d/Harmonic/beta$β/M$M/seed$seed"
    algorithm_list = (
        (algorithm=Metropolis, pool=pool, seed=seed, parallel=false),
        (algorithm=PolicyGradientEstimator, dependencies=(Metropolis,), optimisers=optimisers, q_batch_size=10, parallel=true),
        (algorithm=PolicyGradientUpdate, dependencies=(PolicyGradientEstimator,), scheduler=build_schedule(steps, burn, 2)),
        (algorithm=StoreCallbacks, callbacks=(energy,), scheduler=sampletimes),
        (algorithm=StoreAcceptance, dependencies=(Metropolis,), scheduler=sampletimes),
        (algorithm=StoreObjective, dependencies=(PolicyGradientEstimator,), scheduler=sampletimes),
        (algorithm=StoreTrajectories, scheduler=sampletimes),
        (algorithm=StoreParameters, dependencies=(Metropolis,), scheduler=sampletimes),
        (algorithm=StoreLastFrames, scheduler=[steps]),
        (algorithm=PrintTimeSteps, scheduler=build_schedule(steps, burn, steps ÷ 10)),
    )
    simulation = Simulation(chains, algorithm_list, steps; path=path, verbose=true)
    run!(simulation)
    energies = reduce(vcat, [readdlm(joinpath(path, "chains", "$k", "energy.dat"))[:, 2] for k in 1:M])
    @test isapprox(mean(energies), 0.25, atol=5e-2)
    prms_path = joinpath(path, "moves")
    for (opt, dir) in zip(optimisers, readdir(prms_path))
        prms_data = readlines(joinpath(prms_path, dir, "parameters.dat"))
        sigmas = parse.(Float64, replace.(getindex.(split.(prms_data, " "), 2), r"\[|\]" => ""))
        @test (isa(opt, Static) && sigmas[end] == σ₀) || isapprox(sigmas[end], 1.2, atol=2e-1)
    end
end