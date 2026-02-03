# EXAMPLE: Harmonic Oscillator
include("ising_1d.jl")


###############################################################################

## RUN THE SIMULATION
seed = 42
rng = Xoshiro(seed)
β = 1.0
M = 1
j = 0.1
h = 0.1
N = 100
chains = [Ising1D(N, j, h, β) for _ in 1:M]

pool = (Move(Flip(1), EnergyBias(), ComponentArray(θ=1.0), 1.0),)
steps = 10^4
burn = 0
block = [0, steps ÷ 100]
sampletimes = build_schedule(steps, burn, block)
path = "data/PGMC/ising_1d/beta$(β)J$(j)H$(h)/"
optimisers = (VPG(1e-3),)

algorithm_list = (
    (algorithm=Metropolis, pool=pool, seed=seed, parallel=false),
    (algorithm=PolicyGradientEstimator, dependencies=(Metropolis,), optimisers=optimisers, q_batch_size=10, parallel=true),
    (algorithm=PolicyGradientUpdate, dependencies=(PolicyGradientEstimator,), scheduler=build_schedule(steps, burn, 2)),
    (algorithm=StoreCallbacks, callbacks=(energy,), scheduler=sampletimes),
    (algorithm=StoreAcceptance, dependencies=(Metropolis,), scheduler=sampletimes),
    (algorithm=StoreParameters, dependencies=(Metropolis,), scheduler=sampletimes),
    (algorithm=StoreTrajectories, scheduler=sampletimes),
    (algorithm=PrintTimeSteps, scheduler=build_schedule(steps, burn, steps ÷ 10)),
)
simulation = Simulation(chains, algorithm_list, steps; path=path, verbose=true)
run!(simulation)

## PLOT RESULTS
using Plots, Statistics, Measures, DelimitedFiles
default(tickfontsize=15, guidefontsize=15, titlefontsize=15, legendfontsize=15,
    grid=false, size=(500, 500), minorticks=5)

prms_data = readlines(joinpath(path, "moves", "1", "parameters.dat"))
steps_data = parse.(Int, getindex.(split.(prms_data, " "), 1))
time_steps = steps_data .- steps_data[1]
prms = parse.(Float64, replace.(getindex.(split.(prms_data, " "), 2), r"\[|\]" => ""))
plot(xlabel="t", ylabel="θ(t)", xscale=:log10, legend=false, title="β=$β, M=$M, η=$(optimisers[1].η)")
plot!(time_steps[2:end], prms[2:end], lw=2)
savefig("examples/ising_1d/learning.png")


