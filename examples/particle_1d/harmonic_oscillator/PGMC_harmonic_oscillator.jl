# EXAMPLE: PGMC Harmonic Oscillator
include("../particle_1d.jl")

potential(x) = x^2

###############################################################################

## RUN THE SIMULATION
seed = 42
rng = Xoshiro(seed)
β = 2.0
M = 10
chains = [System(4rand(rng) - 2, β) for _ in 1:M]
pool = (
    Move(Displacement(0.0), StandardGaussian(), ComponentArray(σ=0.2), 0.6),
    Move(Displacement(0.0), StandardGaussian(), ComponentArray(σ=0.1), 0.4),
) 
optimisers = (Static(), VPG(0.001))
steps = 10^5
burn = 0
block = [0, 1, 10]
sampletimes = build_schedule(steps, burn, block)
path = "data/PGMC/particle_1d/Harmonic/beta$β/M$M/seed$seed"
algorithm_list = (
    (algorithm=Metropolis, pool=pool, seed=seed, parallel=false),
    (algorithm=PolicyGradientEstimator, dependencies=(Metropolis,), optimisers=optimisers, parallel=false),
    (algorithm=PolicyGradientUpdate, dependencies=(PolicyGradientEstimator,)),
    (algorithm=StoreCallbacks, callbacks=(callback_energy, callback_acceptance), scheduler=sampletimes),
    (algorithm=StoreTrajectories, scheduler=sampletimes),
    (algorithm=StoreParameters, dependencies=(Metropolis,), scheduler=sampletimes),
    (algorithm=StoreLastFrames, scheduler=[steps]),
    (algorithm=PrintTimeSteps, scheduler=build_schedule(steps, burn, steps ÷ 10)),
)
simulation = Simulation(chains, algorithm_list, steps; path=path, verbose=true)
run!(simulation)

## PLOT RESULTS
using Plots, Statistics, Measures, DelimitedFiles
default(tickfontsize=15, guidefontsize=20, titlefontsize=15, legendfontsize=15,
    grid=false, size=(500, 500), minorticks=5)

energies = readdlm(joinpath(path, "energy.dat"))[:, 2]
@show mean(energies), std(energies)

prms_data = readlines(joinpath(path, "parameters", "2", "parameters.dat"))
steps_data = parse.(Int, getindex.(split.(prms_data, " "), 1))
time_steps = steps_data .- steps_data[1]
prms = parse.(Float64, replace.(getindex.(split.(prms_data, " "), 2), r"\[|\]" => ""))
plot(xlabel="t", ylabel="σ(t)", xscale=:log10, legend=false, title="β=$β, M=$M, η=$(optimisers[2].η)")
plot!(framestyle=:box, fg_legend=:transparent, dpi=300, bottommargin=-1mm, leftmargin=0mm, rightmargin=3mm)
plot!(xlim=(1e0, 1e5), ylim=(0, 1.5))
plot!(time_steps[2:end], prms[2:end], lw=2)

savefig("examples/particle_1d/harmonic_oscillator/learning.png")