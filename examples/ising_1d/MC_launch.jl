# EXAMPLE: Harmonic Oscillator
include("ising_1d.jl")


###############################################################################

## RUN THE SIMULATION
seed = 42
rng = Xoshiro(seed)
β = 0.5
M = 1
j= 0.1
h = 0.1
N=100
chains = [Ising1D(N, j, h, β) for _ in 1:M]
pool = (Move(Flip(1), StandardUniform(), Vector{Float64}(), 1.0), )
steps = 2*10^7
burn = 10^7
block = [0, 10^3]
sampletimes = build_schedule(steps, burn, block)
path = "examples/data/MC/ising_1d/beta$(β)J$(j)H$(h)/"

algorithm_list = (
    (algorithm=Metropolis, pool=pool, seed=seed, parallel=false),
    (algorithm=StoreCallbacks, callbacks=(callback_energy, callback_acceptance), store_first=false, scheduler=sampletimes),
    (algorithm=StoreTrajectories, scheduler=sampletimes),
    (algorithm=PrintTimeSteps, scheduler=build_schedule(steps, burn, steps ÷ 10)),
) 
simulation = Simulation(chains, algorithm_list, steps; path=path, verbose=true)
run!(simulation)

## PLOT RESULTS
using Plots, Statistics, Measures, DelimitedFiles
default(tickfontsize=15, guidefontsize=15, titlefontsize=15, legendfontsize=15,
    grid=false, size=(500, 500), minorticks=5)

energies = readdlm(joinpath(path, "energy.dat"))[:, 2]
@show mean(energies), std(energies)

stephist!(energies, normalize=:pdf, lw=3, c=1)
savefig("examples/ising_1d/density.png")



