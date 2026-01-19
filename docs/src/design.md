# Design strategy

Under the hood, Arianna.jl runs, at each step, a set of *algorithms* that operate on (possibly parallel) *systems* by performing actions that depend only on the *current* state of the system.
Each algorithm has its own *scheduler*, which specifies at which steps of the simulation’s ``internal clock'' it should be executed.
This borad definition of algorithm allows to use the same framework to implement, for instance, different transition kernels, adaptive schemes such as [Policy-guided Monte Carlo](https://thedisorderedorganization.github.io/Arianna.jl/stable/man/policyguided/) (PGMC), user-designed callbacks at specific times, data-management routines, and even molecular dynamics integrators.
To keep this level of abstraction, Arianna.jl relies heavily on Julia's multiple dispatch system, which replaces class-based methods with functions having specialised methods for different combinations of argument types.

Despite its generality, the package is designed designed primarily for MCMC simulations of physical systems.
To this end, as a core feature, it includes a general implementation of the Metropolis-Hastings algorithm that can be customised by defining a few essential methods for the system at hand.
Following a physics-oriented approach, a transition kernel is defined by a *move* which combines the *action* of updating the current state of the system with the *policy* (*i.e.* proposal distribution) that specifies how to sample the action and how to evaluate its probability density.
Multiple moves with specific *weights* can be combined to build a composite transition kernel.
Arianna.jl then handles proposing, accepting or rejecting moves, and updating the system efficiently.

Performance considerations guided several design choices.
The system is *mutable*, and only the parts affected by a move are updated.
This keeps large-system simulations fast on CPU, although it limits direct GPU support.
As a consequence, the package is well suited for PGMC with a relatively small number of parameters, but not for policies based on large generative models.

Another important design choice is the support for multiple synchronised chains running in parallel on different CPU cores.
While traditional MCMC simulations rarely require such synchronisation (and the overhead is often not justified), it becomes essential for certain techniques such as parallel tempering.
PGMC also benefits from parallel chains, which can be used to obtain more stable gradient estimates before each policy update.
Naturally, users can also run completely independent simulations, which is often preferred in high-performance computing (HPC) environments.
While in conventional MCMC simulations this is often avoided as there is no need to syncronise multiple chains and single core simulations are easier to deal with in HPC cluster (also adds overhead), this feature is crucial in some applications such as parallel tempering.
PGMC also benefits from parallel chains, as they can be used to get a better estimate of gradients before each update.
Of course users can run independent simulations.

Since the resulting interface is quite abstract, the package includes a few simple example systems.
The intention, however, is that Arianna.jl serves as a foundation upon which more specialised code can be built.
Our own [ParticlesMC](https://github.com/TheDisorderedOrganization/ParticlesMC) package is an example of this.
