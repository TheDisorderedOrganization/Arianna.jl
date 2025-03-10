"""
    abstract type AD_Backend end

Abstract type representing different automatic differentiation backends.
"""
abstract type AD_Backend end

"""
    ForwardDiff_Backend <: AD_Backend

Backend for automatic differentiation using ForwardDiff.jl.
"""
struct ForwardDiff_Backend <: AD_Backend end

"""
    reward(action::Action, system::AriannaSystem)

Compute the reward for an action in a given system.
"""
reward(action::Action, system::AriannaSystem) = Arianna.raise_error("reward")

"""
    withgrad_log_proposal_density!(∇logq::T, action::Action, policy::Policy, parameters::T, system::AriannaSystem, ::ForwardDiff_Backend;
    shadow=missing) where {T<:AbstractArray}

Compute the gradient of the log proposal density for an action in a given system using ForwardDiff.jl.
"""
function withgrad_log_proposal_density!(∇logq::T, action::Action, policy::Policy, parameters::T, system::AriannaSystem, ::ForwardDiff_Backend;
    shadow=missing) where {T<:AbstractArray}
    logq = log_proposal_density(action, policy, parameters, system)
    ∇logq .= ForwardDiff.gradient(p -> log_proposal_density(action, policy, p, system), parameters)
    return logq
end


"""
    GradientData{T<:AbstractFloat, V<:AbstractArray, V2<:AbstractArray}

Struct for storing gradient data.
"""
struct GradientData{T<:AbstractFloat, V<:AbstractArray, V2<:AbstractArray}
    j::T                # Objective function
    ∇j::V               # Gradient of the objective function
    ∇logq_forward::V    # Gradient of the log proposal density of the forward action
    g::V2               # Metric tensor for natural policy gradient
    n::Int              # Counter for averaging
end

"""
    initialise_gradient_data(parameters::AbstractArray)

Initialise gradient data for a given parameter array.
"""
function initialise_gradient_data(parameters::AbstractArray)
    j = zero(eltype(parameters))
    ∇j = zero(parameters)
    ∇logq_forward = zero(parameters)
    g = ∇j * ∇j'
    n = 0
    return GradientData(j, ∇j, ∇logq_forward, g, n)
end

"""
    Base.:+(gd1::GradientData, gd2::GradientData)

Add two gradient data instances.
"""
function Base.:+(gd1::GradientData, gd2::GradientData)
    return GradientData(
        gd1.j + gd2.j,
        gd1.∇j + gd2.∇j,
        gd1.∇logq_forward + gd2.∇logq_forward,
        gd1.g + gd2.g,
        gd1.n + gd2.n
        )
end

"""
    average(gd::GradientData)

Average gradient data over multiple samples.
"""     
function average(gd::GradientData)
    return GradientData(gd.j / gd.n, gd.∇j ./ gd.n, gd.∇logq_forward ./ gd.n, gd.g ./ gd.n, gd.n)
end

"""
    pgmc_estimate(action::Action, policy::Policy, parameters::AbstractArray{T}, system::AriannaSystem;
    ∇logq_forward=zero(parameters), ∇logq_backward=zero(parameters), shadow=deepcopy(system), ad_backend::AD_Backend=ForwardDiff_Backend()) where {T<:AbstractFloat}

Estimate the gradient of the objective function using policy gradient Monte Carlo.
"""
function pgmc_estimate(action::Action, policy::Policy, parameters::AbstractArray{T}, system::AriannaSystem;
    ∇logq_forward=zero(parameters), ∇logq_backward=zero(parameters), shadow=deepcopy(system), ad_backend::AD_Backend=ForwardDiff_Backend()) where {T<:AbstractFloat}
    ∇logq_forward .= zero(parameters)
    ∇logq_backward .= zero(parameters)
    logq_forward = withgrad_log_proposal_density!(∇logq_forward, action, policy, parameters, system, ad_backend; shadow=shadow)
    x₁, x₂ = perform_action!(system, action)
    Δlogp = delta_log_target_density(x₁, x₂, system)
    r = reward(action, system)
    invert_action!(action, system)
    logq_backward = withgrad_log_proposal_density!(∇logq_backward, action, policy, parameters, system, ad_backend; shadow=shadow)
    revert_action!(system, action)
    α = min(one(T), exp(Δlogp + logq_backward - logq_forward))
    j = r * α
    ∇j = j .* (isone(α) ? ∇logq_forward : ∇logq_backward)
    g = ∇logq_forward * ∇logq_forward'
    return GradientData(j, ∇j, ∇logq_forward, g, 1)
end

"""
    sample_gradient_data(action::Action, policy::Policy, parameters::AbstractArray, system::AriannaSystem, rng;
    ∇logq_forward=zero(parameters), ∇logq_backward=zero(parameters), shadow=deepcopy(system), ad_backend::AD_Backend=ForwardDiff_Backend())

Sample gradient data for a given action, policy, parameters, and system.
"""
function sample_gradient_data(action::Action, policy::Policy, parameters::AbstractArray, system::AriannaSystem, rng;
    ∇logq_forward=zero(parameters), ∇logq_backward=zero(parameters), shadow=deepcopy(system), ad_backend::AD_Backend=ForwardDiff_Backend())
    sample_action!(action, policy, parameters, system, rng)
    return pgmc_estimate(action, policy, parameters, system; ∇logq_forward=∇logq_forward, ∇logq_backward=∇logq_backward, shadow=shadow, ad_backend=ad_backend)
end

nothing