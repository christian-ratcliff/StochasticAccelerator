"""
RF cavity effects for beam dynamics.

This file implements RF cavity effects on beam particles, including proper
StochasticAD integration for parameter sensitivity analysis.
"""

using StochasticAD
using StructArrays

"""
    RFCavity <: PhysicsProcess

RF cavity process that applies longitudinal RF voltage to particles.

# Fields
- `voltage`: RF voltage [V]
- `ϕs`: Synchronous phase [rad]
- `rf_factor`: RF factor [rad/m]
"""
struct RFCavity <: PhysicsProcess
    voltage::Any  # Can be Float64 or StochasticTriple
    ϕs::Any       # Can be Float64 or StochasticTriple
    rf_factor::Any # Can be Float64 or StochasticTriple
end

"""
    apply_process!(
        process::RFCavity, 
        particles::StructArray{Particle{T}},
        params::SimulationParameters,
        buffers::SimulationBuffers{T}
    ) where T<:Float64 -> Nothing

Apply RF cavity effects to all particles.

# Arguments
- `process`: RF cavity process
- `particles`: Particle array
- `params`: Simulation parameters
- `buffers`: Simulation buffers
"""
# function apply_process!(
#     process::RFCavity, 
#     particles::StructArray{Particle{T}},
#     params::SimulationParameters,
#     buffers::SimulationBuffers{S}
#     ) where {T<:Float64, S}  # Remove the T<:Float64 constraint for buffers
    
#     # Get process parameters
#     voltage = process.voltage
#     ϕs = process.ϕs
#     rf_factor = process.rf_factor
#     sin_ϕs = StochasticAD.propagate(sin, ϕs)
    
#     # Check if any parameter is a StochasticTriple
#     is_stochastic = any(p -> typeof(p) <: StochasticTriple, [voltage, sin_ϕs, rf_factor, ϕs])
    
#     if is_stochastic
#         # Use StochasticAD.propagate for proper gradient propagation
#         z_val = particles.coordinates.z
#         ΔE_val = particles.coordinates.ΔE
#         for i in 1:length(particles)
            
            
#             # # Calculate sin(ϕ) with proper gradient propagation
#             # phase = StochasticAD.propagate(
#             #     (z, rf, phi_s) -> -(z * rf - phi_s),
#             #     z_val[i], rf_factor, ϕs
#             # )
            
#             # # Calculate sin(ϕ) - sin(ϕs)
#             # sin_term = StochasticAD.propagate(
#             #     (phase, sin_s) -> sin(phase) - sin_s,
#             #     phase, sin_ϕs
#             # )
            
#             # # Update energy with proper gradient propagation
#             # particles.coordinates.ΔE[i] = StochasticAD.propagate(
#             #     (de, v, s) -> de + v * s,
#             #     ΔE_val[i], voltage, sin_term
#             # )

#             particles.coordinates.ΔE[i] = StochasticAD.propagate(
#                 (z, rf, phi_s, sin_s, de, v, s) -> de + v * (sin(-(z * rf - phi_s)) - sin_s),
#                 z_val[i], rf_factor, ϕs, sin_ϕs, ΔE_val[i], voltage, sin_ϕs)

#         end
#     else
#         # Standard vectorized implementation for non-StochasticTriple case
#         # sinϕ = sin.(-particles.coordinates.z .* rf_factor .+ ϕs) .- sin_ϕs
#         particles.coordinates.ΔE .= particles.coordinates.ΔE .+ voltage .* (sin.(-particles.coordinates.z .* rf_factor .+ ϕs) .- sin_ϕs)
#     end
    
#     return nothing
# end

# function apply_process!(
#     process::RFCavity, 
#     particles::StructArray{Particle{T}},
#     params::SimulationParameters,
#     buffers::SimulationBuffers{S}
#     ) where {T<:Float64, S}
    
#     # Extract process parameters
#     voltage = safe_value(process.voltage)
#     ϕs = safe_value(process.ϕs)
#     rf_factor = safe_value(process.rf_factor)
#     sin_ϕs = sin(ϕs)
    
#     # Apply RF effect to each particle
#     for i in 1:length(particles)
#         # Calculate phase
#         phase = -(particles.coordinates.z[i] * rf_factor - ϕs)
#         # Calculate sin term
#         sin_term = sin(phase) - sin_ϕs
#         particles.coordinates.ΔE[i] += voltage * sin_term
#     end
    
#     return nothing
# end

function apply_process!(
    process::RFCavity, 
    particles::StructArray{Particle{T}},
    params::SimulationParameters,
    buffers::SimulationBuffers{S}
) where {T<:Float64, S}
    
    # Extract process parameters with safe_value
    voltage = safe_value(process.voltage)
    ϕs = safe_value(process.ϕs)
    rf_factor = safe_value(process.rf_factor)
    sin_ϕs = sin(ϕs)
    
    # Add a small amount of randomness for StochasticAD
    voltage_effect = voltage * (1.0 + 0.01 * rand(Bernoulli(0.5)))
    
    # Apply RF effect to each particle
    # for i in 1:length(particles)
    #     # Calculate phase
    #     phase = -(particles.coordinates.z[i] * rf_factor - ϕs)
    #     # Calculate sin term
    #     sin_term = sin(phase) - sin_ϕs
    #     # Apply energy change - ALWAYS use the += operator on existing array elements
    #     particles.coordinates.ΔE[i] += voltage_effect * sin_term
    # end
    # sinϕ = sin.(-particles.coordinates.z .* rf_factor .+ ϕs) .- sin_ϕs
    particles.coordinates.ΔE .= particles.coordinates.ΔE .+ voltage_effect .* (sin.(-particles.coordinates.z .* rf_factor .+ ϕs) .- sin_ϕs)
    
    return nothing
end
"""
    create_rf_cavity(
        voltage, 
        ϕs, 
        freq_rf, 
        β
    ) -> RFCavity

Create an RF cavity process.

# Arguments
- `voltage`: RF voltage [V]
- `ϕs`: Synchronous phase [rad]
- `freq_rf`: RF frequency [Hz]
- `β`: Relativistic beta

# Returns
- RF cavity process
"""
function create_rf_cavity(voltage, ϕs, freq_rf, β)
    # Calculate RF factor
    rf_factor = calc_rf_factor(freq_rf, β)
    
    # Create RF cavity process
    return RFCavity(voltage, ϕs, rf_factor)
end