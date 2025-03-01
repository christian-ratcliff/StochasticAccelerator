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
function apply_process!(
    process::RFCavity, 
    particles::StructArray{Particle{T}},
    params::SimulationParameters,
    buffers::SimulationBuffers{T}
) where T<:Float64
    
    # Get process parameters
    voltage = process.voltage
    ϕs = process.ϕs
    rf_factor = process.rf_factor
    sin_ϕs = sin(ϕs)
    
    # Check if any parameter is a StochasticTriple
    is_stochastic = any(p -> typeof(p) <: StochasticTriple, [voltage, sin_ϕs, rf_factor, ϕs])
    
    if is_stochastic
        # Use StochasticAD.propagate for proper gradient propagation
        StochasticAD.propagate(voltage, sin_ϕs, rf_factor, ϕs) do v, sin_s, rf, phase_s
            @inbounds begin
                
                # @fasthmath sinϕ = sin.(-particles.coordinates.z .* rf_factor .+ ϕs) .- sin_ϕs
                # particles.coordinates.ΔE .= particles.coordinates.ΔE .+ voltage .* sinϕ
                z_vals = particles.coordinates.z[i]  # Cache the value
                ΔE_vals = particles.coordinates.ΔE[i]
                # Iterate and use cached values
                for i in 1:length(particles)
                      # Cache the value
                    
                    ϕ_val = -(z_vals[i] * rf_factor - ϕs)
                    ΔE_vals[i] += voltage * (sin(ϕ_val) - sin_ϕs)
                    
                    particles.coordinates.ΔE[i] = ΔE_i[i] # Store the updated value
                end
            end
            
            return nothing  # Result is not used, but propagation is ensured
        end
    else
        # Standard vectorized implementation for non-StochasticTriple case


        sinϕ = sin.(-particles.coordinates.z .* rf_factor .+ ϕs) .- sin_ϕs
        particles.coordinates.ΔE .= particles.coordinates.ΔE .+ voltage .* sinϕ
        
    end

    # sin(-z_i * rf_factor + ϕs) = sin(-z_i * rf_factor) * cos(ϕs) - cos(-z_i * rf_factor) * sin(ϕs)
    
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