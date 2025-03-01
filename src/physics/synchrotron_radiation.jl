"""
Synchrotron radiation effects for beam dynamics.

This file implements synchrotron radiation damping effects that 
reduce energy spread in the beam.
"""

using StochasticAD
using StructArrays

"""
    SynchrotronRadiation <: PhysicsProcess

Synchrotron radiation process that applies radiation damping to particles.

# Fields
- `E0`: Reference energy [eV]
- `radius`: Accelerator radius [m]
"""
struct SynchrotronRadiation <: PhysicsProcess
    E0::Any     # Can be Float64 or StochasticTriple
    radius::Any # Can be Float64 or StochasticTriple
end

"""
    apply_process!(
        process::SynchrotronRadiation, 
        particles::StructArray{Particle{T}},
        params::SimulationParameters,
        buffers::SimulationBuffers{T}
    ) where T<:Float64 -> Nothing

Apply synchrotron radiation damping to all particles.

# Arguments
- `process`: Synchrotron radiation process
- `particles`: Particle array
- `params`: Simulation parameters
- `buffers`: Simulation buffers
"""
function apply_process!(
    process::SynchrotronRadiation, 
    particles::StructArray{Particle{T}},
    params::SimulationParameters,
    buffers::SimulationBuffers{T}
) where T<:Float64
    
    # Get process parameters
    E0 = process.E0
    radius = process.radius
    
    # Check if any parameter is a StochasticTriple
    is_stochastic = any(p -> typeof(p) <: StochasticTriple, [E0, radius])
    
    if is_stochastic
        # Calculate damping factor with StochasticAD.propagate
        damping_fn = (e0, r) -> begin
            ∂U_∂E = 4 * 8.85e-5 * (e0/1e9)^3 / r
            return 1 - ∂U_∂E
        end
        
        # Get the damping factor with proper gradient propagation
        damping_factor = StochasticAD.propagate(damping_fn, E0, radius)
        
        # Apply to all particles with proper StochasticTriple handling
        for i in 1:length(particles)
            particles.coordinates.ΔE[i] = StochasticAD.propagate(
                (de, df) -> de * df,
                particles.coordinates.ΔE[i],
                damping_factor
            )
        end
    else
        # Standard implementation for non-StochasticTriple case
        ∂U_∂E = 4 * 8.85e-5 * (E0/1e9)^3 / radius
        damping_factor = 1 - ∂U_∂E
        
        # Apply damping to all particles
        particles.coordinates.ΔE .*= damping_factor
    end
    
    return nothing
end

"""
    create_synchrotron_radiation(
        E0,
        radius
    ) -> SynchrotronRadiation

Create a synchrotron radiation process.

# Arguments
- `E0`: Reference energy [eV]
- `radius`: Accelerator radius [m]

# Returns
- Synchrotron radiation process
"""
function create_synchrotron_radiation(E0, radius)
    return SynchrotronRadiation(E0, radius)
end