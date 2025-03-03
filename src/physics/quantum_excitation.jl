"""
Quantum excitation effects for beam dynamics.

This file implements the quantum excitation effects that cause stochastic
energy fluctuations in the beam due to the quantum nature of synchrotron radiation.
"""

using Random
using LoopVectorization
using StochasticAD
using Distributions
using StructArrays

"""
    QuantumExcitation <: PhysicsProcess

Quantum excitation process that applies random energy kicks to particles.

# Fields
- `E0`: Reference energy [eV]
- `radius`: Accelerator radius [m]
- `σ_E0`: Initial energy spread [eV]
"""
struct QuantumExcitation <: PhysicsProcess
    E0::Any     # Can be Float64 or StochasticTriple
    radius::Any # Can be Float64 or StochasticTriple
    σ_E0::Float64
end

"""
    apply_process!(
        process::QuantumExcitation, 
        particles::StructArray{Particle{T}},
        params::SimulationParameters,
        buffers::SimulationBuffers{T}
    ) where T<:Float64 -> Nothing

Apply quantum excitation to all particles.

# Arguments
- `process`: Quantum excitation process
- `particles`: Particle array
- `params`: Simulation parameters
- `buffers`: Simulation buffers
"""
# function apply_process!(
#     process::QuantumExcitation, 
#     particles::StructArray{Particle{T}},
#     params::SimulationParameters,
#     buffers::SimulationBuffers{S}
#     ) where {T<:Float64, S}
    
#     # Get process parameters
#     E0 = process.E0
#     radius = process.radius
#     σ_E0 = process.σ_E0
    
#     # Create a temporary Float64 buffer for random values
#     temp_random = Vector{Float64}(undef, length(buffers.random_buffer))
#     randn!(MersenneTwister(Int(round(rand()*100))), temp_random)
    
#     # Check if any parameter is a StochasticTriple
#     is_stochastic = any(p -> typeof(p) <: StochasticTriple, [E0, radius])
    
#     if is_stochastic
#         # Calculate excitation parameter
#         excitation_fn = (e0, r) -> begin
#             ∂U_∂E = 4 * 8.85e-5 * (e0/1e9)^3 / r
#             return sqrt(1-(1-∂U_∂E)^2) * σ_E0
#         end
        
#         # Get excitation amplitude with proper gradient propagation
#         excitation = StochasticAD.propagate(excitation_fn, E0, radius)
        
#         # Apply to each particle with proper StochasticTriple handling
#         for i in 1:length(particles)
#             # Use StochasticAD.propagate for the random kick
#             kick = StochasticAD.propagate(
#                 (exc, r) -> exc * r,
#                 excitation, 
#                 temp_random[i]
#             )
            
#             # Add kick to energy with proper gradient propagation
#             particles.coordinates.ΔE[i] = StochasticAD.propagate(
#                 (de, k) -> de + k,
#                 particles.coordinates.ΔE[i],
#                 kick
#             )
#         end
#     else
#         # Standard implementation for non-StochasticTriple case
#         ∂U_∂E = 4 * 8.85e-5 * (E0/1e9)^3 / radius
#         excitation = sqrt(1-(1-∂U_∂E)^2) * σ_E0
        
#         # Apply kicks to all particles
#         particles.coordinates.ΔE .+= excitation .* temp_random
#     end
    
#     return nothing
# end
function apply_process!(
    process::QuantumExcitation, 
    particles::StructArray{Particle{T}},
    params::SimulationParameters,
    buffers::SimulationBuffers{S}
    ) where {T<:Float64, S}
    
    # Get process parameters with safe_value
    E0 = safe_value(process.E0)
    radius = safe_value(process.radius)
    σ_E0 = process.σ_E0
    
    # Calculate excitation amplitude
    ∂U_∂E = 4 * 8.85e-5 * (E0/1e9)^3 / radius
    excitation = sqrt(1-(1-∂U_∂E)^2) * σ_E0
    
    # Add randomness with discrete probability for StochasticAD
    excitation *= (1.0 + 0.02 * rand(Bernoulli(0.5)))
    
    # Generate random kicks
    randn!(MersenneTwister(Int(round(rand()*100))), buffers.random_buffer)
    
    # Apply kicks to all particles
    for i in 1:length(particles)
        particles.coordinates.ΔE[i] += excitation * buffers.random_buffer[i]
    end
    
    return nothing
end
"""
    create_quantum_excitation(
        E0,
        radius,
        σ_E0
    ) -> QuantumExcitation

Create a quantum excitation process.

# Arguments
- `E0`: Reference energy [eV]
- `radius`: Accelerator radius [m]
- `σ_E0`: Initial energy spread [eV]

# Returns
- Quantum excitation process
"""
function create_quantum_excitation(E0, radius, σ_E0)
    return QuantumExcitation(E0, radius, σ_E0)
end