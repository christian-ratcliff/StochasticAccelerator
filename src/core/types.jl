"""
Core data structures for beam simulations.

This file defines the fundamental data structures used in the simulation:
- Coordinate: Longitudinal phase space coordinates
- Particle: Particle representation
- PhysicsProcess: Abstract type for physics processes
- SimulationParameters: Complete simulation parameters
- SimulationBuffers: Pre-allocated buffers for efficient computation
"""

using StaticArrays
using StructArrays

"""
    Coordinate{T} <: FieldVector{2, T}

Coordinate in longitudinal phase space.

# Fields
- `z::T`: Longitudinal position relative to reference particle
- `ΔE::T`: Energy deviation from reference energy
"""
struct Coordinate{T} <: FieldVector{2, T}
    z::T
    ΔE::T
end

"""
    Particle{T} <: FieldVector{2, Coordinate}

Particle representation with coordinates and uncertainty.

# Fields
- `coordinates::Coordinate{T}`: Current position in phase space
- `uncertainty::Coordinate{T}`: 
"""
struct Particle{T} <: FieldVector{2, Coordinate}
    coordinates::Coordinate{T}
    uncertainty::Coordinate{T}
end

"""
    PhysicsProcess

Abstract type for all physics processes that can be applied to particles.
"""
abstract type PhysicsProcess end

"""
    BeamParameter

Abstract type for beam parameters that can be optimized.
"""
abstract type BeamParameter end

"""
    SimulationParameters

Type-stable container for all simulation parameters that can include StochasticTriple values.
"""
struct SimulationParameters{TE,TM,TV,TR,TPR,TA,TPS,TF}
    E0::TE               # Reference energy [eV]
    mass::TM             # Particle mass [eV/c²]
    voltage::TV          # RF voltage [V]
    harmonic::Int        # RF harmonic number
    radius::TR           # Accelerator radius [m]
    pipe_radius::TPR     # Beam pipe radius [m]
    α_c::TA              # Momentum compaction factor
    ϕs::TPS              # Synchronous phase [rad]
    freq_rf::TF          # RF frequency [Hz]
    n_turns::Int         # Number of turns to simulate
    use_wakefield::Bool  # Enable wakefield effects
    update_η::Bool       # Update slip factor
    update_E0::Bool      # Update reference energy
    SR_damping::Bool     # Enable synchrotron radiation damping
    use_excitation::Bool # Enable quantum excitation
end

# Convenience constructor for all Float64 parameters
function SimulationParameters(E0::Float64, mass::Float64, voltage::Float64, 
                             harmonic::Int, radius::Float64, pipe_radius::Float64, 
                             α_c::Float64, ϕs::Float64, freq_rf::Float64, 
                             n_turns::Int, use_wakefield::Bool, update_η::Bool, 
                             update_E0::Bool, SR_damping::Bool, use_excitation::Bool)
    return SimulationParameters{Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64}(
        E0, mass, voltage, harmonic, radius, pipe_radius, 
        α_c, ϕs, freq_rf, n_turns, use_wakefield, 
        update_η, update_E0, SR_damping, use_excitation
    )
end

"""
    SimulationBuffers{T<:Float64}

Pre-allocated buffers for efficient computation during simulation.
"""
struct SimulationBuffers{T}
    WF::Vector{T}                # Wakefield values
    potential::Vector{T}         # Potential energy
    Δγ::Vector{T}                # Gamma factor deviations
    η::Vector{T}                 # Slip factor
    coeff::Vector{T}             # Coefficients
    temp_z::Vector{T}            # Temporary z coordinates
    temp_ΔE::Vector{T}           # Temporary energy deviations
    temp_ϕ::Vector{T}            # Temporary phases
    WF_temp::Vector{T}           # Temporary wakefield values
    λ::Vector{T}                 # Line charge density
    convol::Vector{Complex{T}}   # Convolution results
    ϕ::Vector{T}                 # Phase values
    random_buffer::Vector{T}     # Random number buffer
end


const Parameter = Union{Float64, StochasticTriple}
