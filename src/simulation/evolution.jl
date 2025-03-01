"""
Beam evolution and tracking functions.

This file implements the core longitudinal evolution algorithm, 
tracking particles through the accelerator for multiple turns.
"""

using StructArrays
using StaticArrays
using Statistics
using Random
using Distributions
using ProgressMeter
using LoopVectorization
using StochasticAD

"""
    track_particles!(
        particles::StructArray{Particle{T}},
        processes::Vector{PhysicsProcess},
        params::SimulationParameters,
        buffers::SimulationBuffers{T};
        update_η::Bool=true,
        update_E0::Bool=true
    ) where T<:Float64 -> Tuple{T, T, Any}

Track particles through all processes for one turn.

# Arguments
- `particles`: Particle array
- `processes`: Vector of physics processes
- `params`: Simulation parameters
- `buffers`: Simulation buffers
- `update_η`: Whether to update slip factor
- `update_E0`: Whether to update reference energy

# Returns
- `σ_E`: Energy spread at end of turn [eV]
- `σ_z`: Bunch length at end of turn [m]
- `E0`: Updated reference energy [eV]
"""
function track_particles!(
    particles::StructArray{Particle{T}},
    processes::Vector{PhysicsProcess},
    params::SimulationParameters,
    buffers::SimulationBuffers{T};
    update_η::Bool=true,
    update_E0::Bool=true
) where T<:Float64
    
    # Extract parameters
    E0 = params.E0
    mass = params.mass
    harmonic = params.harmonic
    
    # Calculate derived parameters
    γ0 = StochasticAD.propagate((energy, m) -> energy / m, E0, mass)
    β0 = StochasticAD.propagate(γ -> sqrt(1 - 1/γ^2), γ0)
    η0 = StochasticAD.propagate((alpha, gamma) -> alpha - 1/(gamma^2), params.α_c, γ0)
    rf_factor = StochasticAD.propagate((freq, beta) -> freq * 2π / (beta * SPEED_LIGHT), params.freq_rf, β0)
    
    # Get initial stats
    σ_E::T = std(particles.coordinates.ΔE)
    σ_z::T = std(particles.coordinates.z)
    
    # Store previous energy values if updating E0
    if update_E0
        ΔE_before = deepcopy(particles.coordinates.ΔE)
    end
    
    # Apply all physics processes
    for process in processes
        # apply_process!(process, particles, buffers)
        apply_process!(process, particles, params, buffers)
    end
    
    # Update reference energy if needed
    if update_E0
        # Calculate mean energy change
        mean_ΔE_diff = mean(particles.coordinates.ΔE .- ΔE_before)
        
        # Update reference energy
        E0 = StochasticAD.propagate((energy, diff) -> energy + diff, E0, mean_ΔE_diff)
        
        # Recalculate derived parameters
        γ0 = StochasticAD.propagate((energy, m) -> energy / m, E0, mass)
        β0 = StochasticAD.propagate(γ -> sqrt(1 - 1/γ^2), γ0)
        η0 = StochasticAD.propagate((alpha, gamma) -> alpha - 1/(gamma^2), params.α_c, γ0)
        
        # Zero the mean energy deviation
        mean_ΔE = mean(particles.coordinates.ΔE)
        safe_update_energy!(particles, mean_ΔE)
    end
    
    # Update phase advance
    if update_η
        # Update phase for each particle with individual slip factor
        for i in 1:length(particles)
            # Calculate slip factor for this particle
            Δγ_i = particles.coordinates.ΔE[i] / mass
            η_i = StochasticAD.propagate(
                (alpha, gamma, delta_gamma) -> alpha - 1/(gamma + delta_gamma)^2,
                params.α_c, γ0, Δγ_i
            )
            
            # Update position with proper StochasticTriple handling
            particles.coordinates.z[i] = StochasticAD.propagate(
                (eta_i, h, beta, energy, rf, ϕs, ΔE) -> begin
                    coeff_i = 2π * h * eta_i / (beta * beta * energy)
                    ϕ_i = z_to_ϕ(particles.coordinates.z[i], rf, ϕs)
                    ϕ_i += coeff_i * ΔE
                    return ϕ_to_z(ϕ_i, rf, ϕs)
                end,
                η_i, harmonic, β0, E0, rf_factor, params.ϕs, particles.coordinates.ΔE[i]
            )
        end
    else
        # Use constant slip factor for all particles
        apply_phase_advance!(particles, η0, harmonic, β0, E0, rf_factor, params.ϕs)
    end
    
    # Calculate final stats
    σ_E = std(particles.coordinates.ΔE)
    σ_z = std(particles.coordinates.z)
    
    return σ_E, σ_z, E0
end

"""
    apply_phase_advance!(
        particles::StructArray{Particle{T}},
        η0,
        harmonic,
        β0,
        E0,
        rf_factor,
        ϕs
    ) where T<:Float64 -> Nothing

Apply phase advancement to all particles using constant slip factor.

# Arguments
- `particles`: Particle array
- `η0`: Slip factor
- `harmonic`: Harmonic number
- `β0`: Relativistic beta
- `E0`: Reference energy [eV]
- `rf_factor`: RF factor [rad/m]
- `ϕs`: Synchronous phase [rad]
"""
function apply_phase_advance!(
    particles::StructArray{Particle{T}},
    η0,
    harmonic,
    β0,
    E0,
    rf_factor,
    ϕs
) where T<:Float64
    
    # Check if any parameter is a StochasticTriple
    is_stochastic = any(typeof(param) <: StochasticTriple for param in 
                        [η0, harmonic, β0, E0, rf_factor, ϕs])
    
    if is_stochastic
        # Calculate coefficient with proper gradient propagation
        coeff = StochasticAD.propagate(
            (_η0, _harmonic, _β0, _E0) -> begin
                return 2π * _harmonic * _η0 / (_β0 * _β0 * _E0)
            end, 
            η0, harmonic, β0, E0
        )
        
        # Process each particle
        for i in 1:length(particles)
            # Calculate new phase with proper gradient propagation
            particles.coordinates.z[i] = StochasticAD.propagate(
                (_rf, _ϕs, z, c, ΔE) -> begin
                    ϕ = z_to_ϕ(z, _rf, _ϕs)
                    ϕ += c * ΔE
                    return ϕ_to_z(ϕ, _rf, _ϕs)
                end,
                rf_factor, ϕs, particles.coordinates.z[i], coeff, particles.coordinates.ΔE[i]
            )
        end
    else
        # Standard implementation for non-StochasticTriple case
        coeff = 2π * harmonic * η0 / (β0 * β0 * E0)
        
        # Process all particles
        for i in 1:length(particles)
            ϕ_i = z_to_ϕ(particles.coordinates.z[i], rf_factor, ϕs)
            ϕ_i += coeff * particles.coordinates.ΔE[i]
            particles.coordinates.z[i] = ϕ_to_z(ϕ_i, rf_factor, ϕs)
        end
    end
    
    return nothing
end

"""
    safe_update_energy!(particles::StructArray{Particle{T}}, mean_value) where T<:Float64 -> Nothing

Safely update particle energy deviations by subtracting the mean.
Handles StochasticTriple values properly.

# Arguments
- `particles`: Particle array
- `mean_value`: Mean energy deviation to subtract
"""
function safe_update_energy!(particles::StructArray{Particle{T}}, mean_value) where T<:Float64
    # If mean_value is a StochasticTriple, we need special handling
    if typeof(mean_value) <: StochasticTriple
        for i in 1:length(particles)
            # Use propagate to handle the subtraction properly
            particles.coordinates.ΔE[i] = StochasticAD.propagate(
                (e, m) -> e - m,
                particles.coordinates.ΔE[i],
                mean_value
            )
        end
    else
        # If it's a regular Float64, just do the subtraction directly
        particles.coordinates.ΔE .-= mean_value
    end
    return nothing
end

"""
    run_simulation(
        params::SimulationParameters,
        processes::Vector{PhysicsProcess},
        n_particles::Int;
        μ_z::Float64=0.0,
        μ_E::Float64=0.0,
        σ_z0::Float64=0.005,
        σ_E0::Float64=1e6,
        show_progress::Bool=true
    ) -> Tuple{StructArray{Particle{Float64}}, Float64, Float64, Any}

Run a complete beam evolution simulation.

# Arguments
- `params`: Simulation parameters
- `processes`: Vector of physics processes
- `n_particles`: Number of particles
- `μ_z`: Mean longitudinal position [m]
- `μ_E`: Mean energy deviation [eV]
- `σ_z0`: Initial RMS bunch length [m]
- `σ_E0`: Initial RMS energy spread [eV]
- `show_progress`: Whether to show progress bar

# Returns
- `particles`: Final particle distribution
- `σ_E`: Final energy spread [eV]
- `σ_z`: Final bunch length [m]
- `E0`: Final reference energy [eV]
"""
function run_simulation(
    params::SimulationParameters,
    processes::Vector{PhysicsProcess},
    n_particles::Int;
    μ_z::Float64=0.0,
    μ_E::Float64=0.0,
    σ_z0::Float64=0.005,
    σ_E0::Float64=1e6,
    show_progress::Bool=true
)
    # Generate initial particles
    particles, _, _, E0 = generate_particles(
        μ_z, μ_E, σ_z0, σ_E0, n_particles,
        params.E0, params.mass, params.ϕs, params.freq_rf
    )
    
    # Create buffers
    nbins = Int(n_particles/10)
    buffers = create_simulation_buffers(n_particles, nbins, Float64)
    
    # Setup progress meter
    prog = show_progress ? Progress(params.n_turns, desc="Simulating turns: ") : nothing
    
    # Evolve for multiple turns
    for turn in 1:params.n_turns
        # Track particles for one turn
        σ_E, σ_z, E0 = track_particles!(
            particles, 
            processes, 
            params, 
            buffers;
            update_η=params.update_η,
            update_E0=params.update_E0
        )
        
        # Update progress
        show_progress && next!(prog)
    end
    
    # Calculate final statistics
    σ_E = std(particles.coordinates.ΔE)
    σ_z = std(particles.coordinates.z)
    
    return particles, σ_E, σ_z, E0
end

"""
    setup_simulation(
        params::SimulationParameters,
        n_particles::Int;
        μ_z::Float64=0.0,
        μ_E::Float64=0.0,
        σ_z0::Float64=0.005,
        σ_E0::Float64=1e6
    ) -> Tuple{StructArray{Particle{Float64}}, Vector{PhysicsProcess}, SimulationBuffers{Float64}}

Setup a simulation with appropriate physics processes.

# Arguments
- `params`: Simulation parameters
- `n_particles`: Number of particles
- `μ_z`: Mean longitudinal position [m]
- `μ_E`: Mean energy deviation [eV]
- `σ_z0`: Initial RMS bunch length [m]
- `σ_E0`: Initial RMS energy spread [eV]

# Returns
- `particles`: Initial particle distribution
- `processes`: Vector of physics processes
- `buffers`: Simulation buffers
"""
function setup_simulation(
    params::SimulationParameters,
    n_particles::Int;
    μ_z::Float64=0.0,
    μ_E::Float64=0.0,
    σ_z0::Float64=0.005,
    σ_E0::Float64=1e6
)
    # Generate initial particles
    particles, _, _, _ = generate_particles(
        μ_z, μ_E, σ_z0, σ_E0, n_particles,
        params.E0, params.mass, params.ϕs, params.freq_rf
    )
    
    # Create buffers
    nbins = Int(n_particles/10)
    buffers = create_simulation_buffers(n_particles, nbins, Float64)
    
    # Calculate derived parameters
    γ0 = params.E0 / params.mass
    β0 = sqrt(1 - 1/γ0^2)
    
    # Create physics processes
    processes = PhysicsProcess[]
    
    # RF cavity
    push!(processes, create_rf_cavity(
        params.voltage, 
        params.ϕs, 
        params.freq_rf, 
        β0
    ))
    
    # Synchrotron radiation
    if params.SR_damping
        push!(processes, create_synchrotron_radiation(
            params.E0, 
            params.radius
        ))
    end
    
    # Quantum excitation
    if params.use_excitation
        push!(processes, create_quantum_excitation(
            params.E0, 
            params.radius, 
            σ_E0
        ))
    end
    
    # Wakefield
    if params.use_wakefield
        # Calculate current
        nbins::Int = next_power_of_two(Int(10^(ceil(Int, log10(n_particles)-2))))
        # bin_edges = range(-7.5*σ_z0, 7.5*σ_z0, length=nbins+1)
        η0 = params.α_c - 1/γ0^2
        current = (1e11/(10.0^floor(Int, log10(n_particles)))) * n_particles / 
                  params.E0 / (2*π*params.radius) * σ_z0 / (η0 * σ_E0^2)
        
        push!(processes, create_wakefield(
            params.pipe_radius, 
            σ_z0, 
            current, 
            nbins
        ))
    end
    
    return particles, processes, buffers
end