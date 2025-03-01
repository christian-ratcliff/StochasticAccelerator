"""
Particle generation and management functions.

This file implements functions for creating and managing beam particles.
"""

using Distributions
using StructArrays
using StaticArrays
using Random
using StochasticAD

"""
    generate_particles(
        μ_z::T, μ_E::T, σ_z::T, σ_E::T, num_particles::Int,
        energy::TE, mass::TM, ϕs::TPS, freq_rf::TF
    ) where {T<:Float64, TE, TM, TPS, TF} -> Tuple{StructArray{Particle{T}}, T, T, TE}

Generate initial particle distribution.

# Arguments
- `μ_z`: Mean longitudinal position [m]
- `μ_E`: Mean energy deviation [eV]
- `σ_z`: RMS bunch length [m]
- `σ_E`: RMS energy spread [eV]
- `num_particles`: Number of particles
- `energy`: Reference energy [eV]
- `mass`: Particle mass [eV/c²]
- `ϕs`: Synchronous phase [rad]
- `freq_rf`: RF frequency [Hz]

# Returns
- `particles`: Particle distribution
- `σ_E`: Initial energy spread [eV]
- `σ_z`: Initial bunch length [m]
- `energy`: Reference energy [eV]
"""
function generate_particles(
    μ_z::T, μ_E::T, σ_z::T, σ_E::T, num_particles::Int,
    energy, mass, ϕs, freq_rf) where T<:Float64

    # Initial sampling for covariance estimation
    initial_sample_size::Int = min(10_000, num_particles)
    z_samples = rand(Normal(μ_z, σ_z), initial_sample_size)
    E_samples = rand(Normal(μ_E, σ_E), initial_sample_size)

    # Compute covariance matrix
    Σ = Symmetric([cov(z_samples, z_samples) cov(z_samples, E_samples);
                   cov(z_samples, E_samples) cov(E_samples, E_samples)])

    # Create multivariate normal distribution
    μ = SVector{2,T}(μ_z, μ_E)
    dist_total = MvNormal(μ, Σ)

    # Relativistic factors - use propagate if inputs might be StochasticTriple
    γ = StochasticAD.propagate((e, m) -> e / m, energy, mass)
    β = StochasticAD.propagate(g -> sqrt(1 - 1/g^2), γ)
    rf_factor = StochasticAD.propagate((freq, beta) -> freq * 2π / (beta * SPEED_LIGHT), freq_rf, β)

    # Generate correlated random samples
    samples = rand(dist_total, num_particles)  # 2 × num_particles matrix
    z_vals = samples[1, :]
    ΔE_vals = samples[2, :]

    # Create the StructArray of Particles
    particles = StructArray{Particle{Float64}}((
        StructArray(Coordinate.(z_vals, ΔE_vals)),  # coordinates
        StructArray(Coordinate.(zeros(num_particles), zeros(num_particles)))  # uncertainty
    ))

    return particles, σ_E, σ_z, energy
end

"""
    create_simulation_buffers(n_particles::Int, nbins::Int, T::Type=Float64) -> SimulationBuffers{T}

Create pre-allocated buffers for efficient simulation calculations.

# Arguments
- `n_particles`: Number of particles
- `nbins`: Number of bins for histogram calculations
- `T`: Number type (default: Float64)

# Returns
- `buffers`: Pre-allocated buffers
"""
function create_simulation_buffers(n_particles::Int, nbins::Int, T::Type=Float64)
    # Pre-allocate all vectors in parallel groups based on size
    particle_vectors = Vector{Vector{T}}(undef, 9)  # For n_particles sized vectors
    bin_vectors = Vector{Vector{T}}(undef, 2)       # For nbins sized vectors
    
    # Initialize n_particles sized vectors in parallel
    Threads.@threads for i in 1:9
        particle_vectors[i] = Vector{T}(undef, n_particles)
    end
    
    # Initialize nbins sized vectors in parallel
    Threads.@threads for i in 1:2
        bin_vectors[i] = Vector{T}(undef, nbins)
    end
    
    # Complex vector (single allocation)
    complex_vector = Vector{Complex{T}}(undef, nbins)
    
    # Random buffer
    random_buffer = Vector{T}(undef, n_particles)
    
    SimulationBuffers{T}(
        particle_vectors[1],   # WF
        particle_vectors[2],   # potential
        particle_vectors[3],   # Δγ
        particle_vectors[4],   # η
        particle_vectors[5],   # coeff
        particle_vectors[6],   # temp_z
        particle_vectors[7],   # temp_ΔE
        particle_vectors[8],   # temp_ϕ
        bin_vectors[1],        # WF_temp
        bin_vectors[2],        # λ
        complex_vector,        # convol
        particle_vectors[9],   # ϕ
        random_buffer          # random_buffer
    )
end

"""
    copyto_particles!(dst::StructArray{Particle{T}}, src::StructArray{Particle{T}}) where T<:Float64

Efficiently copy particle data without allocations.

# Arguments
- `dst`: Destination particle array
- `src`: Source particle array
"""
function copyto_particles!(dst::StructArray{Particle{T}}, src::StructArray{Particle{T}}) where T<:Float64
    @assert length(dst) == length(src)
    copyto!(dst.coordinates.z, src.coordinates.z)
    copyto!(dst.coordinates.ΔE, src.coordinates.ΔE)
    if hasproperty(dst, :uncertainty) && hasproperty(src, :uncertainty)
        copyto!(dst.uncertainty.z, src.uncertainty.z)
        copyto!(dst.uncertainty.ΔE, src.uncertainty.ΔE)
    end
    return dst
end