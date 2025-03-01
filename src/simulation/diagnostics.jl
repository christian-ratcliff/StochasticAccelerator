"""
Beam diagnostics for analyzing simulation results.

This file implements beam analysis functions for extracting
statistics and quantities of interest from the simulation.
"""

using Statistics
using StructArrays

"""
    analyze_beam(
        particles::StructArray{Particle{T}}
    ) where T<:Float64 -> Dict{Symbol, Any}

Analyze a beam distribution for key parameters.

# Arguments
- `particles`: Particle distribution

# Returns
- Dictionary of beam parameters
"""
function analyze_beam(
    particles::StructArray{Particle{T}}
) where T<:Float64
    
    # Extract coordinates
    z_vals = particles.coordinates.z
    ΔE_vals = particles.coordinates.ΔE
    
    # Calculate basic statistics
    mean_z = mean(z_vals)
    mean_ΔE = mean(ΔE_vals)
    σ_z = std(z_vals)
    σ_E = std(ΔE_vals)
    
    # Calculate correlation
    n = length(z_vals)
    z_centered = z_vals .- mean_z
    ΔE_centered = ΔE_vals .- mean_ΔE
    correlation = sum(z_centered .* ΔE_centered) / (n * σ_z * σ_E)
    
    # Calculate emittance
    emittance = σ_z * σ_E * sqrt(1 - correlation^2)
    
    # Calculate skewness
    z_skewness = sum((z_centered ./ σ_z).^3) / n
    ΔE_skewness = sum((ΔE_centered ./ σ_E).^3) / n
    
    # Calculate kurtosis
    z_kurtosis = sum((z_centered ./ σ_z).^4) / n - 3
    ΔE_kurtosis = sum((ΔE_centered ./ σ_E).^4) / n - 3
    
    # Return results
    return Dict(
        :mean_z => mean_z,
        :mean_ΔE => mean_ΔE,
        :σ_z => σ_z,
        :σ_E => σ_E,
        :correlation => correlation,
        :emittance => emittance,
        :z_skewness => z_skewness,
        :ΔE_skewness => ΔE_skewness,
        :z_kurtosis => z_kurtosis,
        :ΔE_kurtosis => ΔE_kurtosis
    )
end

"""
    calculate_statistics(
        particles::StructArray{Particle{T}}
    ) where T<:Float64 -> Tuple{T, T, T}

Calculate basic beam statistics.

# Arguments
- `particles`: Particle distribution

# Returns
- σ_E: Energy spread [eV]
- σ_z: Bunch length [m]
- emittance: Longitudinal emittance [eV·m]
"""
function calculate_statistics(
    particles::StructArray{Particle{T}}
) where T<:Float64
    
    # Calculate statistics
    σ_E = std(particles.coordinates.ΔE)
    σ_z = std(particles.coordinates.z)
    
    # Calculate correlation
    correlation = cor(particles.coordinates.z, particles.coordinates.ΔE)
    
    # Calculate emittance
    emittance = σ_E * σ_z * sqrt(1 - correlation^2)
    
    return σ_E, σ_z, emittance
end

"""
    calculate_distribution_at_position(
        particles::StructArray{Particle{T}},
        position::T;
        nbins::Int=100,
        z_range::Float64=5.0,
        ΔE_range::Float64=5.0
    ) where T<:Float64 -> Tuple{Vector{T}, Vector{T}, Matrix{T}}

Calculate 2D phase space distribution at a specified position.

# Arguments
- `particles`: Particle distribution
- `position`: Longitudinal position [m]
- `nbins`: Number of bins in each dimension
- `z_range`: Range in z in units of σ_z
- `ΔE_range`: Range in ΔE in units of σ_E

# Returns
- z_bins: Bin centers for z axis
- ΔE_bins: Bin centers for ΔE axis
- distribution: 2D histogram of particle density
"""
function calculate_distribution_at_position(
    particles::StructArray{Particle{T}},
    position::T;
    nbins::Int=100,
    z_range::Float64=5.0,
    ΔE_range::Float64=5.0
) where T<:Float64
    
    # Calculate statistics
    σ_E = std(particles.coordinates.ΔE)
    σ_z = std(particles.coordinates.z)
    mean_z = mean(particles.coordinates.z)
    mean_ΔE = mean(particles.coordinates.ΔE)
    
    # Create bins
    z_bins = range(mean_z - z_range*σ_z, mean_z + z_range*σ_z, length=nbins)
    ΔE_bins = range(mean_ΔE - ΔE_range*σ_E, mean_ΔE + ΔE_range*σ_E, length=nbins)
    
    # Initialize histogram
    distribution = zeros(nbins, nbins)
    
    # Calculate bin widths
    z_bin_width = (z_bins[end] - z_bins[1]) / (nbins - 1)
    ΔE_bin_width = (ΔE_bins[end] - ΔE_bins[1]) / (nbins - 1)
    
    # Fill histogram
    for i in 1:length(particles)
        z = particles.coordinates.z[i]
        ΔE = particles.coordinates.ΔE[i]
        
        # Find bin indices
        z_idx = floor(Int, (z - z_bins[1]) / z_bin_width) + 1
        ΔE_idx = floor(Int, (ΔE - ΔE_bins[1]) / ΔE_bin_width) + 1
        
        # Check if in range
        if 1 <= z_idx <= nbins && 1 <= ΔE_idx <= nbins
            distribution[z_idx, ΔE_idx] += 1
        end
    end
    
    # Normalize
    distribution ./= sum(distribution) * z_bin_width * ΔE_bin_width
    
    return collect(z_bins), collect(ΔE_bins), distribution
end