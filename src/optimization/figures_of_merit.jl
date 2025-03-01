"""
Figures of merit for accelerator optimization.

This file implements various figures of merit (objective functions)
for optimizing the accelerator parameters.
"""

using StructArrays
using Statistics

"""
    energy_spread_fom(particles, σ_E, σ_z, E0) -> Float64

Energy spread figure of merit.

# Arguments
- `particles`: Particle distribution
- `σ_E`: Energy spread [eV]
- `σ_z`: Bunch length [m]
- `E0`: Reference energy [eV]

# Returns
- Energy spread [eV]
"""
function energy_spread_fom(particles, σ_E, σ_z, E0)
    return σ_E
end

"""
    bunch_length_fom(particles, σ_E, σ_z, E0) -> Float64

Bunch length figure of merit.

# Arguments
- `particles`: Particle distribution
- `σ_E`: Energy spread [eV]
- `σ_z`: Bunch length [m]
- `E0`: Reference energy [eV]

# Returns
- Bunch length [m]
"""
function bunch_length_fom(particles, σ_E, σ_z, E0)
    return σ_z
end

"""
    emittance_fom(particles, σ_E, σ_z, E0) -> Float64

Emittance figure of merit.

# Arguments
- `particles`: Particle distribution
- `σ_E`: Energy spread [eV]
- `σ_z`: Bunch length [m]
- `E0`: Reference energy [eV]

# Returns
- Emittance [eV·m]
"""
function emittance_fom(particles, σ_E, σ_z, E0)
    # Calculate correlation
    correlation = cor(particles.coordinates.z, particles.coordinates.ΔE)
    
    # Calculate emittance
    emittance = σ_E * σ_z * sqrt(1 - correlation^2)
    
    return emittance
end

"""
    weighted_emittance_fom(particles, σ_E, σ_z, E0; w_E::Float64=1.0, w_z::Float64=1.0) -> Float64

Weighted emittance figure of merit.

# Arguments
- `particles`: Particle distribution
- `σ_E`: Energy spread [eV]
- `σ_z`: Bunch length [m]
- `E0`: Reference energy [eV]
- `w_E`: Weight for energy spread
- `w_z`: Weight for bunch length

# Returns
- Weighted emittance
"""
function weighted_emittance_fom(particles, σ_E, σ_z, E0; w_E::Float64=1.0, w_z::Float64=1.0)
    # Weight the energy spread and bunch length
    return (σ_E * w_E) * (σ_z * w_z)
end

"""
    bunch_profile_fom(
        particles, σ_E, σ_z, E0;
        target_profile::Function=(z,σ) -> exp(-0.5*(z/σ)^2)/sqrt(2π*σ^2)
    ) -> Float64

Bunch profile figure of merit measuring deviation from target profile.

# Arguments
- `particles`: Particle distribution
- `σ_E`: Energy spread [eV]
- `σ_z`: Bunch length [m]
- `E0`: Reference energy [eV]
- `target_profile`: Target profile function

# Returns
- Profile deviation (lower is better)
"""
function bunch_profile_fom(
    particles, σ_E, σ_z, E0;
    target_profile::Function=(z,σ) -> exp(-0.5*(z/σ)^2)/sqrt(2π*σ^2)
)
    # Get z coordinates
    z_vals = particles.coordinates.z
    
    # Create histogram
    nbins = 100
    z_range = range(minimum(z_vals), maximum(z_vals), length=nbins)
    hist = fit(Histogram, z_vals, z_range)
    counts = hist.weights
    
    # Normalize histogram
    bin_width = step(z_range)
    density = counts ./ (sum(counts) * bin_width)
    
    # Calculate target profile
    z_centers = (z_range[1:end-1] + z_range[2:end]) ./ 2
    target = [target_profile(z - mean(z_vals), σ_z) for z in z_centers]
    
    # Calculate RMS deviation
    deviation = sqrt(mean((density .- target).^2))
    
    return deviation
end

"""
    stability_fom(particles, σ_E, σ_z, E0; ref_σ_E::Float64, ref_σ_z::Float64) -> Float64

Stability figure of merit measuring closeness to reference values.

# Arguments
- `particles`: Particle distribution
- `σ_E`: Energy spread [eV]
- `σ_z`: Bunch length [m]
- `E0`: Reference energy [eV]
- `ref_σ_E`: Reference energy spread [eV]
- `ref_σ_z`: Reference bunch length [m]

# Returns
- Stability measure (lower is better)
"""
function stability_fom(
    particles, σ_E, σ_z, E0;
    ref_σ_E::Float64, ref_σ_z::Float64
)
    # Calculate relative deviations
    rel_dev_E = (σ_E - ref_σ_E) / ref_σ_E
    rel_dev_z = (σ_z - ref_σ_z) / ref_σ_z
    
    # Calculate RMS relative deviation
    deviation = sqrt(rel_dev_E^2 + rel_dev_z^2)
    
    return deviation
end