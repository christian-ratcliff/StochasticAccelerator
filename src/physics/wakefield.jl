"""
Wakefield effects for beam dynamics.

This file implements the wakefield calculations and collective effects
that couple particles within the beam, including:
- Wake function calculation
- Convolution with charge distribution
- Application of wakefield forces to particles
"""

using LoopVectorization
using FFTW
using Interpolations
using FHist
using StochasticAD
using StructArrays

"""
    Wakefield <: PhysicsProcess

Wakefield process that applies collective effects to particles.

# Fields
- `wake_factor`: Wake function amplitude factor
- `wake_sqrt`: Wake function frequency parameter
- `cτ`: Characteristic length [m]
- `current`: Beam current [A]
- `σ_z`: RMS bunch length [m]
- `bin_edges`: Histogram bin edges
"""
struct Wakefield <: PhysicsProcess
    wake_factor::Any  # Can be Float64 or StochasticTriple
    wake_sqrt::Any    # Can be Float64 or StochasticTriple
    cτ::Float64
    current::Any      # Can be Float64 or StochasticTriple
    σ_z::Float64
    bin_edges::Any
end

"""
    apply_process!(
        process::Wakefield, 
        particles::StructArray{Particle{T}},
        params::SimulationParameters,
        buffers::SimulationBuffers{T}
    ) where T<:Float64 -> Nothing

Apply wakefield effects to all particles.

# Arguments
- `process`: Wakefield process
- `particles`: Particle array
- `params`: Simulation parameters
- `buffers`: Simulation buffers
"""
function apply_process!(
    process::Wakefield, 
    particles::StructArray{Particle{T}},
    params::SimulationParameters,
    buffers::SimulationBuffers{S}
    ) where {T<:Float64, S} 
    
    # Get process parameters
    wake_factor = process.wake_factor
    wake_sqrt = process.wake_sqrt
    cτ = process.cτ
    current = process.current
    σ_z = process.σ_z
    bin_edges = process.bin_edges
    
    # Clear buffers
    fill!(buffers.λ, zero(T))
    fill!(buffers.WF_temp, zero(T))
    fill!(buffers.convol, zero(Complex{T}))
    
    # Calculate inverse characteristic length
    inv_cτ::T = 1 / cτ
    
    # Calculate histogram
    bin_centers::Vector{T}, bin_amounts::Vector{T} = calculate_histogram(particles.coordinates.z, bin_edges)
    nbins::Int = length(bin_centers)
    power_2_length::Int = nbins * 2
    
    # Calculate line charge density using Gaussian smoothing
    delta_std::T = (15 * σ_z) / σ_z / 100
    @inbounds for i in eachindex(bin_centers)
        buffers.λ[i] = delta(bin_centers[i], delta_std)
    end
    
    # Check if any parameter is a StochasticTriple
    is_stochastic = any(p -> typeof(p) <: StochasticTriple, [wake_factor, wake_sqrt, current])
    
    if is_stochastic
        # Calculate wake function with proper gradient propagation
        StochasticAD.propagate(wake_factor, wake_sqrt) do wf, ws
            @inbounds for i in eachindex(bin_centers)
                z = bin_centers[i]
                buffers.WF_temp[i] = calculate_wake_function(z, wf, ws, inv_cτ)
            end
            return nothing
        end
    else
        # Standard wake function calculation
        @inbounds for i in eachindex(bin_centers)
            z = bin_centers[i]
            buffers.WF_temp[i] = calculate_wake_function(z, wake_factor, wake_sqrt, inv_cτ)
        end
    end

    # Prepare arrays for convolution
    normalized_amounts = bin_amounts .* (1/length(particles))
    λ = buffers.λ[1:nbins]
    WF_temp = buffers.WF_temp[1:nbins]
    convol = buffers.convol[1:power_2_length]
    
    # Perform convolution
    convol_result = FastLinearConvolution(WF_temp, λ .* normalized_amounts, power_2_length)
    
    # Scale by current - handle StochasticTriple properly
    if typeof(current) <: StochasticTriple
        convol_scaled = StochasticAD.propagate(c -> convol_result .* c, current)
        copyto!(convol, convol_scaled)
    else
        convol .= convol_result .* current
    end
    
    # Interpolate results back to particle positions
    temp_z = range(minimum(particles.coordinates.z), maximum(particles.coordinates.z), length=length(convol))
    resize!(buffers.potential, length(particles.coordinates.z))
    
    # Create an interpolation function
    itp = LinearInterpolation(temp_z, real.(convol), extrapolation_bc=Line())
    
    # Apply the interpolated potential to particles
    if is_stochastic
        # For StochasticTriple, use propagate to ensure proper gradient propagation
        for i in eachindex(particles.coordinates.z)
            z = particles.coordinates.z[i]
            potential_value = itp(z)
            particles.coordinates.ΔE[i] = StochasticAD.propagate(
                (de, pot) -> de - pot,
                particles.coordinates.ΔE[i],
                potential_value
            )
        end
    else
        # Standard application
        @inbounds for i in eachindex(particles.coordinates.z)
            z = particles.coordinates.z[i]
            potential_value = itp(z)
            particles.coordinates.ΔE[i] -= potential_value
        end
    end
    
    return nothing
end

"""
    calculate_wake_function(z::T, wake_factor::T, wake_sqrt::T, inv_cτ::T) where T<:Float64 -> T
    calculate_wake_function(z::T, wake_factor, wake_sqrt, inv_cτ::T) where T<:Float64 -> Any

Calculate the wake function for a given longitudinal position.
Handles both standard floating-point and StochasticTriple inputs.

# Arguments
- `z`: Longitudinal position [m]
- `wake_factor`: Wake function amplitude factor
- `wake_sqrt`: Wake function frequency parameter
- `inv_cτ`: Inverse characteristic length [1/m]

# Returns
- Wake function value
"""
function calculate_wake_function(z::T, wake_factor::T, wake_sqrt::T, inv_cτ::T) where T<:Float64
    return z > 0 ? zero(T) : wake_factor * exp(z * inv_cτ) * cos(wake_sqrt * z)
end

# Overload for StochasticTriple
function calculate_wake_function(z::T, wake_factor, wake_sqrt, inv_cτ::T) where T<:Float64
    return StochasticAD.propagate(
        (wf, ws) -> z > 0 ? zero(T) : wf * exp(z * inv_cτ) * cos(ws * z),
        wake_factor, wake_sqrt
    )
end

"""
    create_wakefield(
        pipe_radius,
        σ_z::Float64,
        current, 
        nbins::Int
    ) -> Wakefield

Create a wakefield process.

# Arguments
- `pipe_radius`: Beam pipe radius [m]
- `σ_z`: RMS bunch length [m]
- `current`: Beam current [A]
- `nbins`: Number of bins for histogram

# Returns
- Wakefield process
"""
function create_wakefield(pipe_radius, σ_z::Float64, current, nbins::Int)
    # Physical parameters
    Z0 = 120π  # Impedance of free space [Ω]
    kp = 3e1   # Plasma wavenumber [1/m]
    cτ = 4e-3  # Characteristic length [m]
    
    # Calculate wake parameters
    wake_factor = StochasticAD.propagate(r -> Z0 * SPEED_LIGHT / (π * r), pipe_radius)
    wake_sqrt = StochasticAD.propagate(r -> sqrt(2 * kp / r), pipe_radius)
    
    # Create bin edges
    bin_edges = range(-7.5*σ_z, 7.5*σ_z, length=nbins+1)
    
    return Wakefield(wake_factor, wake_sqrt, cτ, current, σ_z, bin_edges)
end