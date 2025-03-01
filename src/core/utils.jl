"""
Utility functions for beam simulations.

This file contains common utility functions used throughout the simulation:
- Field copying and assignment
- FFT and convolution operations
- Histogram calculations
- Phase space transformations
"""

using LoopVectorization
using FFTW
using FHist
using Interpolations
using StochasticAD

"""
    z_to_ϕ(z_val, rf_factor, ϕs) -> Any

Convert longitudinal position to RF phase.
Compatible with StochasticTriple.

# Arguments
- `z_val`: Longitudinal position [m]
- `rf_factor`: RF factor [rad/m]
- `ϕs`: Synchronous phase [rad]

# Returns
- Phase [rad]
"""
function z_to_ϕ(z_val, rf_factor, ϕs)
    return -(z_val * rf_factor - ϕs)
end

"""
    ϕ_to_z(ϕ_val, rf_factor, ϕs) -> Any

Convert RF phase to longitudinal position.
Compatible with StochasticTriple.

# Arguments
- `ϕ_val`: Phase [rad]
- `rf_factor`: RF factor [rad/m]
- `ϕs`: Synchronous phase [rad]

# Returns
- Longitudinal position [m]
"""
function ϕ_to_z(ϕ_val, rf_factor, ϕs)
    return (-ϕ_val + ϕs) / rf_factor
end

"""
    calc_rf_factor(freq_rf, β) -> Any

Calculate RF factor from RF frequency and relativistic beta.

# Arguments
- `freq_rf`: RF frequency [Hz]
- `β`: Relativistic beta

# Returns
- RF factor [rad/m]
"""
function calc_rf_factor(freq_rf, β)
    return StochasticAD.propagate(
        (f, beta) -> f * 2π / (beta * SPEED_LIGHT),
        freq_rf, β
    )
end

"""
    delta(x::T, σ::T) where T<:Float64 -> T

Calculate a Gaussian delta function for beam distribution smoothing.

# Arguments
- `x`: Position [m]
- `σ`: Standard deviation [m]

# Returns
- Gaussian function value
"""
function delta(x::T, σ::T)::T where T<:Float64
    σ_inv = 1 / (sqrt(2 * π) * σ)
    exp_factor = -0.5 / (σ^2)
    return σ_inv * exp(x^2 * exp_factor)
end

"""
    FastConv1D(f::AbstractVector{T}, g::AbstractVector{T}) where T -> Vector{Complex{T}}

Compute the fast convolution of two vectors using FFT.

# Arguments
- `f`: First vector
- `g`: Second vector

# Returns
- Convolution result
"""
function FastConv1D(f::AbstractVector{T}, g::AbstractVector{T})::Vector{Complex{T}} where T<:Float64
    return ifft(fft(f) .* fft(g))
end

"""
    FastLinearConvolution(f::AbstractVector{T}, g::AbstractVector{T}, power_2_length::Int) where T

Compute linear convolution with automatic padding to power of 2 length.

# Arguments
- `f`: First vector
- `g`: Second vector
- `power_2_length`: Length for FFT (power of 2)

# Returns
- Convolution result
"""
function FastLinearConvolution(f::AbstractVector{T}, g::AbstractVector{T}, power_2_length::Int64)::Vector{Complex{T}} where T<:Float64
    pad_and_ensure_power_of_two!(f, g, power_2_length)
    return FastConv1D(f, g)
end

"""
    is_power_of_two(n::Int) -> Bool

Check if a number is a power of two using bitwise operations.

# Arguments
- `n`: Number to check

# Returns
- `true` if n is a power of two, `false` otherwise
"""
function is_power_of_two(n::Int64)::Bool
    return (n & (n - 1)) == 0 && n > 0
end

"""
    next_power_of_two(n::Int) -> Int

Find the next power of two greater than or equal to n.

# Arguments
- `n`: Input number

# Returns
- Next power of two
"""
function next_power_of_two(n::Int64)::Int64
    return Int64(2^(ceil(log2(n))))
end

"""
    pad_and_ensure_power_of_two!(f::AbstractVector{T}, g::AbstractVector{T}, power_two_length::Int) where T -> Nothing

Pad vectors to power-of-two length for efficient FFT operations.

# Arguments
- `f`: First vector to pad
- `g`: Second vector to pad
- `power_two_length`: Target length (power of 2)
"""
function pad_and_ensure_power_of_two!(f::AbstractVector{T}, g::AbstractVector{T}, power_two_length::Int) where T<:Float64
    N::Int64 = length(f)
    M::Int64 = length(g)
    
    original_f = copy(f)
    resize!(f, power_two_length)
    f[1:N] = original_f
    f[N+1:end] .= zero(T)
    
    original_g = copy(g)
    resize!(g, power_two_length)
    g[1:M] = original_g
    g[M+1:end] .= zero(T)
    
    return nothing
end

"""
    calculate_histogram(data::Vector{T}, bins_edges) -> Tuple{Vector{T}, Vector{Int}}

Calculate histogram of data with specified bin edges.

# Arguments
- `data`: Data vector
- `bins_edges`: Bin edges

# Returns
- Tuple of bin centers and bin counts
"""
function calculate_histogram(data::Vector{T}, bins_edges) where T<:Float64
    histo = Hist1D(data, binedges=bins_edges)
    centers = (histo.binedges[1][1:end-1] + histo.binedges[1][2:end]) ./ 2
    return collect(centers), histo.bincounts
end

"""
    threaded_fieldwise_copy!(destination, source)

Copy particle fields from source to destination in a thread-safe manner.

# Arguments
- `destination`: Destination StructArray
- `source`: Source StructArray
"""
function threaded_fieldwise_copy!(destination::StructArray{Particle{T}}, source::StructArray{Particle{T}}) where T<:Float64
    @assert length(destination) == length(source)
    Threads.@threads for i in 1:length(source)
        destination[i] = Particle(
            Coordinate(source.coordinates.z[i], source.coordinates.ΔE[i]),
            Coordinate(source.uncertainty.z[i], source.uncertainty.ΔE[i])
        )
    end
end

"""
    assign_to_turn!(particle_trajectory, particle_states, turn)

Assign current particle states to the specified turn in the trajectory.

# Arguments
- `particle_trajectory`: Trajectory to update
- `particle_states`: Current particle states
- `turn`: Turn number
"""
# function assign_to_turn!(particle_trajectory::BeamTurn{T}, particle_states::StructArray{Particle{T}}, turn::Integer) where T<:Float64
#     threaded_fieldwise_copy!(particle_trajectory.states[turn], particle_states)
# end