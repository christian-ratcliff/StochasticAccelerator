
include("../src/StochasticAccelerator.jl")
using .StochasticAccelerator
using Plots
using Statistics
using Random
using ProgressMeter


# Set random seed 
Random.seed!(12345)

"""
Run a basic simulation with default parameters.
"""
function run_basic_simulation()
    # println("Setting up simulation parameters...")

    # Define the machine parameters
    E0 = 4e9            # 4 GeV energy
    mass = MASS_ELECTRON
    voltage = 5e6       # 5 MV RF voltage
    harmonic = 360      # Harmonic number
    radius = 250.0      # Ring radius (m)
    pipe_radius = 0.00025 # Beam pipe radius (m)
    α_c = 3.68e-4       # Momentum compaction
    ϕs = 5π / 6           # Synchronous phase
    n_turns = 100       # Number of turns for simulation

    # Calculate derived parameters
    γ = E0 / mass
    β = sqrt(1 - 1 / γ^2)
    freq_rf = (ϕs + 10 * π / 180) * β * SPEED_LIGHT / (2π)

    # Create simulation parameters
    params = SimulationParameters(
        E0, mass, voltage, harmonic, radius, pipe_radius,
        α_c, ϕs, freq_rf, n_turns,
        true,  # use_wakefield
        true,  # update_η
        true,  # update_E0
        true,  # SR_damping
        true   # use_excitation
    )

    # Initial distribution parameters
    μ_z = 0.0
    μ_E = 0.0
    ω_rev = 2 * π / ((2 * π * radius) / (β * SPEED_LIGHT))
    σ_E0 = 1e6  # 1 MeV energy spread
    σ_z0 = sqrt(2 * π) * SPEED_LIGHT / ω_rev * sqrt(α_c * E0 / harmonic / voltage / abs(cos(ϕs))) * σ_E0 / E0

    println("Initial beam parameters:")
    println("  Energy: $E0 eV")
    println("  Energy spread: $σ_E0 eV")
    println("  Bunch length: $(σ_z0*1000) mm")

    # Setup simulation
    n_particles = 10000
    particles, processes, buffers = setup_simulation(
        params, n_particles;
        μ_z=μ_z, μ_E=μ_E, σ_z0=σ_z0, σ_E0=σ_E0
    )

    println("Running simulation with $n_particles particles for $n_turns turns...")

    # Run simulation
    p = Progress(n_turns, desc="Simulating Turns: ")
    for turn in 1:n_turns
        # Track particles for one turn
        σ_E, σ_z, E0 = track_particles!(
            particles,
            processes,
            params,
            buffers;
            update_η=params.update_η,
            update_E0=params.update_E0
        )

        # Print progress every 10 turns
        # if turn % 10 == 0 || turn == 1 || turn == n_turns
        #     println("Turn $turn: σ_E = $σ_E eV, σ_z = $(σ_z*1000) mm")
        # end
        next!(p)
    end

    # Analyze results
    results = analyze_beam(particles)

    println("\nFinal beam analysis:")
    println("  Mean z: $(results[:mean_z]*1000) mm")
    println("  Mean ΔE: $(results[:mean_ΔE]) eV")
    println("  Energy spread: $(results[:σ_E]) eV")
    println("  Bunch length: $(results[:σ_z]*1000) mm")
    println("  Correlation: $(results[:correlation])")
    println("  Emittance: $(results[:emittance]) eV·m")
    println("  z skewness: $(results[:z_skewness])")
    println("  ΔE skewness: $(results[:ΔE_skewness])")

    # Plot results
    plot_results(particles, results)

    return particles , results
end

"""
Plot the simulation results.
"""
function plot_results(particles, results)
    # Extract coordinates
    z_vals = particles.coordinates.z
    ΔE_vals = particles.coordinates.ΔE

    # Create phase space plot
    p1 = scatter(
        z_vals ./ results[:σ_z],
        ΔE_vals ./ results[:σ_E],
        title="Phase Space",
        xlabel="z/σ_z",
        ylabel="ΔE/σ_E",
        markersize=1,
        markerstrokewidth=0,
        alpha=0.5,
        label=nothing,
        aspect_ratio=:equal
    )

    # Create z distribution histogram
    p2 = histogram(
        z_vals,
        title="z Distribution",
        xlabel="z [m]",
        ylabel="Count",
        legend=false,
        bins=50,
        normalize=:pdf
    )

    # Create ΔE distribution histogram
    p3 = histogram(
        ΔE_vals,
        title="ΔE Distribution",
        xlabel="ΔE [eV]",
        ylabel="Count",
        legend=false,
        bins=50,
        normalize=:pdf
    )

    # Combine plots
    p = plot(p1, p2, p3, layout=(2, 2), size=(1000, 800))
    savefig(p, "basic_simulation_results.png")
    display(p)

    println("Plots saved to basic_simulation_results.png")
end

# Run the simulation
particles, results = run_basic_simulation()

using BenchmarkTools
using StructArrays

E0 = 4e9            # 4 GeV energy
mass = MASS_ELECTRON
voltage = 5e6       # 5 MV RF voltage
harmonic = 360      # Harmonic number
radius = 250.0      # Ring radius (m)
pipe_radius = 0.00025 # Beam pipe radius (m)
α_c = 3.68e-4       # Momentum compaction
ϕs = 5π / 6           # Synchronous phase
n_turns = 100       # Number of turns for simulation

# Calculate derived parameters
γ = E0 / mass
β = sqrt(1 - 1 / γ^2)
freq_rf = (ϕs + 10 * π / 180) * β * SPEED_LIGHT / (2π)

# Create simulation parameters
params = SimulationParameters(
    E0, mass, voltage, harmonic, radius, pipe_radius,
    α_c, ϕs, freq_rf, n_turns,
    true,  # use_wakefield
    true,  # update_η
    true,  # update_E0
    true,  # SR_damping
    true   # use_excitation
)

# Initial distribution parameters
μ_z = 0.0
μ_E = 0.0
ω_rev = 2 * π / ((2 * π * radius) / (β * SPEED_LIGHT))
σ_E0 = 1e6  # 1 MeV energy spread
σ_z0 = sqrt(2 * π) * SPEED_LIGHT / ω_rev * sqrt(α_c * E0 / harmonic / voltage / abs(cos(ϕs))) * σ_E0 / E0

println("Initial beam parameters:")
println("  Energy: $E0 eV")
println("  Energy spread: $σ_E0 eV")
println("  Bunch length: $(σ_z0*1000) mm")

# Setup simulation
n_particles = 10000
particles, processes, buffers = setup_simulation(
    params, n_particles;
    μ_z=μ_z, μ_E=μ_E, σ_z0=σ_z0, σ_E0=σ_E0
)

# typeof(processes)

function longitidunal_evolve(particles::StructArray{Particle{Float64}}, params::SimulationParameters, buffers::SimulationBuffers, processes::Vector{PhysicsProcess})
    # Run simulation
    p = Progress(n_turns, desc="Simulating Turns: ")
    for turn in 1:n_turns
        # Track particles for one turn
        σ_E, σ_z, E0 = track_particles!(
            particles,
            processes,
            params,
            buffers;
            update_η=params.update_η,
            update_E0=params.update_E0
        )

        # Print progress every 10 turns
        # if turn % 10 == 0 || turn == 1 || turn == n_turns
        #     println("Turn $turn: σ_E = $σ_E eV, σ_z = $(σ_z*1000) mm")
        # end
        next!(p)
    end
end

@btime longitidunal_evolve($particles, $params, $buffers, $processes)