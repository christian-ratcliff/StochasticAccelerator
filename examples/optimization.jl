include("../src/StochasticAccelerator.jl")
begin
    using .StochasticAccelerator
    using Plots
    using Statistics
    using Random
    using StochasticAD
    using Optimisers
end

# Set random seed for reproducibility
Random.seed!(12345)

"""
Create default machine parameters.
"""
function create_default_parameters()
    # Define the machine parameters
    E0 = 4e9            # 4 GeV energy
    mass = MASS_ELECTRON
    voltage = 5e6       # 5 MV RF voltage
    harmonic = 360      # Harmonic number
    radius = 250.0      # Ring radius (m)
    pipe_radius = 0.00025 # Beam pipe radius (m)
    α_c = 3.68e-4       # Momentum compaction
    ϕs = 5π / 6           # Synchronous phase
    n_turns = 50        # Number of turns for simulation

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

    return params
end

"""
Optimize RF voltage and momentum compaction for minimum emittance.
"""
function optimize_emittance()
    println("Optimizing RF voltage and momentum compaction for minimum emittance...")

    # Create default parameters
    base_params = create_default_parameters()

    # Create parameter mapping
    param_mapping = Dict{Symbol,Int}(
        :voltage => 1,
        :α_c => 2
    )

    # Create model with reduced particle count for faster execution
    model = create_accelerator_model(
        base_params,
        param_mapping;
        n_particles=500
    )

    # Print initial parameter values
    println("Initial parameters:")
    println("  RF voltage: $(model.params[1]/1e6) MV")
    println("  Momentum compaction: $(model.params[2]*1e4) × 10⁻⁴")

    # Run model with initial parameters
    particles, σ_E, σ_z, E0 = run_model(model, model.params)
    initial_emittance = emittance_fom(particles, σ_E, σ_z, E0)
    println("Initial emittance: $(initial_emittance*1e6) MeV·mm")

    # Set up parameter constraints
    constraints = Dict(
        1 => (2e6, 8e6),      # Voltage: 2-8 MV
        2 => (1e-4, 1e-3)      # α_c: 1e-4 - 1e-3
    )

    # Define callback function for tracking optimization progress
    function callback(iteration, params, fom)
        if iteration % 5 == 0
            println("Iteration $iteration:")
            println("  RF voltage: $(params[1]/1e6) MV")
            println("  Momentum compaction: $(params[2]*1e4) × 10⁻⁴")
            println("  Emittance: $(fom*1e6) MeV·mm")
        end
    end

    # Run optimization
    optimal_params, fom_history, param_history = optimize_parameters(
        model,
        emittance_fom;
        iterations=30,
        optimizer=Adam(0.001),
        constraints=constraints,
        callback=callback
    )

    # Run model with optimal parameters
    particles, σ_E, σ_z, E0 = run_model(model, optimal_params)
    final_emittance = emittance_fom(particles, σ_E, σ_z, E0)

    # Print results
    println("\nOptimization results:")
    println("  Initial emittance: $(initial_emittance*1e6) MeV·mm")
    println("  Final emittance: $(final_emittance*1e6) MeV·mm")
    println("  Improvement: $(round((1 - final_emittance/initial_emittance) * 100, digits=2))%")
    println("  Optimal RF voltage: $(optimal_params[1]/1e6) MV")
    println("  Optimal momentum compaction: $(optimal_params[2]*1e4) × 10⁻⁴")

    # Plot optimization progress
    plot_optimization_results(fom_history, param_history)

    # Plot final beam distribution
    plot_final_beam(particles)

    return optimal_params, fom_history, param_history
end

"""
Optimize RF voltage and phase for stable beam.
"""
function optimize_stability()
    println("Optimizing RF voltage and phase for beam stability...")

    # Create default parameters
    base_params = create_default_parameters()

    # Create parameter mapping
    param_mapping = Dict{Symbol,Int}(
        :voltage => 1,
        :ϕs => 2
    )

    # Create model with reduced particle count for faster execution
    model = create_accelerator_model(
        base_params,
        param_mapping;
        n_particles=500
    )

    # Define reference values for stability
    ref_σ_E = 1.2e6  # 1.2 MeV energy spread
    ref_σ_z = 0.004  # 4 mm bunch length

    # Define stability figure of merit
    stability_fn(particles, σ_E, σ_z, E0) = stability_fom(
        particles, σ_E, σ_z, E0;
        ref_σ_E=ref_σ_E, ref_σ_z=ref_σ_z
    )

    # Print initial parameter values
    println("Initial parameters:")
    println("  RF voltage: $(model.params[1]/1e6) MV")
    println("  RF phase: $(model.params[2]) rad")

    # Run model with initial parameters
    particles, σ_E, σ_z, E0 = run_model(model, model.params)
    initial_stability = stability_fn(particles, σ_E, σ_z, E0)
    println("Initial stability measure: $initial_stability")
    println("  Energy spread: $σ_E eV (reference: $ref_σ_E eV)")
    println("  Bunch length: $(σ_z*1000) mm (reference: $(ref_σ_z*1000) mm)")

    # Set up parameter constraints
    constraints = Dict(
        1 => (2e6, 8e6),      # Voltage: 2-8 MV
        2 => (π / 2, π)         # Phase: π/2 - π rad
    )

    # Run optimization
    optimal_params, fom_history, param_history = optimize_parameters(
        model,
        stability_fn;
        iterations=30,
        optimizer=Adam(0.0005),
        constraints=constraints
    )

    # Run model with optimal parameters
    particles, σ_E, σ_z, E0 = run_model(model, optimal_params)
    final_stability = stability_fn(particles, σ_E, σ_z, E0)

    # Print results
    println("\nOptimization results:")
    println("  Initial stability measure: $initial_stability")
    println("  Final stability measure: $final_stability")
    println("  Improvement: $(round((1 - final_stability/initial_stability) * 100, digits=2))%")
    println("  Optimal RF voltage: $(optimal_params[1]/1e6) MV")
    println("  Optimal RF phase: $(optimal_params[2]) rad")
    println("  Final energy spread: $σ_E eV (reference: $ref_σ_E eV)")
    println("  Final bunch length: $(σ_z*1000) mm (reference: $(ref_σ_z*1000) mm)")

    # Plot optimization progress
    plot_stability_optimization(fom_history, param_history, ref_σ_E, ref_σ_z)

    return optimal_params, fom_history, param_history
end

"""
Plot optimization results.
"""
function plot_optimization_results(fom_history, param_history)
    # Extract parameter history
    voltage_history = [params[1] / 1e6 for params in param_history]  # MV
    α_c_history = [params[2] * 1e4 for params in param_history]      # 10⁻⁴

    # Plot emittance history
    p1 = plot(
        1:length(fom_history),
        fom_history .* 1e6,  # Convert to MeV·mm
        title="Emittance vs. Iteration",
        xlabel="Iteration",
        ylabel="Emittance [MeV·mm]",
        legend=false,
        linewidth=2
    )

    # Plot voltage history
    p2 = plot(
        1:length(voltage_history),
        voltage_history,
        title="RF Voltage vs. Iteration",
        xlabel="Iteration",
        ylabel="RF Voltage [MV]",
        legend=false,
        linewidth=2
    )

    # Plot momentum compaction history
    p3 = plot(
        1:length(α_c_history),
        α_c_history,
        title="Momentum Compaction vs. Iteration",
        xlabel="Iteration",
        ylabel="Momentum Compaction [10⁻⁴]",
        legend=false,
        linewidth=2
    )

    # Combine plots
    p = plot(p1, p2, p3, layout=(3, 1), size=(800, 900))
    savefig(p, "optimization_results.png")
    display(p)

    println("Plots saved to optimization_results.png")
end

"""
Plot stability optimization results.
"""
function plot_stability_optimization(fom_history, param_history, ref_σ_E, ref_σ_z)
    # Extract parameter history
    voltage_history = [params[1] / 1e6 for params in param_history]  # MV
    phase_history = [params[2] for params in param_history]        # rad

    # Plot stability measure history
    p1 = plot(
        1:length(fom_history),
        fom_history,
        title="Stability Measure vs. Iteration",
        xlabel="Iteration",
        ylabel="Stability Measure",
        legend=false,
        linewidth=2
    )

    # Plot voltage history
    p2 = plot(
        1:length(voltage_history),
        voltage_history,
        title="RF Voltage vs. Iteration",
        xlabel="Iteration",
        ylabel="RF Voltage [MV]",
        legend=false,
        linewidth=2
    )

    # Plot phase history
    p3 = plot(
        1:length(phase_history),
        phase_history,
        title="RF Phase vs. Iteration",
        xlabel="Iteration",
        ylabel="RF Phase [rad]",
        legend=false,
        linewidth=2
    )

    # Combine plots
    p = plot(p1, p2, p3, layout=(3, 1), size=(800, 900))
    savefig(p, "stability_optimization.png")
    display(p)

    println("Plots saved to stability_optimization.png")
end

"""
Plot the final beam distribution.
"""
function plot_final_beam(particles)
    # Extract coordinates
    z_vals = particles.coordinates.z
    ΔE_vals = particles.coordinates.ΔE

    # Calculate statistics
    σ_z = std(z_vals)
    σ_E = std(ΔE_vals)
    correlation = cor(z_vals, ΔE_vals)

    # Create phase space plot
    p = scatter(
        z_vals ./ σ_z,
        ΔE_vals ./ σ_E,
        title="Optimized Beam Distribution",
        xlabel="z/σ_z",
        ylabel="ΔE/σ_E",
        markersize=1,
        markerstrokewidth=0,
        alpha=0.5,
        label=nothing,
        aspect_ratio=:equal
    )

    # Add correlation information
    annotate!(0, -2, text("σ_z = $(round(σ_z*1000, digits=2)) mm", 10))
    annotate!(0, -2.3, text("σ_E = $(round(σ_E/1e3, digits=2)) keV", 10))
    annotate!(0, -2.6, text("r = $(round(correlation, digits=3))", 10))

    savefig(p, "optimized_beam.png")
    display(p)

    println("Plot saved to optimized_beam.png")
end

# Run the emittance optimization
optimal_params, fom_history, param_history = optimize_emittance()

# Run the stability optimization
stability_params, stability_fom_history, stability_param_history = optimize_stability()