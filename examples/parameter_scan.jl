include("../src/StochasticAccelerator.jl")

using .StochasticAccelerator
using Plots
using Statistics
using Random
using StochasticAD

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
Scan RF voltage and analyze its effect on beam parameters.
"""
function scan_rf_voltage()
    println("Scanning RF voltage...")

    # Create default parameters
    base_params = create_default_parameters()

    # Create parameter mapping
    param_mapping = Dict{Symbol,Int}(
        :voltage => 1
    )

    # Create model with reduced particle count for faster execution
    model = create_accelerator_model(
        base_params,
        param_mapping;
        n_particles=1000
    )

    # Define voltage range
    voltage_range = range(3e6, 7e6, length=10)

    # Perform scan
    println("Running voltage scan with energy spread figure of merit...")
    voltage_values, energy_spread_values, grad_energy, grad_uncertainty = scan_parameter(
        model,
        energy_spread_fom,
        1,  # Voltage index
        voltage_range;
        calculate_gradients=true,
        n_gradient_samples=5
    )

    # Perform scan for bunch length
    println("Running voltage scan with bunch length figure of merit...")
    _, bunch_length_values, grad_length, grad_length_uncertainty = scan_parameter(
        model,
        bunch_length_fom,
        1,  # Voltage index
        voltage_range;
        calculate_gradients=true,
        n_gradient_samples=5
    )

    # Plot results
    plot_voltage_scan_results(
        voltage_values,
        energy_spread_values,
        bunch_length_values,
        grad_energy,
        grad_length,
        grad_uncertainty,
        grad_length_uncertainty
    )

    return voltage_values, energy_spread_values, bunch_length_values, grad_energy, grad_length
end

"""
Scan momentum compaction factor and RF voltage together.
"""
function scan_momentum_compaction_and_voltage()
    println("Scanning momentum compaction and RF voltage...")

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

    # Define parameter ranges
    voltage_range = range(3e6, 7e6, length=5)
    α_c_range = range(1e-4, 5e-4, length=5)

    # Perform scan
    println("Running 2D parameter scan with emittance figure of merit...")
    param_grid, emittance_grid = multi_parameter_scan(
        model,
        emittance_fom,
        [1, 2],  # Indices for voltage and α_c
        [voltage_range, α_c_range]
    )

    # Plot results
    plot_2d_scan_results(param_grid, emittance_grid)

    return param_grid, emittance_grid
end

"""
Plot the results of the RF voltage scan.
"""
function plot_voltage_scan_results(
    voltage_values,
    energy_spread_values,
    bunch_length_values,
    grad_energy,
    grad_length,
    grad_uncertainty,
    grad_length_uncertainty
)
    # Convert to more readable units
    voltage_MV = voltage_values ./ 1e6
    energy_spread_MeV = energy_spread_values ./ 1e6
    bunch_length_mm = bunch_length_values .* 1e3

    # Scale gradients for plotting
    grad_energy_scaled = grad_energy ./ 1e6 .* 1e6  # MeV/MV
    grad_length_scaled = grad_length .* 1e3 ./ 1e6  # mm/MV
    grad_uncertainty_scaled = grad_uncertainty ./ 1e6 .* 1e6
    grad_length_uncertainty_scaled = grad_length_uncertainty .* 1e3 ./ 1e6

    # Energy spread plot
    p1 = plot(
        voltage_MV,
        energy_spread_MeV,
        title="Energy Spread vs. RF Voltage",
        xlabel="RF Voltage [MV]",
        ylabel="Energy Spread [MeV]",
        legend=false,
        marker=:circle,
        linewidth=2
    )

    # Energy spread gradient plot
    p2 = plot(
        voltage_MV,
        grad_energy_scaled,
        title="d(Energy Spread)/d(Voltage)",
        xlabel="RF Voltage [MV]",
        ylabel="Gradient [MeV/MV]",
        legend=false,
        marker=:circle,
        ribbon=grad_uncertainty_scaled,
        linewidth=2
    )
    # Add zero line
    hline!(p2, [0.0], linestyle=:dash, color=:black)

    # Bunch length plot
    p3 = plot(
        voltage_MV,
        bunch_length_mm,
        title="Bunch Length vs. RF Voltage",
        xlabel="RF Voltage [MV]",
        ylabel="Bunch Length [mm]",
        legend=false,
        marker=:circle,
        linewidth=2
    )

    # Bunch length gradient plot
    p4 = plot(
        voltage_MV,
        grad_length_scaled,
        title="d(Bunch Length)/d(Voltage)",
        xlabel="RF Voltage [MV]",
        ylabel="Gradient [mm/MV]",
        legend=false,
        marker=:circle,
        ribbon=grad_length_uncertainty_scaled,
        linewidth=2
    )
    # Add zero line
    hline!(p4, [0.0], linestyle=:dash, color=:black)

    # Combine plots
    p = plot(p1, p2, p3, p4, layout=(2, 2), size=(1000, 800))
    savefig(p, "voltage_scan_results.png")
    display(p)

    println("Plots saved to voltage_scan_results.png")
end

"""
Plot the results of the 2D parameter scan.
"""
function plot_2d_scan_results(param_grid, emittance_grid)
    # Extract parameter values
    voltage_range = param_grid[1]
    α_c_range = param_grid[2]

    # Convert to more readable units
    voltage_MV = voltage_range ./ 1e6
    α_c_values = α_c_range .* 1e4

    # Create heatmap
    p = heatmap(
        voltage_MV,
        α_c_values,
        emittance_grid' .* 1e6,  # Convert to MeV·mm
        title="Emittance vs. RF Voltage and Momentum Compaction",
        xlabel="RF Voltage [MV]",
        ylabel="Momentum Compaction [10⁻⁴]",
        colorbar_title="Emittance [MeV·mm]",
        c=:viridis
    )

    savefig(p, "2d_scan_results.png")
    display(p)

    println("Plot saved to 2d_scan_results.png")
end

# Run the voltage scan
voltage_values, energy_spread_values, bunch_length_values, grad_energy, grad_length = scan_rf_voltage()

# Run the 2D parameter scan
param_grid, emittance_grid = scan_momentum_compaction_and_voltage()