include("../src/StochasticAccelerator.jl")

begin
    using .StochasticAccelerator
    using Plots
    using Statistics
    using Random
    using StochasticAD
end

# Set random seed for reproducibility
Random.seed!(1234)

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
    # plot_2d_scan_results_fd(param_grid, emittance_grid)
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
    emittance_values = emittance_grid .* 1e6  # Convert to MeV·mm
    
    # Create a 3-panel layout
    l = @layout [a{0.5w} [b; c]]
    
    # 1. Main contour plot of emittance
    p1 = contourf(
        voltage_MV,
        α_c_values,
        emittance_values',
        title="Emittance vs. RF Voltage and Momentum Compaction",
        xlabel="RF Voltage [MV]",
        ylabel="Momentum Compaction [10⁻⁴]",
        c=:viridis,
        colorbar_title="Emittance [MeV·mm]",
        levels=10
    )
    
    # Add contour lines and mark the minimum
    contour!(p1, voltage_MV, α_c_values, emittance_values', linecolor=:black, linewidth=0.5)
    min_val, min_idx = findmin(emittance_values)
    min_i, min_j = Tuple(CartesianIndices(emittance_values)[min_idx])
    scatter!(p1, [voltage_MV[min_i]], [α_c_values[min_j]], 
             color=:red, markersize=8, markershape=:star, label="Minimum")
    
    # 2. Calculate derivative with respect to voltage
    dE_dV = zeros(length(voltage_range), length(α_c_range))
    
    # Use forward/backward/central differences as appropriate
    for i in 1:length(voltage_range)
        for j in 1:length(α_c_range)
            if i == 1
                dE_dV[i,j] = (emittance_values[i+1,j] - emittance_values[i,j]) / 
                             (voltage_range[i+1] - voltage_range[i])
            elseif i == length(voltage_range)
                dE_dV[i,j] = (emittance_values[i,j] - emittance_values[i-1,j]) / 
                             (voltage_range[i] - voltage_range[i-1])
            else
                dE_dV[i,j] = (emittance_values[i+1,j] - emittance_values[i-1,j]) / 
                             (voltage_range[i+1] - voltage_range[i-1])
            end
        end
    end
    
    # Plot voltage derivative
    dE_dV_scaled = dE_dV .* 1e6 ./ 1e6  # MeV·mm / MV
    max_dV = maximum(abs.(dE_dV_scaled))
    p2 = contourf(
        voltage_MV,
        α_c_values,
        dE_dV_scaled',
        title="∂(Emittance)/∂(Voltage)",
        xlabel="RF Voltage [MV]",
        ylabel="Momentum Compaction [10⁻⁴]",
        c=:RdBu,  # Red-Blue diverging colormap for derivatives
        clims=(-max_dV, max_dV),  # Symmetric limits
        colorbar_title="MeV·mm/MV"
    )
    # Add zero contour line (where derivative changes sign)
    contour!(p2, voltage_MV, α_c_values, dE_dV_scaled', levels=[0], 
             linecolor=:black, linewidth=2, linestyle=:dash)
    
    # 3. Calculate derivative with respect to α_c
    dE_dα = zeros(length(voltage_range), length(α_c_range))
    for i in 1:length(voltage_range)
        for j in 1:length(α_c_range)
            if j == 1
                dE_dα[i,j] = (emittance_values[i,j+1] - emittance_values[i,j]) / 
                             (α_c_range[j+1] - α_c_range[j])
            elseif j == length(α_c_range)
                dE_dα[i,j] = (emittance_values[i,j] - emittance_values[i,j-1]) / 
                             (α_c_range[j] - α_c_range[j-1])
            else
                dE_dα[i,j] = (emittance_values[i,j+1] - emittance_values[i,j-1]) / 
                             (α_c_range[j+1] - α_c_range[j-1])
            end
        end
    end
    
    # Plot α_c derivative
    dE_dα_scaled = dE_dα .* 1e6 ./ 1e4  # MeV·mm / 10⁻⁴
    max_dα = maximum(abs.(dE_dα_scaled))
    p3 = contourf(
        voltage_MV,
        α_c_values,
        dE_dα_scaled',
        title="∂(Emittance)/∂(α_c)",
        xlabel="RF Voltage [MV]",
        ylabel="Momentum Compaction [10⁻⁴]",
        c=:RdBu,  # Red-Blue diverging colormap
        clims=(-max_dα, max_dα),  # Symmetric limits
        colorbar_title="MeV·mm/10⁻⁴"
    )
    # Add zero contour line
    contour!(p3, voltage_MV, α_c_values, dE_dα_scaled', levels=[0], 
             linecolor=:black, linewidth=2, linestyle=:dash)
    
    # Combine plots
    p = plot(p1, p2, p3, layout=l, size=(1200, 800))
    
    # Add annotation about minimum
    min_info = "Minimum emittance: $(round(min_val, digits=2)) MeV·mm\nat V=$(round(voltage_MV[min_i], digits=2)) MV, α_c=$(round(α_c_values[min_j], digits=2))×10⁻⁴"
    annotate!(p1, voltage_MV[1] + 0.2*(voltage_MV[end] - voltage_MV[1]), 
              α_c_values[1] + 0.1*(α_c_values[end] - α_c_values[1]), 
              text(min_info, 8, :left))
    
    savefig(p, "2d_scan_results.png")
    display(p)
    
    println("Enhanced plot saved to 2d_scan_results.png")
    println("Minimum emittance: $(round(min_val, digits=3)) MeV·mm at $(round(voltage_MV[min_i], digits=2)) MV, $(round(α_c_values[min_j], digits=2))×10⁻⁴")
    
    return p
end

# function plot_2d_scan_results(param_grid, emittance_grid)
#     # Extract parameter values
#     voltage_range = param_grid[1]
#     α_c_range = param_grid[2]
    
#     # Get gradients calculated by StochasticAD
#     dE_dV_grid = gradient_grids[1] 
#     dE_dα_grid = gradient_grids[2]
    
#     # Convert to more readable units
#     voltage_MV = voltage_range ./ 1e6
#     α_c_values = α_c_range .* 1e4
#     emittance_values = emittance_grid .* 1e6  # Convert to MeV·mm
    
#     # Scale gradients to appropriate units
#     dE_dV_scaled = dE_dV_grid .* 1e6 ./ 1e6  # MeV·mm/MV
#     dE_dα_scaled = dE_dα_grid .* 1e6 ./ 1e4  # MeV·mm/10⁻⁴
    
#     # Create a 3-panel layout
#     l = @layout [a{0.5w} [b; c]]
    
#     # 1. Main contour plot of emittance
#     p1 = contourf(
#         voltage_MV,
#         α_c_values,
#         emittance_values',
#         title="Emittance vs. RF Voltage and Momentum Compaction",
#         xlabel="RF Voltage [MV]",
#         ylabel="Momentum Compaction [10⁻⁴]",
#         c=:viridis,
#         colorbar_title="Emittance [MeV·mm]",
#         levels=10
#     )
    
#     # Add contour lines and mark the minimum
#     min_val, min_idx = findmin(emittance_values)
#     min_i, min_j = Tuple(CartesianIndices(emittance_values)[min_idx])
#     scatter!(p1, [voltage_MV[min_i]], [α_c_values[min_j]], 
#              color=:red, markersize=8, markershape=:star, label="Minimum")
    
#     # 2. Plot voltage derivative from StochasticAD
#     max_dV = maximum(abs.(dE_dV_scaled))
#     p2 = contourf(
#         voltage_MV,
#         α_c_values,
#         dE_dV_scaled',
#         title="∂(Emittance)/∂(Voltage) via StochasticAD",
#         xlabel="RF Voltage [MV]",
#         ylabel="Momentum Compaction [10⁻⁴]",
#         c=:RdBu,
#         clims=(-max_dV, max_dV),
#         colorbar_title="MeV·mm/MV"
#     )
    
#     # 3. Plot α_c derivative from StochasticAD
#     max_dα = maximum(abs.(dE_dα_scaled))
#     p3 = contourf(
#         voltage_MV,
#         α_c_values,
#         dE_dα_scaled',
#         title="∂(Emittance)/∂(α_c) via StochasticAD",
#         xlabel="RF Voltage [MV]",
#         ylabel="Momentum Compaction [10⁻⁴]",
#         c=:RdBu,
#         clims=(-max_dα, max_dα),
#         colorbar_title="MeV·mm/10⁻⁴"
#     )
    
#     # Combine plots
#     p = plot(p1, p2, p3, layout=l, size=(1200, 800))
    
#     # Add annotation about minimum
#     min_info = "Minimum emittance: $(round(min_val, digits=2)) MeV·mm\nat V=$(round(voltage_MV[min_i], digits=2)) MV, α_c=$(round(α_c_values[min_j], digits=2))×10⁻⁴"
#     annotate!(p1, voltage_MV[1] + 0.2*(voltage_MV[end] - voltage_MV[1]), 
#               α_c_values[1] + 0.1*(α_c_values[end] - α_c_values[1]), 
#               text(min_info, 8, :left))
    
#     savefig(p, "2d_scan_results.png")
#     display(p)
    
#     println("Enhanced plot saved to 2d_scan_results.png")
#     println("Minimum emittance: $(round(min_val, digits=3)) MeV·mm at $(round(voltage_MV[min_i], digits=2)) MV, $(round(α_c_values[min_j], digits=2))×10⁻⁴")
    
#     return p
# end


# Run the voltage scan
voltage_values, energy_spread_values, bunch_length_values, grad_energy, grad_length = scan_rf_voltage();

# Run the 2D parameter scan
param_grid, emittance_grid = scan_momentum_compaction_and_voltage();

using Distributions
function test_stochastic_ad()
    # Define a simple function
    f(x) = x^2 + rand(Bernoulli(x)) * 3
    
    # Get a stochastic triple
    st = StochasticAD.stochastic_triple(f, 0.5)
    println("Stochastic triple: ", st)
    
    # Get derivative estimate
    deriv = StochasticAD.derivative_contribution(st)
    println("Derivative estimate: ", deriv)
    
    # Also try direct derivative_estimate
    direct_deriv = StochasticAD.derivative_estimate(f, 0.5)
    println("Direct derivative estimate: ", direct_deriv)
end
test_stochastic_ad()

function simplified_voltage_scan()
    println("Running simplified voltage scan with full debug...")
    
    # Define our own custom function that directly relates voltage to energy spread
    function voltage_to_energy_spread(v)
        # Simplified physics model: energy spread increases with sqrt of voltage
        # This is a simplified model based on synchrotron oscillation theory
        base_spread = 1e6  # 1 MeV
        v_normalized = v / 5e6  # Normalize to 5 MV
        # Add randomness to make it stochastic (needed for StochasticAD)
        noise = rand(Bernoulli(0.5)) * 0.1 * base_spread
        return base_spread * sqrt(v_normalized) + noise
    end
    
    # Define voltage range
    voltage_range = range(3e6, 7e6, length=5)
    
    # Scan voltage
    println("Scanning voltage directly...")
    for voltage in voltage_range
        # Get energy spread
        energy_spread = voltage_to_energy_spread(voltage)
        println("Voltage: $voltage V, Energy spread: $energy_spread eV")
        
        # Get derivative using StochasticAD
        deriv = StochasticAD.derivative_estimate(voltage_to_energy_spread, voltage)
        println("  Derivative: $deriv eV/V")
    end
end

# Add this to parameter_scan.jl and run it
simplified_voltage_scan()

function rebuild_voltage_scan()
    println("Running rebuilt voltage scan...")
    
    # Define our base parameters
    E0 = 4e9            # 4 GeV energy
    mass = MASS_ELECTRON
    voltage_base = 5e6  # 5 MV RF voltage
    harmonic = 360      # Harmonic number
    radius = 250.0      # Ring radius (m)
    pipe_radius = 0.00025 # Beam pipe radius (m)
    α_c = 3.68e-4       # Momentum compaction
    ϕs = 5π / 6         # Synchronous phase
    n_turns = 10        # Reduced number of turns for speed
    
    # Define a function that directly relates voltage to energy spread and bunch length
    # This uses the simplified model but calls the actual simulation
    function voltage_to_metrics(v)
        println("Running with voltage: $v")
        
        # Create parameters with this voltage
        params = SimulationParameters(
            E0, mass, v, harmonic, radius, pipe_radius,
            α_c, ϕs, (ϕs + 10 * π / 180) * sqrt(1 - mass^2/E0^2) * SPEED_LIGHT / (2π), 
            n_turns, true, true, true, true, true
        )
        
        # Setup a minimal simulation
        n_particles = 1000  # Reduced for speed
        particles, processes, buffers = setup_simulation(
            params, n_particles; μ_z=0.0, μ_E=0.0, σ_z0=0.005, σ_E0=1e6
        )
        
        # Run the simulation
        for turn in 1:n_turns
            σ_E, σ_z, E0_final = track_particles!(
                particles, processes, params, buffers;
                update_η=params.update_η, update_E0=params.update_E0
            )
        end
        
        # Calculate final statistics
        final_σ_E = std(particles.coordinates.ΔE)
        final_σ_z = std(particles.coordinates.z)
        
        # Add Bernoulli randomness to make it stochastic (required for StochasticAD)
        final_σ_E += rand(Bernoulli(0.5)) * 0.01 * final_σ_E
        
        # Return energy spread and bunch length
        return (final_σ_E, final_σ_z)
    end
    
    # Define a function for energy spread only
    function voltage_to_energy_spread(v)
        return voltage_to_metrics(v)[1]
    end
    
    # Define a function for bunch length only
    function voltage_to_bunch_length(v)
        return voltage_to_metrics(v)[2]
    end
    
    # Define voltage range
    voltage_range = range(3e6, 7e6, length=5)
    
    # Initialize results
    energy_spread_values = Float64[]
    bunch_length_values = Float64[]
    energy_spread_derivs = Float64[]
    bunch_length_derivs = Float64[]
    
    # Scan voltage
    println("Scanning voltage...")
    for voltage in voltage_range
        # Get metrics directly
        σ_E, σ_z = voltage_to_metrics(voltage)
        push!(energy_spread_values, σ_E)
        push!(bunch_length_values, σ_z)
        
        println("Voltage: $voltage V")
        println("  Energy spread: $σ_E eV")
        println("  Bunch length: $(σ_z*1000) mm")
        
        # Get derivatives using StochasticAD
        deriv_E = StochasticAD.derivative_estimate(voltage_to_energy_spread, voltage)
        push!(energy_spread_derivs, deriv_E)
        println("  Energy spread derivative: $deriv_E eV/V")
        
        deriv_z = StochasticAD.derivative_estimate(voltage_to_bunch_length, voltage)
        push!(bunch_length_derivs, deriv_z)
        println("  Bunch length derivative: $deriv_z m/V")
    end
    
    return voltage_range, energy_spread_values, bunch_length_values, energy_spread_derivs, bunch_length_derivs
end

# Add this to parameter_scan.jl and run it
results = rebuild_voltage_scan()


function simplified_rf_scan()
    println("Running simplified RF scan with StochasticAD...")
    
    # Define a simple model with explicit randomness
    function rf_model(voltage)
        # Basic physical parameters
        E0 = 4e9  # 4 GeV
        alpha_c = 3.68e-4  # Momentum compaction
        harmonic = 360
        
        # Calculate equilibrium energy spread and bunch length
        # These follow from standard synchrotron radiation theory
        σ_E_base = 1e6  # Base energy spread in eV
        
        # Add randomness through Bernoulli (needed for StochasticAD)
        rand_factor = 1.0 + 0.05 * rand(Bernoulli(0.5))
        
        
        # Energy spread scales approximately as voltage^(1/4) in radiation-dominated regime
        σ_E = σ_E_base * (voltage/5e6)^0.25 * rand_factor
        
        return σ_E
    end
    
    # Define voltage range
    voltage_range = range(3e6, 7e6, length=5)
    
    # Initialize results
    results = []
    
    # Scan voltage
    for voltage in voltage_range
        # Get energy spread
        energy_spread = rf_model(voltage)
        
        # Calculate derivative using StochasticAD
        deriv = StochasticAD.derivative_estimate(rf_model, voltage)
        
        # Print results
        println("Voltage: $voltage V")
        println("  Energy spread: $energy_spread eV")
        println("  Derivative: $deriv eV/V")
        
        push!(results, (voltage, energy_spread, deriv))
    end
    
    return results
end

# Now run the scan
results = simplified_rf_scan()

# Plot the results
voltage_values = [r[1] for r in results]
energy_spread_values = [r[2] for r in results]
derivative_values = [r[3] for r in results]

# Convert to more readable units
voltage_MV = voltage_values ./ 1e6
energy_spread_MeV = energy_spread_values ./ 1e6
derivative_scaled = derivative_values ./ 1e6 .* 1e6  # MeV/MV

# Create plots
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

p2 = plot(
    voltage_MV,
    derivative_scaled,
    title="d(Energy Spread)/d(Voltage)",
    xlabel="RF Voltage [MV]",
    ylabel="Gradient [MeV/MV]",
    legend=false,
    marker=:circle,
    linewidth=2
)

# Combine plots
p = plot(p1, p2, layout=(2, 1), size=(800, 600))
display(p)
savefig(p, "rf_scan_results.png")