include("../src/StochasticAccelerator.jl")
begin
    using .StochasticAccelerator
    using BenchmarkTools
    using StructArrays
    using Profile
    using ProgressMeter
    using ProfileSVG
end

begin
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
end;

begin
    println("Initial beam parameters:")
    println("  Energy: $E0 eV")
    println("  Energy spread: $σ_E0 eV")
    println("  Bunch length: $(σ_z0*1000) mm")
end

# Setup simulation
begin
    n_particles = Int64(1e5)
    particles, processes, buffers = setup_simulation(
        params, n_particles;
        μ_z=μ_z, μ_E=μ_E, σ_z0=σ_z0, σ_E0=σ_E0
    )
end;



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

@benchmark longitidunal_evolve($particles, $params,$buffers, $processes)

@profile longitidunal_evolve(particles, params, buffers, processes)
Profile.print()

@ProfileSVG.profview longitidunal_evolve(particles, params, buffers, processes)

particles.coordinates.z
# Precompute constants
z_vals = copy(particles.coordinates.z) ; # Cache the value
ΔE_vals = copy(particles.coordinates.ΔE) ; # Cache the value
sinϕ = sin.(-particles.coordinates.z .* 2.4 .+ ϕs) .- sin(ϕs) ;
sin_ϕs = sin(ϕs)
-particles.coordinates.z .* rf_factor .+ ϕs
@btime sinϕ = sin.(-particles.coordinates.z .* rf_factor .+ ϕs) .- sin_ϕs ;
@btime @fastmath sinϕ = sin.(-particles.coordinates.z .* rf_factor .+ ϕs) .- sin_ϕs  ;

# Iterate and use cached values
@btime for i in 1:length(particles)
    
    
    # ϕ_val = -(z_vals[i] * rf_factor - ϕs)
    ΔE_vals[i] += voltage * sinϕ[i]
    
    # particles.coordinates.ΔE[i] = ΔE_i  # Store the updated value
end

@btime ΔE_vals .= ΔE_vals .+ voltage .* sinϕ

rf_factor = freq_rf * 2π / (β* SPEED_LIGHT)
# 531.276 μs (9 allocations: 1.53 MiB)


@inline function sine_lookup(x::Float64)


    # Find the two closest points in the table
    index = floor(Int, x / 0.0000000001) + 1  # Get index of closest table point (1-based indexing)
    
    # # If x is exactly on a table point, return the corresponding sine value
    # if x == table_x[index]
    #     return table_sin[index]
    # end
    
    # # Linear interpolation: Find the two closest points surrounding x
    # x0, x1 = table_x[index], table_x[index + 1]
    # sin0, sin1 = table_sin[index], table_sin[index + 1]
    
    # Perform linear interpolation
    # return sin0 + (sin1 - sin0) * (x - x0) / (x1 - x0)
    return table_sin[index]
end
# To compute the sine of a value:
@btime sine_lookup.(-particles.coordinates.z .* rf_factor .+ ϕs) .- sin(ϕs) ;

@btime @. sin.(-particles.coordinates.z * rf_factor + ϕs) - 0.5 ;

using Base.Threads
function threaded_update!(particles::StructArray{Particle{Float64}}, rf_factor, ϕs, sin_ϕs, voltage, n_particles::Int64)
    # Extract variables from particles
    z = particles.coordinates.z
    ΔE = particles.coordinates.ΔE 
    
    # Use threads to compute sinϕ and update ΔE in parallel
    @threads for i in 1:n_particles
        # Compute sinϕ for the i-th particle
        sinϕ = sin(-z[i] * rf_factor + ϕs) - sin_ϕs
        
        # Update the energy change for the i-th particle
        ΔE[i] += voltage * sinϕ
    end
end

@btime threaded_update!($particles, $rf_factor, $ϕs, $sin_ϕs, $voltage, $n_particles)
