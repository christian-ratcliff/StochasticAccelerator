"""
Optimization model for accelerator parameter optimization.

This file implements the integration between the accelerator simulation
and StochasticAD for gradient-based optimization.
"""

using StochasticAD
using Optimisers
using StructArrays

"""
    ParameterMapping{N}

Mapping between physical parameters and optimization variables.

# Fields
- `indices::Dict{Symbol, Int}`: Dictionary mapping parameter names to indices
- `keys::Vector{Symbol}`: Vector of parameter names
"""
struct ParameterMapping{N}
    indices::Dict{Symbol, Int}
    keys::Vector{Symbol}
    
    function ParameterMapping(indices::Dict{Symbol, Int})
        # Get keys in sorted order by index
        keys_var = sort(collect(keys(indices)), by=k -> indices[k])
        return new{length(keys_var)}(indices, keys_var)
    end
end

"""
    AcceleratorModel{T}

Model for accelerator parameter optimization with StochasticAD integration.

# Fields
- `params::Vector{T}`: Parameter vector
- `base_params::SimulationParameters`: Base simulation parameters
- `mapping::ParameterMapping`: Parameter mapping
- `n_particles::Int`: Number of particles
- `processes::Vector{PhysicsProcess}`: Physics processes
"""
struct AcceleratorModel{T}
    params::Vector{T}
    base_params::SimulationParameters
    mapping::ParameterMapping
    n_particles::Int
    processes::Vector{PhysicsProcess}
    
    function AcceleratorModel(
        params::Vector{T},
        base_params::SimulationParameters,
        mapping::ParameterMapping,
        n_particles::Int,
        processes::Vector{PhysicsProcess}
    ) where T <: Real
        return new{T}(params, base_params, mapping, n_particles, processes)
    end
end

"""
    create_parameter_mapping(params::Dict{Symbol, Int}) -> ParameterMapping

Create a parameter mapping from a dictionary.

# Arguments
- `params`: Dictionary mapping parameter names to indices

# Returns
- Parameter mapping
"""
function create_parameter_mapping(params::Dict{Symbol, Int})
    return ParameterMapping(params)
end

"""
    apply_parameters!(
        model::AcceleratorModel{T},
        params::Vector{T}
    ) where T <: Real -> SimulationParameters

Apply parameter values to create new simulation parameters.

# Arguments
- `model`: Accelerator model
- `params`: Parameter vector

# Returns
- Simulation parameters with updated values
"""
function apply_parameters!(
    model::AcceleratorModel{T},
    params::Vector{T}
) where T <: Real
    
    # Start with base parameters
    base = model.base_params
    
    # Create parameters with appropriate types
    E0 = haskey(model.mapping.indices, :E0) ? 
        params[model.mapping.indices[:E0]] : base.E0
        
    voltage = haskey(model.mapping.indices, :voltage) ? 
        params[model.mapping.indices[:voltage]] : base.voltage
        
    radius = haskey(model.mapping.indices, :radius) ? 
        params[model.mapping.indices[:radius]] : base.radius
        
    pipe_radius = haskey(model.mapping.indices, :pipe_radius) ? 
        params[model.mapping.indices[:pipe_radius]] : base.pipe_radius
        
    α_c = haskey(model.mapping.indices, :α_c) ? 
        params[model.mapping.indices[:α_c]] : base.α_c
        
    ϕs = haskey(model.mapping.indices, :ϕs) ? 
        params[model.mapping.indices[:ϕs]] : base.ϕs
    
    freq_rf = haskey(model.mapping.indices, :freq_rf) ? 
        params[model.mapping.indices[:freq_rf]] : base.freq_rf
    
    # Create simulation parameters with correct types
    return SimulationParameters(
        E0, base.mass, voltage, base.harmonic, radius, pipe_radius, 
        α_c, ϕs, freq_rf, base.n_turns, base.use_wakefield, 
        base.update_η, base.update_E0, base.SR_damping, base.use_excitation
    )
end

"""
    create_parameter_vector(
        base_params::SimulationParameters,
        mapping::ParameterMapping
    ) -> Vector{Float64}

Create a parameter vector from simulation parameters.

# Arguments
- `base_params`: Base simulation parameters
- `mapping`: Parameter mapping

# Returns
- Parameter vector
"""
function create_parameter_vector(
    base_params::SimulationParameters,
    mapping::ParameterMapping
)
    # Create parameter vector
    params = zeros(length(mapping.keys))
    
    # Fill parameter vector
    for (i, key) in enumerate(mapping.keys)
        if key == :E0
            params[i] = base_params.E0
        elseif key == :voltage
            params[i] = base_params.voltage
        elseif key == :radius
            params[i] = base_params.radius
        elseif key == :pipe_radius
            params[i] = base_params.pipe_radius
        elseif key == :α_c
            params[i] = base_params.α_c
        elseif key == :ϕs
            params[i] = base_params.ϕs
        elseif key == :freq_rf
            params[i] = base_params.freq_rf
        end
    end
    
    return params
end

"""
    create_accelerator_model(
        base_params::SimulationParameters,
        param_mapping::Dict{Symbol, Int};
        n_particles::Int=1000
    ) -> AcceleratorModel{Float64}

Create an accelerator model for optimization.

# Arguments
- `base_params`: Base simulation parameters
- `param_mapping`: Dictionary mapping parameter names to indices
- `n_particles`: Number of particles

# Returns
- Accelerator model
"""
function create_accelerator_model(
    base_params::SimulationParameters,
    param_mapping::Dict{Symbol, Int};
    n_particles::Int=1000
)
    # Create parameter mapping
    mapping = create_parameter_mapping(param_mapping)
    
    # Create parameter vector
    params = create_parameter_vector(base_params, mapping)
    
    # Setup simulation
    _, processes, _ = setup_simulation(
        base_params, n_particles
    )
    
    # Create model
    return AcceleratorModel(params, base_params, mapping, n_particles, processes)
end

"""
    run_model(
        model::AcceleratorModel{T},
        params::Vector{T}
    ) where T <: Real -> Tuple{StructArray{Particle{Float64}}, Float64, Float64, Any}

Run the accelerator model with the given parameters.

# Arguments
- `model`: Accelerator model
- `params`: Parameter vector

# Returns
- `particles`: Final particle distribution
- `σ_E`: Final energy spread [eV]
- `σ_z`: Final bunch length [m]
- `E0`: Final reference energy [eV]
"""
function run_model(
    model::AcceleratorModel{T},
    params::Vector{T}
) where T <: Real
    # Create simulation parameters
    sim_params = apply_parameters!(model, params)
    
    # Initial distribution parameters
    μ_z = 0.0
    μ_E = 0.0
    σ_z0 = 0.005  # 5 mm bunch length
    σ_E0 = 1e6    # 1 MeV energy spread
    
    # Generate particles
    particles, _, _, _ = generate_particles(
        μ_z, μ_E, σ_z0, σ_E0, model.n_particles,
        sim_params.E0, sim_params.mass, sim_params.ϕs, sim_params.freq_rf
    )
    
    # Create buffers
    nbins = Int(model.n_particles/10)
    buffers = create_simulation_buffers(model.n_particles, nbins, Float64)
    
    # Run simulation for multiple turns
    σ_E = σ_E0
    σ_z = σ_z0
    E0 = sim_params.E0
    
    for turn in 1:sim_params.n_turns
        # Track particles for one turn
        σ_E, σ_z, E0 = track_particles!(
            particles, 
            model.processes, 
            sim_params, 
            buffers;
            update_η=sim_params.update_η,
            update_E0=sim_params.update_E0
        )
    end
    
    return particles, σ_E, σ_z, E0
end

"""
    create_stochastic_model(
        model::AcceleratorModel{Float64},
        fom_function::Function
    ) -> StochasticModel

Create a StochasticModel for the accelerator model.

# Arguments
- `model`: Accelerator model
- `fom_function`: Figure of merit function taking (particles, σ_E, σ_z, E0)

# Returns
- StochasticModel for optimization
"""
function create_stochastic_model(
    model::AcceleratorModel{Float64},
    fom_function::Function
)
    # Create objective function
    function objective(p)
        # Run model
        particles, σ_E, σ_z, E0 = run_model(model, p)
        
        # Calculate figure of merit
        return fom_function(particles, σ_E, σ_z, E0)
    end
    
    # Create StochasticModel
    return StochasticModel(objective, model.params)
end