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
struct AcceleratorModel
    params::Vector{Parameter}
    base_params::SimulationParameters
    mapping::ParameterMapping
    n_particles::Int
    processes::Vector{PhysicsProcess}
    
    function AcceleratorModel(
        params::Vector{Float64},
        base_params::SimulationParameters,
        mapping::ParameterMapping,
        n_particles::Int,
        processes::Vector{PhysicsProcess}
    )
        param_vector = convert(Vector{Parameter}, params)
        return new(param_vector, base_params, mapping, n_particles, processes)
    end
    
    function AcceleratorModel(
        params::Vector{Parameter},
        base_params::SimulationParameters,
        mapping::ParameterMapping,
        n_particles::Int,
        processes::Vector{PhysicsProcess}
    )
        return new(params, base_params, mapping, n_particles, processes)
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
    model::AcceleratorModel,
    params::Vector{<:Parameter}
) 
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
    params = Vector{Parameter}(undef, length(mapping.keys))
    
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
function run_model(model::AcceleratorModel, params::Vector{<:Parameter})
    
    return StochasticAD.propagate(params...) do p_vals...
        # Create simulation parameters with the parameter values
        param_dict = Dict(zip(model.mapping.keys, p_vals))
        
        # Extract parameters with proper types
        E0 = get(param_dict, :E0, model.base_params.E0)
        voltage = get(param_dict, :voltage, model.base_params.voltage)
        radius = get(param_dict, :radius, model.base_params.radius)
        pipe_radius = get(param_dict, :pipe_radius, model.base_params.pipe_radius)
        α_c = get(param_dict, :α_c, model.base_params.α_c)
        ϕs = get(param_dict, :ϕs, model.base_params.ϕs)
        freq_rf = get(param_dict, :freq_rf, model.base_params.freq_rf)
        
        # Create simulation parameters with extracted values
        sim_params = SimulationParameters(
            E0, model.base_params.mass, voltage, model.base_params.harmonic, 
            radius, pipe_radius, α_c, ϕs, freq_rf, model.base_params.n_turns,
            model.base_params.use_wakefield, model.base_params.update_η, 
            model.base_params.update_E0, model.base_params.SR_damping, 
            model.base_params.use_excitation
        )
        
        # Generate particles with Float64 values
        particles, σ_E0, σ_z0, _ = generate_particles(
            0.0, 0.0, 0.005, 1e6, model.n_particles,
            sim_params.E0, sim_params.mass, sim_params.ϕs, sim_params.freq_rf
        )
        
        # Create buffers
        nbins = Int(model.n_particles/10)
        buffers = create_simulation_buffers(model.n_particles, nbins, Float64)
        
        # Run simulation
        σ_E = σ_E0
        σ_z = σ_z0
        E0_final = sim_params.E0
        
        for turn in 1:sim_params.n_turns
            σ_E, σ_z, E0_final = track_particles!(
                particles, model.processes, sim_params, buffers;
                update_η=sim_params.update_η, update_E0=sim_params.update_E0
            )
        end
        
        # Return the final results
        return particles, σ_E, σ_z, E0_final
    end
end

"""
    create_stochastic_model(
        model::AcceleratorModel{Float64},const Parameter = Union{Float64, StochasticTriple}

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
    model::AcceleratorModel,
    fom_function::Function
)
    # Create objective function
    function objective(p)
        # Convert params to Parameter type if they're not already
        p_params = convert(Vector{Parameter}, p)
        
        # Run model with Parameter vector
        particles, σ_E, σ_z, E0 = run_model(model, p_params)
        
        # Calculate figure of merit
        # println(typeof(particles))
        # println(particles[1])
        return fom_function(particles, σ_E, σ_z, E0)
    end
    
    # Create StochasticModel with concrete Float64 parameters
    # Extract values from any StochasticTriple parameters
    float_params = map(safe_value, model.params)
    
    return StochasticModel(objective, float_params)
end