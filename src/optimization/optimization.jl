"""
Optimization algorithms for accelerator parameter optimization.

This file implements optimization algorithms and routines for
finding optimal accelerator parameters.
"""

using StochasticAD
using Optimisers
using Statistics
using ProgressMeter

"""
    optimize_parameters(
        model::AcceleratorModel{Float64},
        fom_function::Function;
        iterations::Int=100,
        optimizer=Descent(0.01),
        show_progress::Bool=true,
        callback::Function=nothing,
        constraints::Dict{Int, Tuple{Float64, Float64}}=Dict{Int, Tuple{Float64, Float64}}()
    ) -> Tuple{Vector{Float64}, Vector{Float64}, Vector{Vector{Float64}}}

Optimize accelerator parameters using gradient descent.

# Arguments
- `model`: Accelerator model
- `fom_function`: Figure of merit function
- `iterations`: Number of iterations
- `optimizer`: Optimizer from Optimisers.jl
- `show_progress`: Whether to show progress bar
- `callback`: Callback function called after each iteration
- `constraints`: Dictionary mapping parameter indices to (min, max) constraints

# Returns
- `optimal_params`: Optimal parameters
- `fom_history`: Figure of merit history
- `param_history`: Parameter history
"""
function optimize_parameters(
    model::AcceleratorModel,
    fom_function::Function;
    iterations::Int=100,
    optimizer=Descent(0.01),
    show_progress::Bool=true,
    callback::Function=nothing,
    constraints::Dict{Int, Tuple{Float64, Float64}}=Dict{Int, Tuple{Float64, Float64}}()
    )
    # Create StochasticModel
    stoch_model = create_stochastic_model(model, fom_function)
    
    # Setup optimizer
    opt_state = Optimisers.setup(optimizer, stoch_model)
    
    # Track history
    fom_history = Float64[]
    param_history = Vector{Parameter}[]
    
    # Setup progress meter
    prog = show_progress ? Progress(iterations, desc="Optimizing: ") : nothing
    
    # Run optimization
    for i in 1:iterations
        # Calculate current FoM
        particles, σ_E, σ_z, E0 = run_model(model, stoch_model.p)
        current_fom = fom_function(particles, σ_E, σ_z, E0)
        
        # Record history
        push!(fom_history, current_fom)
        push!(param_history, copy(stoch_model.p))
        
        # Print progress
        if show_progress
            param_desc = ""
            for (j, key) in enumerate(model.mapping.keys)
                param_desc *= "$key=$(round(safe_value(stoch_model.p[j]), digits=6)) "
            end
            ProgressMeter.update!(prog, i, desc="Optimizing (FoM=$current_fom): ", valuecolor=:blue, showvalues=[(:Parameters, param_desc)])
        end
        
        # Call callback if provided
        if callback !== nothing
            callback(i, stoch_model.p, current_fom)
        end
        
        # Calculate gradient
        grad = stochastic_gradient(stoch_model)
        
        # Update parameters
        Optimisers.update!(opt_state, stoch_model, grad)
        
        # Apply constraints
        for (idx, (min_val, max_val)) in constraints
            val = safe_value(stoch_model.p[idx])
            stoch_model.p[idx] = clamp(val, min_val, max_val)
        end
    end
    
    return stoch_model.p, fom_history, param_history
end

"""
    scan_parameter(
        model::AcceleratorModel{Float64},
        fom_function::Function,
        param_index::Int,
        param_range::AbstractVector{Float64};
        calculate_gradients::Bool=true,
        n_gradient_samples::Int=5
    ) -> Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}}

Scan a parameter over a range of values.

# Arguments
- `model`: Accelerator model
- `fom_function`: Figure of merit function
- `param_index`: Index of parameter to scan
- `param_range`: Range of parameter values
- `calculate_gradients`: Whether to calculate gradients
- `n_gradient_samples`: Number of samples for gradient uncertainty

# Returns
- `param_values`: Parameter values
- `fom_values`: Figure of merit values
- `gradient_values`: Gradient values (if calculated)
- `gradient_uncertainties`: Gradient uncertainties (if calculated)
"""
function scan_parameter(
    model::AcceleratorModel,
    fom_function::Function,
    param_index::Int,
    param_range::AbstractVector{Float64};
    calculate_gradients::Bool=true,
    n_gradient_samples::Int=5
)
    # Initialize result arrays
    param_values = copy(param_range)
    fom_values = zeros(length(param_range))
    gradient_values = calculate_gradients ? zeros(length(param_range)) : Float64[]
    gradient_uncertainties = calculate_gradients ? zeros(length(param_range)) : Float64[]
    
    # Create copy of model
    model_copy = deepcopy(model)
    
    # Scan parameter
    for (i, param_value) in enumerate(param_range)
        # Update parameter
        model_copy.params[param_index] = param_value
        
        # Run model
        particles, σ_E, σ_z, E0 = run_model(model_copy, model_copy.params)
        
        # Calculate figure of merit
        fom_values[i] = fom_function(particles, σ_E, σ_z, E0)
        
        # Calculate gradient if requested
        if calculate_gradients
            # Create StochasticModel
            stoch_model = create_stochastic_model(model_copy, fom_function)
            
            # Calculate gradient with uncertainty
            println(stochastic_gradient(stoch_model))
            gradient_samples = [stochastic_gradient(stoch_model)[param_index] for _ in 1:n_gradient_samples]
            gradient_values[i] = mean(gradient_samples)
            gradient_uncertainties[i] = std(gradient_samples) / sqrt(n_gradient_samples)
        end
    end
    
    return param_values, fom_values, gradient_values, gradient_uncertainties
end


"""
    multi_parameter_scan(
        model::AcceleratorModel{Float64},
        fom_function::Function,
        param_indices::Vector{Int},
        param_ranges::Vector{AbstractVector{Float64}}
    ) -> Tuple{Vector{Vector{Float64}}, Array{Float64}}

Scan multiple parameters over ranges of values.

# Arguments
- `model`: Accelerator model
- `fom_function`: Figure of merit function
- `param_indices`: Indices of parameters to scan
- `param_ranges`: Ranges of parameter values

# Returns
- `param_grid`: Grid of parameter values
- `fom_grid`: Grid of figure of merit values
"""
function multi_parameter_scan(
    model::AcceleratorModel,
    fom_function::Function,
    param_indices::Vector{Int},
    param_ranges::Vector{AbstractVector{Float64}}
)
    # Check that we have two parameters at most
    @assert length(param_indices) <= 2 "Cannot scan more than 2 parameters at once"
    
    # Create copy of model
    model_copy = deepcopy(model)
    
    if length(param_indices) == 1
        # Single parameter scan
        param_values, fom_values, _, _ = scan_parameter(
            model_copy,
            fom_function,
            param_indices[1],
            param_ranges[1];
            calculate_gradients=false
        )
        
        return [param_values], fom_values
    else
        # Two parameter scan
        param1_range = param_ranges[1]
        param2_range = param_ranges[2]
        
        # Initialize result grid
        fom_grid = zeros(length(param1_range), length(param2_range))
        
        # Scan parameters
        for (i, param1_value) in enumerate(param1_range)
            for (j, param2_value) in enumerate(param2_range)
                # Update parameters
                model_copy.params[param_indices[1]] = param1_value
                model_copy.params[param_indices[2]] = param2_value
                
                # Run model
                particles, σ_E, σ_z, E0 = run_model(model_copy, model_copy.params)
                
                # Calculate figure of merit
                fom_grid[i, j] = fom_function(particles, σ_E, σ_z, E0)
            end
        end
        
        return [param1_range, param2_range], fom_grid
    end
end


