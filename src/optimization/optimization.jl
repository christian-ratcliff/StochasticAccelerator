"""
Optimization algorithms for accelerator parameter optimization.

This file implements optimization algorithms and routines for
finding optimal accelerator parameters.
"""

using Optimisers
using Statistics
using ProgressMeter
using StochasticAD: StochasticTriple, derivative_contribution, value, PrunedFIsBackend

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
# function scan_parameter(
#     model::AcceleratorModel,
#     fom_function::Function,
#     param_index::Int,
#     param_range::AbstractVector{Float64};
#     calculate_gradients::Bool=true,
#     n_gradient_samples::Int=5
#     )
#     # Initialize result arrays
#     param_values = copy(param_range)
#     fom_values = zeros(length(param_range))
#     gradient_values = calculate_gradients ? zeros(length(param_range)) : Float64[]
#     gradient_uncertainties = calculate_gradients ? zeros(length(param_range)) : Float64[]
    
#     # Create copy of model
#     model_copy = deepcopy(model)
    
#     # Scan parameter
#     for (i, param_value) in enumerate(param_range)
#         # Update parameter
#         model_copy.params[param_index] = param_value
        
#         # Run model
#         particles, σ_E, σ_z, E0 = run_model(model_copy, model_copy.params)
        
#         # Calculate figure of merit
#         fom_values[i] = fom_function(particles, σ_E, σ_z, E0)
        
#         # Calculate gradient if requested
#         if calculate_gradients
#             # Update parameter, DEBUG
#             model_copy.params[param_index] = param_value
            
#             # Run model
#             particles, σ_E, σ_z, E0 = run_model(model_copy, model_copy.params)
            
#             println("Parameter: ", param_value)
#             println("σ_E type: ", typeof(σ_E))
#             println("σ_E value: ", σ_E)
        
#             # Create StochasticModel
#             stoch_model = create_stochastic_model(model_copy, fom_function)
            
#             # Calculate gradient with uncertainty
#             # println(typeof(stochastic_gradient(stoch_model)))
#             gradient_samples = [stochastic_gradient(stoch_model).p[param_index] for _ in 1:n_gradient_samples]
#             gradient_values[i] = mean(gradient_samples)
#             gradient_uncertainties[i] = std(gradient_samples) / sqrt(n_gradient_samples)
#         end
#     end
    
#     return param_values, fom_values, gradient_values, gradient_uncertainties
# end

# function scan_parameter(
#     model::AcceleratorModel,
#     fom_function::Function,
#     param_index::Int,
#     param_range::AbstractVector{Float64};
#     calculate_gradients::Bool=true,
#     n_gradient_samples::Int=5
#     )
#     # Initialize result arrays
#     param_values = copy(param_range)
#     fom_values = zeros(length(param_range))
#     gradient_values = calculate_gradients ? zeros(length(param_range)) : Float64[]
#     gradient_uncertainties = calculate_gradients ? zeros(length(param_range)) : Float64[]
    
#     # Create copy of model
#     model_copy = deepcopy(model)
    
#     # Scan parameter
#     for (i, param_value) in enumerate(param_range)
#         # Update parameter
#         # Instead of setting param_value directly, do this:
#         st_param = StochasticAD.stochastic_triple(param_value)
#         println("Parameter type after StochasticTriple creation: ", typeof(model_copy.params[param_index]))

#         model_copy.params[param_index] = st_param
        
#         # Run model with this StochasticTriple parameter
#         particles, σ_E, σ_z, E0 = run_model(model_copy, model_copy.params)
#         println("σ_E type after run_model: ", typeof(σ_E))
#         println("Can extract derivative: ", typeof(σ_E) <: StochasticTriple ? "yes" : "no")
#         # Calculate figure of merit
#         fom_values[i] = safe_value(fom_function(particles, σ_E, σ_z, E0))
        
#         # Calculate gradient if requested
#         if calculate_gradients
#             # Extract the gradient directly from the StochasticTriple
#             if typeof(σ_E) <: StochasticTriple && fom_function == energy_spread_fom
#                 gradient_values[i] = derivative_contribution(σ_E)
#             elseif typeof(σ_z) <: StochasticTriple && fom_function == bunch_length_fom
#                 gradient_values[i] = derivative_contribution(σ_z)
#             else
#                 # Fallback to StochasticModel approach
#                 stoch_model = create_stochastic_model(model_copy, fom_function)
#                 gradient_samples = [stochastic_gradient(stoch_model)[param_index] for _ in 1:n_gradient_samples]
#                 gradient_values[i] = mean(gradient_samples)
#                 gradient_uncertainties[i] = std(gradient_samples) / sqrt(n_gradient_samples)
#             end
#         end
#     end
    
#     return param_values, fom_values, gradient_values, gradient_uncertainties
# end

# function scan_parameter(
#     model::AcceleratorModel,
#     fom_function::Function,
#     param_index::Int,
#     param_range::AbstractVector{Float64};
#     calculate_gradients::Bool=true,
#     n_gradient_samples::Int=5
#     )
#     # Initialize result arrays
#     param_values = copy(param_range)
#     fom_values = zeros(length(param_range))
#     gradient_values = calculate_gradients ? zeros(length(param_range)) : Float64[]
#     gradient_uncertainties = calculate_gradients ? zeros(length(param_range)) : Float64[]
    
#     # Scan parameter
#     for (i, param_value) in enumerate(param_range)
#         # Create a fresh copy of the model
#         model_copy = deepcopy(model)
        
#         # Use regular Float64 parameters - no StochasticTriple here
#         model_copy.params[param_index] = param_value
        
#         # Run model to get FOM value
#         particles, σ_E, σ_z, E0 = run_model(model_copy, model_copy.params)
        
#         # Calculate figure of merit
#         fom_values[i] = fom_function(particles, σ_E, σ_z, E0)
        
#         # Calculate gradient using StochasticAD.derivative_estimate if requested
#         if calculate_gradients
#             # Define a function wrapper for what we're differentiating
#             function param_to_metric(p)
#                 # Create a new model copy for each evaluation
#                 local model_inner = deepcopy(model)
#                 model_inner.params[param_index] = p
                
#                 # Run model
#                 local part, local_σ_E, local_σ_z, local_E0 = run_model(model_inner, model_inner.params)
                
#                 # Return the metric of interest
#                 if fom_function == energy_spread_fom
#                     return local_σ_E
#                 elseif fom_function == bunch_length_fom
#                     return local_σ_z
#                 else
#                     return fom_function(part, local_σ_E, local_σ_z, local_E0)
#                 end
#             end
            
#             # Collect multiple gradient samples
#             grad_samples = zeros(n_gradient_samples)
#             for j in 1:n_gradient_samples
#                 # This is the key StochasticAD function that estimates the derivative
#                 grad_samples[j] = StochasticAD.derivative_estimate(param_to_metric, param_value)
#                 println("Gradient sample $j: $(grad_samples[j])")
#             end
            
#             # Calculate mean and uncertainty
#             gradient_values[i] = mean(grad_samples)
#             gradient_uncertainties[i] = std(grad_samples) / sqrt(n_gradient_samples)
#             println("Parameter $param_value: Gradient = $(gradient_values[i]) ± $(gradient_uncertainties[i])")
#         end
#     end
    
#     return param_values, fom_values, gradient_values, gradient_uncertainties
# end

# function scan_parameter(
#     model::AcceleratorModel,
#     fom_function::Function,
#     param_index::Int,
#     param_range::AbstractVector{Float64};
#     calculate_gradients::Bool=true,
#     n_gradient_samples::Int=5
#     )
#     # Initialize result arrays
#     param_values = copy(param_range)
#     fom_values = zeros(length(param_range))
#     gradient_values = calculate_gradients ? zeros(length(param_range)) : Float64[]
#     gradient_uncertainties = calculate_gradients ? zeros(length(param_range)) : Float64[]
    
#     # Scan parameter
#     for (i, param_value) in enumerate(param_range)
#         println("Processing parameter value: $param_value")
        
#         # Define a function that we can differentiate directly
#         function direct_param_function(p)
#             # Create a fresh model
#             local model_inner = deepcopy(model)
#             # Set the parameter
#             model_inner.params[param_index] = p
#             # Run simulation
#             local particles, local_σ_E, local_σ_z, local_E0 = run_model(model_inner, model_inner.params)
#             # Return metric based on function
#             if fom_function == energy_spread_fom
#                 return local_σ_E
#             elseif fom_function == bunch_length_fom
#                 return local_σ_z
#             else
#                 return fom_function(particles, local_σ_E, local_σ_z, local_E0)
#             end
#         end
        
#         # Run with regular parameter to get FOM value
#         regular_result = direct_param_function(param_value)
#         fom_values[i] = regular_result
        
#         # Calculate gradient using stochastic_triple if requested
#         if calculate_gradients
#             gradient_samples = Float64[]
            
#             # Take multiple samples
#             for j in 1:n_gradient_samples
#                 # Create a stochastic triple to track derivatives
#                 st = StochasticAD.stochastic_triple(direct_param_function, param_value)
#                 grad = StochasticAD.derivative_contribution(st)
#                 println("Gradient sample $j: $grad")
#                 push!(gradient_samples, grad)
#             end
            
#             # Calculate mean and uncertainty
#             gradient_values[i] = mean(gradient_samples)
#             gradient_uncertainties[i] = std(gradient_samples) / sqrt(length(gradient_samples))
#             println("Parameter $param_value: Gradient = $(gradient_values[i]) ± $(gradient_uncertainties[i])")
#         end
#     end
    
#     return param_values, fom_values, gradient_values, gradient_uncertainties
# end

# function scan_parameter(
#     model::AcceleratorModel,
#     fom_function::Function,
#     param_index::Int,
#     param_ranges::AbstractVector{Float64};
#     calculate_gradients::Bool=true,
#     n_gradient_samples::Int=5
#     )
#     # Initialize result arrays
#     param_values = copy(param_ranges)
#     fom_values = zeros(length(param_ranges))
#     gradient_values = calculate_gradients ? zeros(length(param_ranges)) : Float64[]
#     gradient_uncertainties = calculate_gradients ? zeros(length(param_ranges)) : Float64[]
#     gradient_grids = calculate_gradients ? [] : nothing
    
#     # Scan parameter values
#     for (i, param_value) in enumerate(param_ranges)
#         # Create a model copy with this parameter value
#         model_copy = deepcopy(model)
#         model_copy.params[param_index] = param_value
        
#         # Run model to get FOM value
#         particles, σ_E, σ_z, E0 = run_model(model_copy, model_copy.params)
#         fom_values[i] = fom_function(particles, σ_E, σ_z, E0)
        
#         if calculate_gradients
#             # Define a clean function that goes directly from parameter to FOM
#             parameter_to_fom(p) = begin
#                 # Make a fresh copy of the model
#                 m = deepcopy(model)
#                 # Set the parameter we're scanning
#                 m.params[param_index] = p
#                 # Run the model
#                 part, se, sz, e0 = run_model(m, m.params)
#                 # Return the appropriate metric
#                 return fom_function(part, se, sz, e0)
#             end
            
#             # Use StochasticAD's derivative_estimate to get unbiased gradient
#             grad_samples = Float64[]
#             for j in 1:n_gradient_samples
#                 # Get a derivative estimate using StochasticAD
#                 grad = StochasticAD.derivative_estimate(parameter_to_fom, param_value)
#                 push!(grad_samples, grad)
#                 println("Gradient sample $j: $grad")
#             end
            
#             # Calculate mean and standard deviation
#             gradient_values[i] = mean(grad_samples)
#             gradient_uncertainties[i] = std(grad_samples) / sqrt(n_gradient_samples)
#             println("Parameter $param_value: gradient = $(gradient_values[i]) ± $(gradient_uncertainties[i])")
#         end
#     end
    
#     return param_values, fom_values, gradient_values, gradient_uncertainties
# end 
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
    
    # Scan parameter
    for (i, param_value) in enumerate(param_range)
        println("Processing parameter value: $param_value")
        
        # Define a wrapper function for StochasticAD
        function parameter_to_fom(p)
            # Create a fresh model copy
            local model_copy = deepcopy(model)
            # Set the parameter
            model_copy.params[param_index] = p
            # Run model
            local particles, local_σ_E, local_σ_z, local_E0 = run_model(model_copy, model_copy.params)
            
            # Return appropriate metric
            if fom_function == energy_spread_fom
                return local_σ_E
            elseif fom_function == bunch_length_fom
                return local_σ_z
            else
                return fom_function(particles, local_σ_E, local_σ_z, local_E0)
            end
        end
        
        # Run model with current parameter value to get FOM
        model_copy = deepcopy(model)
        model_copy.params[param_index] = param_value
        particles, σ_E, σ_z, E0 = run_model(model_copy, model_copy.params)
        fom_values[i] = fom_function(particles, σ_E, σ_z, E0)
        
        # Calculate gradient if requested
        if calculate_gradients
            # Collect multiple samples to reduce variance
            samples = Float64[]
            for j in 1:n_gradient_samples
                grad = StochasticAD.derivative_estimate(parameter_to_fom, param_value)
                println("  Gradient sample $j: $grad")
                push!(samples, grad)
            end
            
            # Calculate mean and uncertainty
            gradient_values[i] = mean(samples)
            gradient_uncertainties[i] = std(samples) / sqrt(n_gradient_samples)
            println("  Parameter $param_value: Gradient = $(gradient_values[i]) ± $(gradient_uncertainties[i])")
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
# function multi_parameter_scan(
#     model::AcceleratorModel,
#     fom_function::Function,
#     param_indices::Vector{Int},
#     param_ranges::Vector{<:AbstractVector{<:Real}}
# )
#     # Check that we have two parameters at most
#     @assert length(param_indices) <= 2 "Cannot scan more than 2 parameters at once"
    
#     # Create copy of model
#     model_copy = deepcopy(model)
    
#     if length(param_indices) == 1
#         # Single parameter scan
#         param_values, fom_values, _, _ = scan_parameter(
#             model_copy,
#             fom_function,
#             param_indices[1],
#             param_ranges[1];
#             calculate_gradients=false
#         )
        
#         return [param_values], fom_values
#     else
#         # Two parameter scan
#         param1_range = param_ranges[1]
#         param2_range = param_ranges[2]
        
#         # Initialize result grid
#         fom_grid = zeros(length(param1_range), length(param2_range))
        
#         # Scan parameters
#         for (i, param1_value) in enumerate(param1_range)
#             for (j, param2_value) in enumerate(param2_range)
#                 # Update parameters
#                 model_copy.params[param_indices[1]] = param1_value
#                 model_copy.params[param_indices[2]] = param2_value
                
#                 # Run model
#                 particles, σ_E, σ_z, E0 = run_model(model_copy, model_copy.params)
                
#                 # Calculate figure of merit
#                 fom_grid[i, j] = fom_function(particles, σ_E, σ_z, E0)
#             end
#         end
        
#         return [param1_range, param2_range], fom_grid
#     end
# end

# function multi_parameter_scan(
#     model::AcceleratorModel,
#     fom_function::Function,
#     param_indices::Vector{Int},
#     param_ranges::Vector{<:AbstractVector{<:Real}}
#     )
#     # Check that we have two parameters at most
#     @assert length(param_indices) <= 2 "Cannot scan more than 2 parameters at once"
    
#     # Create copy of model
#     model_copy = deepcopy(model)
    
#     if length(param_indices) == 1
#         # Single parameter scan with gradients
#         param_values, fom_values, gradient_values, _ = scan_parameter(
#             model_copy,
#             fom_function,
#             param_indices[1],
#             param_ranges[1];
#             calculate_gradients=true
#         )
        
#         return [param_values], fom_values, [gradient_values]
#     else
#         # Two parameter scan
#         param1_range = param_ranges[1]
#         param2_range = param_ranges[2]
        
#         # Initialize result grid
#         fom_grid = zeros(length(param1_range), length(param2_range))
#         # Initialize gradient grids - one for each parameter
#         gradient1_grid = zeros(length(param1_range), length(param2_range))
#         gradient2_grid = zeros(length(param1_range), length(param2_range))
        
#         # Scan parameters
#         for (i, param1_value) in enumerate(param1_range)
#             for (j, param2_value) in enumerate(param2_range)
#                 # Update parameters
#                 model_copy.params[param_indices[1]] = param1_value
#                 model_copy.params[param_indices[2]] = param2_value
                
#                 # Run model
#                 particles, σ_E, σ_z, E0 = run_model(model_copy, model_copy.params)
                
#                 # Calculate figure of merit
#                 fom_grid[i, j] = fom_function(particles, σ_E, σ_z, E0)
                
#                 # Calculate gradients using StochasticAD
#                 stoch_model = create_stochastic_model(model_copy, fom_function)
#                 gradient = stochastic_gradient(stoch_model)
                
#                 # Extract gradients for each parameter
#                 gradient1_grid[i, j] = gradient.p[param_indices[1]]
#                 gradient2_grid[i, j] = gradient.p[param_indices[2]]
#             end
#         end
        
#         return [param1_range, param2_range], fom_grid, [gradient1_grid, gradient2_grid]
#     end
# end
# function multi_parameter_scan(
#     model::AcceleratorModel,
#     fom_function::Function,
#     param_indices::Vector{Int},
#     param_ranges::Vector{<:AbstractVector{<:Real}}
#     )
#     # Check that we have two parameters at most
#     @assert length(param_indices) <= 2 "Cannot scan more than 2 parameters at once"
    
#     if length(param_indices) == 1
#         # Single parameter scan with gradients
#         param_values, fom_values, gradient_values, _ = scan_parameter(
#             model,
#             fom_function,
#             param_indices[1],
#             param_ranges[1];
#             calculate_gradients=true
#         )
        
#         return [param_values], fom_values, [gradient_values]
#     else
#         # Two parameter scan
#         param1_range = param_ranges[1]
#         param2_range = param_ranges[2]
        
#         # Initialize result grids
#         fom_grid = zeros(length(param1_range), length(param2_range))
#         gradient1_grid = zeros(length(param1_range), length(param2_range))
#         gradient2_grid = zeros(length(param1_range), length(param2_range))
        
#         # Scan parameters
#         for (i, param1_value) in enumerate(param1_range)
#             for (j, param2_value) in enumerate(param2_range)
#                 # Define a function for parameter 1
#                 param1_to_fom(p1) = begin
#                     m = deepcopy(model)
#                     m.params[param_indices[1]] = p1
#                     m.params[param_indices[2]] = param2_value
#                     part, se, sz, e0 = run_model(m, m.params)
#                     return fom_function(part, se, sz, e0)
#                 end
                
#                 # Define a function for parameter 2
#                 param2_to_fom(p2) = begin
#                     m = deepcopy(model)
#                     m.params[param_indices[1]] = param1_value 
#                     m.params[param_indices[2]] = p2
#                     part, se, sz, e0 = run_model(m, m.params)
#                     return fom_function(part, se, sz, e0)
#                 end
                
#                 # Run model to get FOM value
#                 model_copy = deepcopy(model)
#                 model_copy.params[param_indices[1]] = param1_value
#                 model_copy.params[param_indices[2]] = param2_value
#                 particles, σ_E, σ_z, E0 = run_model(model_copy, model_copy.params)
#                 fom_grid[i, j] = fom_function(particles, σ_E, σ_z, E0)
                
#                 # Calculate gradients using StochasticAD
#                 gradient1_grid[i, j] = StochasticAD.derivative_estimate(param1_to_fom, param1_value)
#                 gradient2_grid[i, j] = StochasticAD.derivative_estimate(param2_to_fom, param2_value)
#             end
#         end
        
#         return [param1_range, param2_range], fom_grid, [gradient1_grid, gradient2_grid]
#     end
# end

function multi_parameter_scan(
    model::AcceleratorModel,
    fom_function::Function,
    param_indices::Vector{Int},
    param_ranges::Vector{<:AbstractVector{<:Real}}
)
    # Check that we have two parameters at most
    @assert length(param_indices) <= 2 "Cannot scan more than 2 parameters at once"
    
    if length(param_indices) == 1
        # Single parameter scan
        param_values, fom_values, gradient_values, gradient_uncertainties = scan_parameter(
            model,
            fom_function,
            param_indices[1],
            param_ranges[1];
            calculate_gradients=true
        )
        
        return [param_values], fom_values, [gradient_values]
    else
        # Two parameter scan
        param1_range = param_ranges[1]
        param2_range = param_ranges[2]
        
        # Initialize result grids
        fom_grid = zeros(length(param1_range), length(param2_range))
        gradient1_grid = zeros(length(param1_range), length(param2_range))
        gradient2_grid = zeros(length(param1_range), length(param2_range))
        
        # Scan parameters
        for (i, param1_value) in enumerate(param1_range)
            for (j, param2_value) in enumerate(param2_range)
                println("Processing parameters: ($(param1_value), $(param2_value))")
                
                # Create wrapper functions for differentiation
                param1_to_fom(p1) = begin
                    local model_copy = deepcopy(model)
                    model_copy.params[param_indices[1]] = p1
                    model_copy.params[param_indices[2]] = param2_value
                    local particles, local_σ_E, local_σ_z, local_E0 = run_model(model_copy, model_copy.params)
                    
                    # Return appropriate metric
                    if fom_function == energy_spread_fom
                        return local_σ_E
                    elseif fom_function == bunch_length_fom
                        return local_σ_z
                    else
                        return fom_function(particles, local_σ_E, local_σ_z, local_E0)
                    end
                end
                
                param2_to_fom(p2) = begin
                    local model_copy = deepcopy(model)
                    model_copy.params[param_indices[1]] = param1_value
                    model_copy.params[param_indices[2]] = p2
                    local particles, local_σ_E, local_σ_z, local_E0 = run_model(model_copy, model_copy.params)
                    
                    # Return appropriate metric
                    if fom_function == energy_spread_fom
                        return local_σ_E
                    elseif fom_function == bunch_length_fom
                        return local_σ_z
                    else
                        return fom_function(particles, local_σ_E, local_σ_z, local_E0)
                    end
                end
                
                # Run model to get FOM value
                model_copy = deepcopy(model)
                model_copy.params[param_indices[1]] = param1_value
                model_copy.params[param_indices[2]] = param2_value
                particles, σ_E, σ_z, E0 = run_model(model_copy, model_copy.params)
                fom_value = fom_function(particles, σ_E, σ_z, E0)
                fom_grid[i, j] = fom_value
                
                # Calculate gradients
                gradient1_grid[i, j] = StochasticAD.derivative_estimate(param1_to_fom, param1_value)
                gradient2_grid[i, j] = StochasticAD.derivative_estimate(param2_to_fom, param2_value)
                
                println("  FOM: $fom_value")
                println("  Gradient1: $(gradient1_grid[i, j])")
                println("  Gradient2: $(gradient2_grid[i, j])")
            end
        end
        
        return [param1_range, param2_range], fom_grid, [gradient1_grid, gradient2_grid]
    end
end