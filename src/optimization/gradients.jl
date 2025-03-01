"""
Gradient calculation for accelerator parameter optimization.

This file implements gradient calculation using StochasticAD
for optimizing accelerator parameters.
"""

using StochasticAD
using Statistics

"""
    calculate_gradient(
        model::AcceleratorModel{Float64},
        fom_function::Function
    ) -> Vector{Float64}

Calculate the gradient of a figure of merit with respect to model parameters.

# Arguments
- `model`: Accelerator model
- `fom_function`: Figure of merit function

# Returns
- Gradient vector
"""
function calculate_gradient(
    model::AcceleratorModel{Float64},
    fom_function::Function
)
    # Create StochasticModel
    stoch_model = create_stochastic_model(model, fom_function)
    
    # Calculate gradient
    return stochastic_gradient(stoch_model)
end

"""
    calculate_gradient_with_uncertainty(
        model::AcceleratorModel{Float64},
        fom_function::Function;
        n_samples::Int=10
    ) -> Tuple{Vector{Float64}, Vector{Float64}}

Calculate the gradient of a figure of merit with uncertainty.

# Arguments
- `model`: Accelerator model
- `fom_function`: Figure of merit function
- `n_samples`: Number of samples for uncertainty estimation

# Returns
- Tuple of (mean_gradient, uncertainty)
"""
function calculate_gradient_with_uncertainty(
    model::AcceleratorModel{Float64},
    fom_function::Function;
    n_samples::Int=10
)
    # Create StochasticModel
    stoch_model = create_stochastic_model(model, fom_function)
    
    # Collect multiple gradient samples
    gradient_samples = [stochastic_gradient(stoch_model) for _ in 1:n_samples]
    
    # Calculate mean and standard deviation
    mean_gradient = mean(gradient_samples)
    uncertainty = std(gradient_samples) ./ sqrt(n_samples)
    
    return mean_gradient, uncertainty
end

"""
    estimate_jacobian(
        model::AcceleratorModel{Float64};
        n_samples::Int=5
    ) -> Tuple{Matrix{Float64}, Matrix{Float64}}

Estimate the Jacobian matrix of energy spread and bunch length
with respect to model parameters.

# Arguments
- `model`: Accelerator model
- `n_samples`: Number of samples for uncertainty estimation

# Returns
- Tuple of (jacobian, uncertainty)
"""
function estimate_jacobian(
    model::AcceleratorModel{Float64};
    n_samples::Int=5
)
    # Number of parameters
    n_params = length(model.params)
    
    # Initialize Jacobian and uncertainty matrices
    jacobian = zeros(2, n_params)
    uncertainty = zeros(2, n_params)
    
    # Calculate gradients for energy spread
    grad_E, unc_E = calculate_gradient_with_uncertainty(
        model, energy_spread_fom;
        n_samples=n_samples
    )
    
    # Calculate gradients for bunch length
    grad_z, unc_z = calculate_gradient_with_uncertainty(
        model, bunch_length_fom;
        n_samples=n_samples
    )
    
    # Fill Jacobian and uncertainty matrices
    jacobian[1, :] = grad_E
    jacobian[2, :] = grad_z
    uncertainty[1, :] = unc_E
    uncertainty[2, :] = unc_z
    
    return jacobian, uncertainty
end

"""
    calculate_finite_difference_gradient(
        model::AcceleratorModel{Float64},
        fom_function::Function;
        step_size::Float64=1e-6
    ) -> Vector{Float64}

Calculate the gradient using finite difference method (for validation).

# Arguments
- `model`: Accelerator model
- `fom_function`: Figure of merit function
- `step_size`: Step size for finite difference

# Returns
- Gradient vector
"""
function calculate_finite_difference_gradient(
    model::AcceleratorModel{Float64},
    fom_function::Function;
    step_size::Float64=1e-6
)
    # Number of parameters
    n_params = length(model.params)
    
    # Initialize gradient vector
    gradient = zeros(n_params)
    
    # Calculate baseline value
    particles_base, σ_E_base, σ_z_base, E0_base = run_model(model, model.params)
    fom_base = fom_function(particles_base, σ_E_base, σ_z_base, E0_base)
    
    # Calculate gradient for each parameter
    for i in 1:n_params
        # Create perturbed parameters
        params_plus = copy(model.params)
        params_plus[i] += step_size
        
        # Run model with perturbed parameters
        particles_plus, σ_E_plus, σ_z_plus, E0_plus = run_model(model, params_plus)
        fom_plus = fom_function(particles_plus, σ_E_plus, σ_z_plus, E0_plus)
        
        # Calculate finite difference
        gradient[i] = (fom_plus - fom_base) / step_size
    end
    
    return gradient
end