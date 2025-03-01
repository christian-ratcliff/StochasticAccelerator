"""
StochasticAccelerator.jl - Main module for stochastic Haissinski simulations

This module implements a high-performance beam evolution simulation for particle accelerators
with StochasticAD integration for parameter sensitivity analysis.
"""
module StochasticAccelerator

# Standard library imports
using Statistics
using LinearAlgebra
using Random

# External dependencies
using StochasticAD
using Distributions
using StructArrays
using LoopVectorization
using FFTW
using Interpolations
using ProgressMeter
using FHist
using Plots
using LaTeXStrings
using Optimisers
using Base.Threads

# Include core files
include("core/constants.jl")
include("core/types.jl")
include("core/particles.jl")
include("core/utils.jl")

# Include physics processes
include("physics/rf_cavity.jl")
include("physics/synchrotron_radiation.jl")
include("physics/quantum_excitation.jl")
include("physics/wakefield.jl")

# Include simulation
include("simulation/evolution.jl")
include("simulation/diagnostics.jl")

# Include optimization
include("optimization/model.jl")
include("optimization/figures_of_merit.jl")
include("optimization/gradients.jl")
include("optimization/optimization.jl")

# Export core types and constants
export SPEED_LIGHT, ELECTRON_CHARGE, MASS_ELECTRON
export Coordinate, Particle, PhysicsProcess, SimulationParameters, SimulationBuffers

# Export physics processes
export RFCavity, SynchrotronRadiation, QuantumExcitation, Wakefield
export apply_process!

# Export simulation functions
export generate_particles, run_simulation, track_particles!, setup_simulation, run_model
export analyze_beam, calculate_statistics

# Export optimization functionality
export AcceleratorModel, BeamParameter, ParameterMapping
export create_stochastic_model, calculate_gradient, calculate_gradient_with_uncertainty, create_accelerator_model
export optimize_parameters, scan_parameter, multi_parameter_scan
export energy_spread_fom, bunch_length_fom, emittance_fom


end # module