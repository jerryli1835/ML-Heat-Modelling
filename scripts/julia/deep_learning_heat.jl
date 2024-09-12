# Neural networks and optimization
using Lux
using NeuralPDE
using Optimization, OptimizationOptimJL, OptimizationOptimisers
import ModelingToolkit: Interval
# Packages for numerical solutions
using MethodOfLines
using OrdinaryDiffEq

# Other utilities
using ComponentArrays
using Plots
using Random

# PDE definitions
include("heat_conduction_mtk.jl")

# Fourier Features layers
include("fourier_features_layer.jl")

# function for generating refernce simulation for PINN using numercial method
function simulate_heat_conduction_simple(t_domain, z_domain; pde=linear_heat_conduction, num_cells=100, kwargs...)
    heat_pde_system, ivars, dvars = pde(; t_domain, z_domain, kwargs...)
    t, z = ivars
    dz = (z_domain[2] - z_domain[1]) / num_cells
    
    # solve with method of lines
    @info "Running reference simulation (dz=$dz)"
    mol_discretization = MOLFiniteDifference([z => dz], t, approx_order=2)
    mol_prob = discretize(heat_pde_system, mol_discretization)
    mol_sol = solve(mol_prob, Rodas4P2(), abstol=1e-6, reltol=1e-8)
    if mol_sol.retcode != ReturnCode.Success
        @warn "Numerical solver did not converge; retcode=$(mol_sol.retcode)"
    end
    return (
        prob = mol_prob,
        sol = mol_sol,
    )
end


# function for setting up PDE probelm for refernce data in Deep O Net using numercial method
function setup_pde_problem(t_domain, z_domain; pde=linear_heat_conduction, num_cells=100, kwargs...)
    heat_pde_system, ivars, dvars = pde(; t_domain, z_domain, kwargs...)
    t, z = ivars
    dz = (z_domain[2] - z_domain[1]) / (num_cells+1)
    # solve with method of lines
    mol_discretization = MOLFiniteDifference([z => dz], t, approx_order=2)
    mol_prob = discretize(heat_pde_system, mol_discretization)
    zs = first(values(mol_prob.problem_type.discretespace.grid))
    return mol_prob, zs
end


# function for setting up PDE probelm with parameters for refernce data in Deep O Net using numercial method
function setup_pde_problem_with_params(t_domain, z_domain; pde=linear_heat_conduction, num_cells=100, kwargs...)
    heat_pde_system, ivars, dvars, params = pde(; t_domain, z_domain, kwargs...)
    t, z = ivars
    dz = (z_domain[2] - z_domain[1]) / (num_cells+1)
    # solve with method of lines
    mol_discretization = MOLFiniteDifference([z => dz], t, approx_order=2)
    mol_prob = discretize(heat_pde_system, mol_discretization)
    zs = first(values(mol_prob.problem_type.discretespace.grid))
    return mol_prob, zs
end


# function for solving PDE probelm for refernce data in Deep O Net using numercial method
function solve_pde_problem(prob, solver=CVODE_BDF(; linear_solver=:Band, jac_upper=1, jac_lower=1))
    return solve(prob, solver)
end


# function for training a PINN using Neural PDE
function fit_pinn_heat_conduction_simple(t_domain, z_domain; pde=linear_heat_conduction, 
    network=Lux.Chain(Dense(2, 36),Lux.Dense(36, 36, Lux.σ),Lux.Dense(36, 36, Lux.σ),Lux.Dense(36, 1)), maxiters=1000, strategy=GridTraining(0.05), opt=BFGS(), init_params=nothing, kwargs...)
    heat_pde_system, ivars, dvars = pde(; t_domain, z_domain, kwargs...)
    t, z = ivars

    # now solve with PINN
    @info "Fitting PINN"
    ps, st = Lux.setup(Random.default_rng(), network)
    if isnothing(init_params)
        ps = ps |> ComponentArray
    else
        ps = init_params
    end
    pinn_discretization = PhysicsInformedNN(network, strategy, init_params=ps)
    sym_prob = symbolic_discretize(heat_pde_system, pinn_discretization)
    phi = sym_prob.phi

    pde_loss_functions = sym_prob.loss_functions.pde_loss_functions
    bc_loss_functions = sym_prob.loss_functions.bc_loss_functions

    callback = function (p, l)
        println("loss: ", l)
        println("pde_losses: ", map(l_ -> l_(p), pde_loss_functions))
        println("bcs_losses: ", map(l_ -> l_(p), bc_loss_functions))
        return false
    end

    loss_functions =  [pde_loss_functions;bc_loss_functions]

    function loss_function(θ,p)
        sum(map(l->l(θ) ,loss_functions))
    end

    f_ = OptimizationFunction(loss_function, Optimization.AutoZygote())
    prob = Optimization.OptimizationProblem(f_, sym_prob.flat_init_params)

    res = Optimization.solve(prob,OptimizationOptimJL.BFGS(); callback = callback, maxiters=1000)
    return (
        model = network,
        prob = sym_prob,
        phi = pinn_discretization.phi,
        res = res,
    )
end
