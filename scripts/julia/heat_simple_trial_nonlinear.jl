using DiffEqBase
using MethodOfLines
using ModelingToolkit
using IntervalSets
using IfElse
using Lux
using NeuralPDE
using Optimization, OptimizationOptimJL, OptimizationOptimisers
import ModelingToolkit: Interval
# Packages for numerical solutions
using OrdinaryDiffEq

# Other utilities
using ComponentArrays
using Plots
using Random
using LineSearches

const default_conductivities = (
    kw = 0.57, # [W/Km^2] water [Hillel(1982)]
    ko = 0.25, # [W/Km^2] organic [Hillel(1982)]
    km = 3.8, # [W/Km^2] mineral [Hillel(1982)]
    ka = 0.025, # [W/Km^2] air [Hillel(1982)]
    ki = 2.2, # [W/Km^2] ice [Hillel(1982)]
)

const default_heat_capcities = (
    cw = 4.2*10^6, # [J/m^3K] heat capacity water
    co = 2.5*10^6, # [J/m^3K]  heat capacity organic
    cm = 2*10^6, # [J/m^3K]  heat capacity mineral
    ca = 0.00125*10^6, # [J/m^3K]  heat capacity pore space
    ci = 1.9*10^6, # [J/m^3K]  heat capacity ice
)
# including previously defined functions
include("heat_conduction_mtk.jl")
include("neural_pde_func_updated.jl")

# experiment 2.1, short time/space domain, with sine waves as initial condition, for nonlinear case
t_domain = (0.0, 10.0)
z_domain = (0.0, 10.0)

# strategy = QuasiRandomTraining(256)
ref1 = simulate_heat_conduction_simple(t_domain, z_domain, pde=nonlinear_heat_conduction);
pinn1 = fit_pinn_heat_conduction_simple(t_domain, z_domain, maxiters=1000, opt=BFGS(), pde=nonlinear_heat_conduction);

# plot graphs
cm = cgrad(:RdBu, rev=true);
plt1 = heatmap(collect(LinRange(t_domain[1], t_domain[end], 100)), collect(LinRange(z_domain[1], z_domain[end], 100)), (t,z) -> ref1.sol(t,z)[1], yflip=true, cmap=cm, title="Numerical")
plt2 = heatmap(collect(LinRange(t_domain[1], t_domain[end], 100)), collect(LinRange(z_domain[1], z_domain[end], 100)), (t,z) -> pinn1.phi([t,z], pinn1.res.u)[1], yflip=true, cmap=cm, title="PINN")
plt3 = heatmap(collect(LinRange(t_domain[1], t_domain[end], 100)), collect(LinRange(z_domain[1], z_domain[end], 100)), (t,z) -> pinn1.phi([t,z], pinn1.res.u)[1] - ref1.sol(t,z)[1], yflip=true, cmap=cm, title="Error")
plot(plt1, plt2, plt3, dpi=150)
savefig("heat_trail2.1.png")
