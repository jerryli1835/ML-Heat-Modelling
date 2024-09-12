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

# PDE definitions
include("heat_conduction_mtk.jl")

# functions of solving PDE with numerical and PINN methods
include("deep_learning_heat.jl")


# experiment 1, short time/space domain, with sine wave as initial condition
#set up domain and initial conditions
t_domain = (0.0, 1.0)
z_domain = (0.0, 1.0)
int_1(x) = sin(pi*x)
# obtain solutions
ref1 = simulate_heat_conduction_simple(t_domain, z_domain, T0=int_1, α=0.1);
pinn1 = fit_pinn_heat_conduction_simple(t_domain, z_domain, T0=int_1, maxiters=1000, α=0.1);
# plot graphs
cm = cgrad(:RdBu, rev=true);
plt1 = heatmap(collect(LinRange(t_domain[1], t_domain[end], 100)), collect(LinRange(z_domain[1], z_domain[end], 100)), (t,z) -> ref1.sol(t,z)[1], yflip=true, cmap=cm, title="Numerical")
plt2 = heatmap(collect(LinRange(t_domain[1], t_domain[end], 100)), collect(LinRange(z_domain[1], z_domain[end], 100)), (t,z) -> pinn1.phi([t,z], pinn1.res.u)[1], yflip=true, cmap=cm, title="PINN")
plt3 = heatmap(collect(LinRange(t_domain[1], t_domain[end], 100)), collect(LinRange(z_domain[1], z_domain[end], 100)), (t,z) -> pinn1.phi([t,z], pinn1.res.u)[1] - ref1.sol(t,z)[1], yflip=true, cmap=cm, title="Error")
plot(plt1, plt2, plt3, dpi=150)
savefig("pinn_experiment1.png")


# experiment 2, larger time domain,  with sine wave as initial condition
# set up domain and initial conditions
t_domain = (0.0, 10.0)
z_domain = (0.0, 1.0)
int_2(x) = sin(pi*x)
# obtain solutions
ref2 = simulate_heat_conduction_simple(t_domain, z_domain, T0=int_2, α=1e-3);
pinn2 = fit_pinn_heat_conduction_simple(t_domain, z_domain, T0=int_2, maxiters=1500, α=1e-3);
# plot graphs
cm = cgrad(:RdBu, rev=true)
plt1 = heatmap(collect(LinRange(t_domain[1], t_domain[end], 100)), collect(LinRange(z_domain[1], z_domain[end], 100)), (t,z) -> ref2.sol(t,z)[1], yflip=true, cmap=cm, title="Numerical")
plt2 = heatmap(collect(LinRange(t_domain[1], t_domain[end], 100)), collect(LinRange(z_domain[1], z_domain[end], 100)), (t,z) -> pinn2.phi([t,z], cpu(pinn2.res.u))[1], yflip=true, cmap=cm, title="PINN")
plt3 = heatmap(collect(LinRange(t_domain[1], t_domain[end], 100)), collect(LinRange(z_domain[1], z_domain[end], 100)), (t,z) -> pinn2.phi([t,z], pinn2.res.u)[1] - ref2.sol(t,z)[1], yflip=true, cmap=cm, title="Error")
plot(plt1, plt2, plt3, dpi=150)
savefig("pinn_experiment2.png")


# experiment 3, larger space domain, with sine wave as initial condition
# set up domain and initial conditions
t_domain = (0.0, 1.0)
z_domain = (0.0, 10.0)
int_3(x) = sin(0.2*pi*x)
# obtain solutions
ref3 = simulate_heat_conduction_simple(t_domain, z_domain, T0=int_3, α=0.1);
pinn3 = fit_pinn_heat_conduction_simple(t_domain, z_domain, T0=int_3, maxiters=1500, α=0.1);
# plot graphs
cm = cgrad(:RdBu, rev=true);
plt1 = heatmap(collect(LinRange(t_domain[1], t_domain[end], 100)), collect(LinRange(z_domain[1], z_domain[end], 100)), (t,z) -> ref3.sol(t,z)[1], yflip=true, cmap=cm, title="Numerical")
plt2 = heatmap(collect(LinRange(t_domain[1], t_domain[end], 100)), collect(LinRange(z_domain[1], z_domain[end], 100)), (t,z) -> pinn3.phi([t,z], cpu(pinn3.res.u))[1], yflip=true, cmap=cm, title="PINN")
plt3 = heatmap(collect(LinRange(t_domain[1], t_domain[end], 100)), collect(LinRange(z_domain[1], z_domain[end], 100)), (t,z) -> pinn3.phi([t,z], pinn3.res.u)[1] - ref3.sol(t,z)[1], yflip=true, cmap=cm, title="Error")
plot(plt1, plt2, plt3, dpi=150)
savefig("pinn_experiment3.png")


# experiment 4, increased both time and space domain, with sine wave as initial condition
# set up domain and initial conditions
t_domain = (0.0, 50.0)
z_domain = (0.0, 50.0)
int_4(x) = sin(0.2*pi*x)
# obtain solutions
ref4 = simulate_heat_conduction_simple(t_domain, z_domain, T0=int_4, α=1e-3);
pinn4 = fit_pinn_heat_conduction_simple(t_domain, z_domain, T0=int_4, maxiters=1000, α=1e-3, opt=LBFGS());
# plot graphs
cm = cgrad(:RdBu, rev=true);
plt1 = heatmap(collect(LinRange(t_domain[1], t_domain[end], 100)), collect(LinRange(z_domain[1], z_domain[end], 100)), (t,z) -> ref4.sol(t,z)[1], yflip=true, cmap=cm, title="Numerical")
plt2 = heatmap(collect(LinRange(t_domain[1], t_domain[end], 100)), collect(LinRange(z_domain[1], z_domain[end], 100)), (t,z) -> pinn4.phi([t,z], cpu(pinn4.res.u))[1], yflip=true, cmap=cm, title="PINN")
plt3 = heatmap(collect(LinRange(t_domain[1], t_domain[end], 100)), collect(LinRange(z_domain[1], z_domain[end], 100)), (t,z) -> pinn4.phi([t,z], pinn4.res.u)[1] - ref4.sol(t,z)[1], yflip=true, cmap=cm, title="Error")
plot(plt1, plt2, plt3, dpi=150)
savefig("pinn_experiment4.png")


# experiment 5, short time/space domain, with sine wave as initial condition and using a hard constraint setting for initial condition
# set up domain and initial conditions
t_domain = (0.0, 1.0)
z_domain = (0.0, 1.0)
int_5(x) = sin(pi*x)
# set up the hard constraint
f1 = Lux.Parallel(
    (x,y) -> x.*y,
    WrappedFunction(x -> IfElse.ifelse.(x[1:1,:] .> 0, 1.0, 0.0)),
    Lux.Chain(
        Dense(2, 16),
        Dense(16, 16, Lux.σ),
        Dense(16, 16, Lux.σ),
        Dense(16, 1),
    ),
)
f2 = WrappedFunction(x -> sin.(π*x[2:2,:]))
new_network = Parallel(
    +,
    f1,
    f2,
)
# obtain solutions
ref5 = simulate_heat_conduction_simple(t_domain, z_domain, T0=int_5, α=0.4);
pinn5 = fit_pinn_heat_conduction_simple(t_domain, z_domain, network=new_network, T0=int_5, maxiters=500, α=0.4, opt=LBFGS());
# plot graphs
cm = cgrad(:RdBu, rev=true);
plt1 = heatmap(collect(LinRange(t_domain[1], t_domain[end], 100)), collect(LinRange(z_domain[1], z_domain[end], 100)), (t,z) -> ref5.sol(t,z)[1], yflip=true, cmap=cm, title="Numerical")
plt2 = heatmap(collect(LinRange(t_domain[1], t_domain[end], 100)), collect(LinRange(z_domain[1], z_domain[end], 100)), (t,z) -> pinn5.phi([t,z], pinn5.res.u)[1], yflip=true, cmap=cm, title="PINN")
plt3 = heatmap(collect(LinRange(t_domain[1], t_domain[end], 100)), collect(LinRange(z_domain[1], z_domain[end], 100)), (t,z) -> pinn5.phi([t,z], pinn5.res.u)[1] - ref5.sol(t,z)[1], yflip=true, cmap=cm, title="Error")
plot(plt1, plt2, plt3, dpi=150)
savefig("pinn_experiment5.png")


# experiment 6, short time/space domain, with a linear function as initial condition
# set up domain and initial conditions
t_domain = (0.0, 1.0)
z_domain = (0.0, 1.0)
# initial condition
int_6(x) = 0.5*x
# obtain solutions
ref6 = simulate_heat_conduction_simple(t_domain, z_domain, T0=int_6, α=0.1);
pinn6 = fit_pinn_heat_conduction_simple(t_domain, z_domain, T0=int_6, maxiters=1000, α=0.1);
# plot graphs
cm = cgrad(:RdBu, rev=true);
plt1 = heatmap(collect(LinRange(t_domain[1], t_domain[end], 100)), collect(LinRange(z_domain[1], z_domain[end], 100)), (t,z) -> ref6.sol(t,z)[1], yflip=true, cmap=cm, title="Numerical")
plt2 = heatmap(collect(LinRange(t_domain[1], t_domain[end], 100)), collect(LinRange(z_domain[1], z_domain[end], 100)), (t,z) -> pinn6.phi([t,z], pinn6.res.u)[1], yflip=true, cmap=cm, title="PINN")
plt3 = heatmap(collect(LinRange(t_domain[1], t_domain[end], 100)), collect(LinRange(z_domain[1], z_domain[end], 100)), (t,z) -> pinn6.phi([t,z], pinn6.res.u)[1] - ref6.sol(t,z)[1], yflip=true, cmap=cm, title="Error")
plot(plt1, plt2, plt3, dpi=150)
savefig("pinn_experiment6.png")


# experiment 7, moderate time/space domain, with sine waves as initial condition, for nonlinear case
# set up domain and initial conditions
t_domain = (0.0, 10.0)
z_domain = (0.0, 10.0)
# obtain solutions
ref7 = simulate_heat_conduction_simple(t_domain, z_domain, pde=nonlinear_heat_conduction);
pinn7 = fit_pinn_heat_conduction_simple(t_domain, z_domain, maxiters=1000, opt=BFGS(), pde=nonlinear_heat_conduction);
# plot graphs
cm = cgrad(:RdBu, rev=true);
plt1 = heatmap(collect(LinRange(t_domain[1], t_domain[end], 100)), collect(LinRange(z_domain[1], z_domain[end], 100)), (t,z) -> ref7.sol(t,z)[1], yflip=true, cmap=cm, title="Numerical")
plt2 = heatmap(collect(LinRange(t_domain[1], t_domain[end], 100)), collect(LinRange(z_domain[1], z_domain[end], 100)), (t,z) -> pinn7.phi([t,z], pinn7.res.u)[1], yflip=true, cmap=cm, title="PINN")
plt3 = heatmap(collect(LinRange(t_domain[1], t_domain[end], 100)), collect(LinRange(z_domain[1], z_domain[end], 100)), (t,z) -> pinn7.phi([t,z], pinn7.res.u)[1] - ref7.sol(t,z)[1], yflip=true, cmap=cm, title="Error")
plot(plt1, plt2, plt3, dpi=150)
savefig("pinn_experiment7.png")
