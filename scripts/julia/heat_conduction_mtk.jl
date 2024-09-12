########
# Simple heat conduction model (no forcings)
# Uses ModelingToolkit
########

using DiffEqBase
using MethodOfLines
using ModelingToolkit
using IntervalSets
using IfElse
using Statistics

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

"""
    linear_heat_conduction(; t_domain=(0.0,24*3600.0), z_domain=(0.0,100.0))

Simple linear 1D heat conduction on a rectangular domain.
"""
function linear_heat_conduction(;
    t_domain=(0.0,24*3600.0),
    z_domain=(0.0,10.0),
    α=0.1,
    T0=z -> sin(pi*z),
    T_ub=t -> 0.0,
    T_lb=t -> 0.0,
)
    dvars = @variables T(..)
    ivars = @parameters t z
    # params = @parameters α
    Dzz =  Differential(z) * Differential(z)
    Dt = Differential(t)
    eqs = [
        Dt(T(t,z)) ~ α*Dzz(T(t,z))
    ]
    # Space and time domains
    domains = [
        t ∈ Interval(t_domain...),
        z ∈ Interval(z_domain...),
    ]
    bcs = [
        T(t_domain[1],z) ~ T0(z),
        T(t,z_domain[1]) ~ T_ub(t), #sin(2π*t),
        T(t,z_domain[2]) ~ T_lb(t), #cos(2π*t),
    ]
    # TODO: report bug to SciML
    # expected form of parameter argument seems to differ bewteen MethodOfLines and NeuralPDE
    # ps = declare_params ? [α => α_def] : params
    # pdesys = PDESystem(eqs, bcs, domains, [t,z], [T(t,z)], ps, name=:heat, defaults=Dict(α => α_def))
    pdesys = PDESystem(eqs, bcs, domains, [t,z], [T(t,z)], name=:heat)
    return (; pdesys, ivars, dvars)
end

"""
    nonlinear_heat_conduction(cond, cond_params; t_domain=(0.0,24*3600.0), z_domain=(0.0,100.0))

Nonlinear heat conduction with conductivity function `cond(T, params...)` and the `cond_params`
which should be parameters created using `cond_params = @parameters ...`.
"""
function nonlinear_heat_conduction(;
    cond=(T,ks...) -> (1 + exp(-T^2)),
    cond_params=default_conductivities,
    hc=(T,cs...) -> (100+10*exp(-T^2)),
    hc_params=default_heat_capcities,
    t_domain=(0.0,24*3600.0),
    z_domain=(0.0,10.0),
    T0=z -> sin(pi*z),
    T_ub=t -> 0.0,
    T_lb=t -> 0.0
)
    dvars = @variables T(..)
    ivars = @parameters t z
    Dz = Differential(z)
    Dt = Differential(t)
    Dzz =  Differential(z) * Differential(z)
    eqs = [
        Dt(T(t,z)) ~ (-2*T(t,z)*exp(-T(t,z)^2)*(Dz(T(t,z)))^2 + cond(T(t,z), cond_params...)*Dzz(T(t,z))) / hc(T(t,z), hc_params...)
    ]
    # Space and time domains
    domains = [
        t ∈ Interval(t_domain...),
        z ∈ Interval(z_domain...),
    ]
    bcs = [
        T(t_domain[1],z) ~ T0(z),
        T(t,z_domain[1]) ~ T_ub(t),
        T(t,z_domain[2]) ~ T_lb(t),
    ]
    pdesys = PDESystem(eqs, bcs, domains, [t,z], [T(t,z)], name=:heat)
    return (; pdesys, ivars, dvars)
end

function nonlinear_heat_conduction_with_params(;
    cond=(T,k0,λ,σ) -> (k0 + σ*exp(-λ*T^2)),
    hc=(T,c0,λ,σ) -> (c0 + σ*exp(-λ*T^2)),
    param_values=(k0=1.0, c0=1.0, λ=1.0, σ=1.0),
    t_domain=(0.0,24*3600.0),
    z_domain=(0.0,10.0),
    T0=z -> sin(pi*z),
    T_ub=t -> 0.0, 
    T_lb=t -> 0.0
)
    dvars = @variables T(..)
    ivars = @parameters t z
    params = @parameters c0 k0 λ σ
    Dz = Differential(z)
    Dt = Differential(t)
    Dzz =  Differential(z) * Differential(z)
    eqs = [
        Dt(T(t,z)) ~ (-2*T(t,z)*exp(-T(t,z)^2)*(Dz(T(t,z)))^2 + cond(T(t,z), k0, λ, σ)*Dzz(T(t,z))) / hc(T(t,z), c0*1e5, λ, σ)
    ]
    # Space and time domains
    domains = [
        t ∈ Interval(t_domain...),
        z ∈ Interval(z_domain...),
    ]
    bcs = [
        T(t_domain[1],z) ~ T0(z),
        T(t,z_domain[1]) ~ T_ub(t),
        T(t,z_domain[2]) ~ T_lb(t),
    ]
    # pdesys = PDESystem(eqs, bcs, domains, [t,z], [T(t,z)], params, name=:heat, defaults=Dict(map(Pair, params, values(param_values))...))
    pdesys = PDESystem(eqs, bcs, domains, [t,z], [T(t,z)], collect(map(Pair, params, values(param_values))), name=:heat)
    return (; pdesys, ivars, dvars, params)
end
