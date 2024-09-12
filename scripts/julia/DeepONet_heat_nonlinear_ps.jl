using DiffEqBase
using MethodOfLines
using Optimization, OptimizationOptimJL, OptimizationOptimisers
import ModelingToolkit: Interval
# Packages for numerical solutions
using OrdinaryDiffEq
using DataDeps, CSV, DataFrames, MLUtils
# using NeuralOperators, Flux
# using CUDA, FluxTraining, BSON
# Other utilities
using ComponentArrays
using QuasiMonteCarlo
using Plots
using Random
using NPZ

include("heat_conduction_mtk.jl")
include("deep_learning_heat.jl")


#function for generating training data using numerical method and fixed grid sampling
function generate_train_data_fixedgrid_with_params(;num=500, grid_size_t=4*24, grid_size_z=0.5, num_params=4, t_domain = (0.0, 24*3600.0), z_domain=(0.0, 10.0), kwargs...)
    grid_t_num,grid_z_num = Int((t_domain[2]-t_domain[1])/grid_size_t +1), Int((z_domain[2]-z_domain[1])/grid_size_z +1)
    cord = zeros((2, grid_t_num*grid_z_num))
    grid_t = collect(LinRange(t_domain[1], t_domain[end], grid_t_num))
    grid_z = collect(LinRange(z_domain[1], z_domain[end], grid_z_num))
    #set up the pde problem (discritise once) for a random condition
    prob, zs = setup_pde_problem_with_params(t_domain, z_domain, pde=nonlinear_heat_conduction_with_params, T0=x -> 5*x)
    #generate an N by 2 matrix containing the grid points
    tmp = []
    for i in grid_t
        for j in grid_z
            push!(tmp, (i,j))
        end
    end
    for p in 1:length(tmp)
        cord[1,p] = tmp[p][1]
        cord[2,p] = tmp[p][2]
    end
    #generate the matrix containing training data points, and evaluated input functionals
    data = zeros((num,  grid_t_num*grid_z_num))
    X = zeros((num_params, num))
    for n in 1:num
        @info "Loop $n"
        new_p = [rand(0.5:0.01:2.50), rand(1.0:0.01:2.0), rand(1.0:0.01:2.0), rand(1.0:0.01:2.0)]
        newprob = remake(prob, p=new_p)
        sol = solve_pde_problem(newprob, Rodas4P2())
        data_n = [sol(cord[1,j], cord[2,j])[1] for j=1:length(tmp)]
        for p in 1:length(tmp)
            data[n,p] = data_n[p]
        end
        for q in 1:num_params
            X[q,n] = new_p[q]
        end
    end
    #return the values: input functional signals, grid points for PDE, and reference data for PDE
    return X, cord, data

end

#function for generating training data using numerical method, using quasiMonteCarlo sampling
function generate_train_data_quasi(;num=100, num_cord=1024, t_domain = (0.0, 3600), z_domain=(0.0, 10.0))
    #generate an N by 2 matrix containing the sample points
    cord = QuasiMonteCarlo.sample(num_cord, [t_domain[1], z_domain[1]], [t_domain[2], z_domain[2]], SobolSample())
    grid_ic = collect(LinRange(z_domain[1], z_domain[2], 21))
    #set up the pde problem (discritise once) for a random condition
    prob, zs = setup_pde_problem(t_domain, z_domain, pde=nonlinear_heat_conduction, T0=x -> x)

    #generate the matrix containing training data points, and evaluated input functionals
    data = zeros((num,  num_cord))
    X = zeros((length(grid_ic), num))
    for n in 1:num
        input_func(x) = 0.0001*n*x
        @info "Loop $n"
        newprob = remake(prob, u0=input_func.(zs))
        sol = solve_pde_problem(newprob, Rodas4P2())
        data_n = [sol(cord[1,j], cord[2,j])[1] for j=1:num_cord]
        for p in 1:num_cord
            data[n,p] = data_n[p]
        end
        func_val_n = [input_func(a) for a in grid_ic]
        for q in 1:length(grid_ic)
            X[q,n] = func_val_n[q]
        end
    end
    #return the values: input functional signals, grid points for PDE, and reference data for PDE
    return X, cord, data

end

#generate the training data for 1000 initial condiditons
X, y, data = generate_train_data_fixedgrid_with_params(num=1000)
X, y, data = generate_train_data_quasi(num=1000, num_cord=10000)
u0 = X'
xt = y'
npzwrite("heat_train_nonlinear_1h_params.npz", u0=u0, xt=xt, u=data)

#function for generating testing data using numerical method and fixed grid sampling
function generate_test_data_fixedgrid_with_params(;num=500, grid_size_t=4*24, grid_size_z=0.5, num_params=4, t_domain = (0.0, 24*3600.0), z_domain=(0.0, 10.0), kwargs...)
    grid_t_num,grid_z_num = Int((t_domain[2]-t_domain[1])/grid_size_t +1), Int((z_domain[2]-z_domain[1])/grid_size_z +1)
    cord = zeros((2, grid_t_num*grid_z_num))
    grid_t = collect(LinRange(t_domain[1], t_domain[end], grid_t_num))
    grid_z = collect(LinRange(z_domain[1], z_domain[end], grid_z_num))
    #set up the pde problem (discritise once) for a random condition
    prob, zs = setup_pde_problem_with_params(t_domain, z_domain, pde=nonlinear_heat_conduction_with_params, T0=x -> 5*x)
    #generate an N by 2 matrix containing the grid points
    tmp = []
    for i in grid_t
        for j in grid_z
            push!(tmp, (i,j))
        end
    end
    for p in 1:length(tmp)
        cord[1,p] = tmp[p][1]
        cord[2,p] = tmp[p][2]
    end
    #generate the matrix containing training data points, and evaluated input functionals
    data = zeros((num,  grid_t_num*grid_z_num))
    X = zeros((num_params, num))
    for n in 1:num
        @info "Loop $n"
        new_p = [rand(0.5:0.01:2.50), rand(1.0:0.01:2.0), rand(1.0:0.01:2.0), rand(1.0:0.01:2.0)]
        newprob = remake(prob, p=new_p)
        sol = solve_pde_problem(newprob, Rodas4P2())
        data_n = [sol(cord[1,j], cord[2,j])[1] for j=1:length(tmp)]
        for p in 1:length(tmp)
            data[n,p] = data_n[p]
        end
        for q in 1:num_params
            X[q,n] = new_p[q]
        end
    end
    #return the values: input functional signals, grid points for PDE, and reference data for PDE
    return X, cord, data

end

#function for generating testing data using numerical method, using random grid sampling
function generate_test_data_randomgrid(;num=100, num_cord=2048, t_domain = (0.0, 3600), z_domain=(0.0, 10.0))
    #generate an N by 2 matrix containing the sample points
    cord = QuasiMonteCarlo.sample(num_cord, [t_domain[1], z_domain[1]], [t_domain[2], z_domain[2]], GridSample([4,0.5]))
    grid_ic = collect(LinRange(z_domain[1], z_domain[2], 21))
    #set up the pde problem (discritise once) for a random condition
    prob, zs = setup_pde_problem(t_domain, z_domain, pde=nonlinear_heat_conduction, T0=x -> x)

    #generate the matrix containing training data points, and evaluated input functionals
    data = zeros((num,  num_cord))
    X = zeros((length(grid_ic), num))
    for n in 1:num
        input_func(x) = 0.0073*n*x
        @info "Loop $n"
        newprob = remake(prob, u0=input_func.(zs))
        sol = solve_pde_problem(newprob, Rodas4P2())
        data_n = [sol(cord[1,j], cord[2,j])[1] for j=1:num_cord]
        for p in 1:num_cord
            data[n,p] = data_n[p]
        end
        func_val_n = [input_func(a) for a in grid_ic]
        for q in 1:length(grid_ic)
            X[q,n] = func_val_n[q]
        end
    end
    #return the values: input functional signals, grid points for PDE, and reference data for PDE
    return X, cord, data

end

#generate the testing data for 100 cases
X_test, y_test, data_test = generate_test_data_fixedgrid_with_params(num=100)
X_test, y_test, data_test = generate_test_data_randomgrid(num=100,  num_cord=8000)
u0_test = X_test'
xt_test = y_test'
npzwrite("heat_test_nonlinear_1h_params.npz", u0=u0_test, xt=xt_test, u=data_test)



# train
random_coords = QuasiMonteCarlo.sample(1024, [0.0,.0], [24*3600.0,10.0], SobolSample())
# test
t_domain = (0.0, 3600)
z_domain=(0.0, 10.0)
cord = QuasiMonteCarlo.sample(8000, [t_domain[1], z_domain[1]], [t_domain[2], z_domain[2]], GridSample([4,0.5]))
grid_coords = QuasiMonteCarlo.sample(2048, [0.0,.0], [24*3600.0,10.0], GridSample([60.0,0.1]))
scatter(cord[1,:], cord[2,:])

# function for training a DeepONet
function train_DeepONet(simulate_func=simulate_heat_conduction_simple, num=50, grid_size=0.1, t_domain = (0.0, 1.0), z_domain=(0.0, 1.0))
    X, cord, data = generate_data()
    model = NeuralOperators.DeepONet((11, 1024, 1024), (2, 1024, 1024))
    loss(m, X, reference, coordinates) = Flux.Losses.mse(m(X, coordinates), reference)
    opt_state = Flux.setup(Flux.Adam(0.01), model)
    nn_data = [(X, data, cord)]
    Flux.@epochs 1000 Flux.train!(loss, model, nn_data, opt_state)
    return loss(model, X, data, cord), data, model(X, cord)
end

loss, data, m = train_DeepONet();

model = NeuralOperators.DeepONet((11, 1024, 1024), (2, 1024, 1024))
test1 = model(X, A)
loss(m, X, reference, coordinates) = Flux.Losses.mse(m(X, coordinates), reference)

opt_state = Flux.setup(Flux.Adam(0.01), model)
nn_data = [(X, data, cord)]
Flux.@epochs 1000 Flux.train!(loss, model, nn_data, opt_state)

loss(model, X, data, cord)

ref = simulate_heat_conduction_simple((0.0,24*3600.0), (0.0,10.0), T0=x -> sin(2π*x), α=0.4)
U = first(values(ref.sol.u))
X = U[1:end-1,:]'
Y = U[2:end,:]'
model2 = Flux.Chain(Flux.Dense(101 => 1024, tanh), Flux.Dense(1024 => 101))
loss2(m, x, y) = Flux.Losses.mse(m(x), y)
opt_state = Flux.setup(Flux.Adam(0.01), model2)
nn_data2 = [(X, Y)]
Flux.@epochs 1000 Flux.train!(loss2, model2, nn_data2, opt_state)
loss2(model2, X, Y)

maximum(abs.(model2(X) .- Y))
