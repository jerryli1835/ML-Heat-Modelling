import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import deepxde as dde
from deepxde.backend import tf
from scipy.interpolate import LinearNDInterpolator
import xarray as xr
import pandas as pd


# load data from files
dataset = xr.load_dataset("data/cglite_freeW_ERA_MKL_2010-2020/cglite_freeW_ERA_MKL_2010-2020.nc")
input_data = pd.read_csv("data/cglite_freeW_ERA_MKL_2010-2020/input_data.csv")
parameters = pd.read_csv("data/cglite_freeW_ERA_MKL_2010-2020/parameters.csv")


# generate training data with input being parameters and boundary conditions
def generate_train_data(num_run = 100, num_time=9, num_space=150, num_params = 13, num_input=365, time_span=365, space_span=11.9):
    num_coords = time_span*num_space
    num = num_run*num_time
    data_T = dataset["T"].values
    num_u0 = num_params + num_input
    xt = np.zeros((num_coords, 2))
    u = np.zeros((num, num_coords))
    u0 = np.zeros((num, num_u0))
    time_coords = np.array([n for n in range(365)])
    space_coords = dataset["T"].coords["z"].values[0:num_space]
    input_data_arr = np.array(input_data["Tair"])
    tmp_list = []
    for i in time_coords:
        for j in space_coords:
            tmp_list.append((i,j))
    for q in range(num_coords):
        xt[q, 0] = tmp_list[q][0]/time_span
        xt[q, 1] = tmp_list[q][1]/space_span
    for n in range(num_run):
        arr_params = np.array(parameters.iloc[n])
        for y in range(num_time):
            data_T_slice = data_T[n,(122+y*366):(487+y*366),0:num_space].reshape(num_coords, 1)
            arr_input = input_data_arr[(122+y*366):(487+y*366):int(time_span/num_input)]
            for p in range(num_coords):
                u[n*num_time + y, p] = data_T_slice[p]
            for a in range(num_params):
                u0[n*num_time + y, a] = arr_params[a]
            for b in range(num_input):
                u0[n*num_time + y, num_params+b] = arr_input[b]
    u0 = np.float32(u0)
    xt = np.float32(xt)
    u = np.float32(u)
    return (u0, xt), u


# generate testing data with input being parameters and boundary conditions
def generate_test_data(num_run = 100, num_time=1, num_space=150, num_params = 13, num_input=365, time_span=365, space_span=11.9):
    num_coords = time_span*num_space
    num = num_run*num_time
    data_T = dataset["T"].values
    num_u0 = num_params + num_input
    xt = np.zeros((num_coords, 2))
    u = np.zeros((num, num_coords))
    u0 = np.zeros((num, num_u0))
    time_coords = np.array([n for n in range(365)])
    space_coords = dataset["T"].coords["z"].values[0:num_space]
    input_data_arr = np.array(input_data["Tair"])
    tmp_list = []
    for i in time_coords:
        for j in space_coords:
            tmp_list.append((i,j))
    for q in range(num_coords):
        xt[q, 0] = tmp_list[q][0]/time_span
        xt[q, 1] = tmp_list[q][1]/space_span
    for n in range(num_run):
        arr_params = np.array(parameters.iloc[n])
        for y in range(num_time):
            data_T_slice = data_T[n,3408:3773,0:num_space].reshape(num_coords, 1)
            arr_input = input_data_arr[3408:3773:int(time_span/num_input)]
            for p in range(num_coords):
                u[n*num_time + y, p] = data_T_slice[p]
            for a in range(num_params):
                u0[n*num_time + y, a] = arr_params[a]
            for b in range(num_input):
                u0[n*num_time + y, num_params+b] = arr_input[b]
    u0 = np.float32(u0)
    xt = np.float32(xt)
    u = np.float32(u)
    return (u0, xt), u


# generate training data with input being only parameters
def generate_train_data_without_boundary(num_run = 100, num_time=9, num_space=150, num_params = 13, time_span=365, space_span=11.9):
    num_coords = time_span*num_space
    num = num_run*num_time
    num_u0 = num_params
    data_T = dataset["T"].values
    xt = np.zeros((num_coords, 2))
    u = np.zeros((num, num_coords))
    u0 = np.zeros((num, num_u0))
    time_coords = np.array([n for n in range(365)])
    space_coords = dataset["T"].coords["z"].values[0:num_space]
    tmp_list = []
    for i in time_coords:
        for j in space_coords:
            tmp_list.append((i,j))
    for q in range(num_coords):
        xt[q, 0] = tmp_list[q][0]/time_span
        xt[q, 1] = tmp_list[q][1]/space_span
    for n in range(num_run):
        arr_params = np.array(parameters.iloc[n])
        for y in range(num_time):
            data_T_slice = data_T[n,(122+y*366):(487+y*366),0:num_space].reshape(num_coords, 1)
            for p in range(num_coords):
                u[n*num_time + y, p] = data_T_slice[p]
            for a in range(num_params):
                u0[n*num_time + y, a] = arr_params[a]
    u0 = np.float32(u0)
    xt = np.float32(xt)
    u = np.float32(u)
    return (u0, xt), u


# generate testing data with input being only parameters
def generate_test_data_without_boundary(num_run = 100, num_time=1, num_space=150, num_params = 13, time_span=365, space_span=11.9):
    num_coords = time_span*num_space
    num = num_run*num_time
    num_u0 = num_params
    data_T = dataset["T"].values
    xt = np.zeros((num_coords, 2))
    u = np.zeros((num, num_coords))
    u0 = np.zeros((num, num_u0))
    time_coords = np.array([n for n in range(365)])
    space_coords = dataset["T"].coords["z"].values[0:num_space]
    tmp_list = []
    for i in time_coords:
        for j in space_coords:
            tmp_list.append((i,j))
    for q in range(num_coords):
        xt[q, 0] = tmp_list[q][0]/time_span
        xt[q, 1] = tmp_list[q][1]/space_span
    for n in range(num_run):
        arr_params = np.array(parameters.iloc[n])
        for y in range(num_time):
            data_T_slice = data_T[n,3408:3773,0:num_space].reshape(num_coords, 1)
            for p in range(num_coords):
                u[n*num_time + y, p] = data_T_slice[p]
            for a in range(num_params):
                u0[n*num_time + y, a] = arr_params[a]
    u0 = np.float32(u0)
    xt = np.float32(xt)
    u = np.float32(u)
    return (u0, xt), u


# function for training the DeepONet and testing the model
def train_DeepONet(nx=13, num_epochs=50000, func_train_data = generate_train_data, func_test_data = generate_test_data):
    x_train, y_train = func_train_data()
    x_test, y_test = func_test_data()
    data = dde.data.TripleCartesianProd(x_train, y_train, x_test, y_test)

    net = dde.maps.DeepONetCartesianProd(
        [nx, 512, 512, 1024], [2, 512, 512, 512, 1024], "relu", "Glorot normal"
    )

    model = dde.Model(data, net)
    model.compile(
        "adam",
        lr=1e-3,
        decay=("inverse time", 1, 1e-4),
        metrics=["mean l2 relative error"],
    )
    # callbacks = [dde.callbacks.EarlyStopping(monitor='loss_test', patience=10000)]
    # dde.callbacks.ModelCheckpoint(filepath='best_model', save_better_only=True, period=1000, monitor='test loss')
    losshistory, train_state = model.train(epochs=num_epochs, batch_size=None)
    
    return model, data 


# function for saving and plotting the data as heatmap
def plot_data_heatmap(model=None, data=None, key=10, num_coords=54750, num_space = 150, mode = "reference"):
    data_x = data1.test_x
    data_y = data1.test_y
    y_pred = model1.predict(data_x)
    
    xt = np.zeros((num_coords, 2))
    time_coords = np.array([n for n in range(365)])
    space_coords = dataset["T"].coords["z"].values[0:num_space]
    tmp_list = []
    for i in time_coords:
        for j in space_coords:
            tmp_list.append((i,j))
    for q in range(num_coords):
        xt[q, 0] = tmp_list[q][0]
        xt[q, 1] = tmp_list[q][1]
    
    y_predict_arr = y_pred[key].reshape(num_coords, 1)
    y_true_arr = data_y[key].reshape(num_coords, 1)
    np.savetxt("permafrost_4.{}.dat".format(key), np.hstack((xt, y_true_arr, y_predict_arr)))
    
    data_plot = np.genfromtxt("permafrost_4.{}.dat".format(key), delimiter=' ')

    # constructing the grid
    x = data_plot[:,1]
    t = data_plot[:,0]
    X = np.linspace(min(x), max(x))
    T = np.linspace(min(t), max(t))
    X, T = np.meshgrid(X, T)
    
    # plotting the refernce data
    if mode == "reference":
        interp1 = LinearNDInterpolator(list(zip(data_plot[:,1], data_plot[:,0])), data_plot[:,2])
        Z = interp1(X, T)
        graph1 = plt.pcolormesh(T, X, Z)
        plt.gca().invert_yaxis()
        plt.ylabel('space(m)')
        plt.xlabel('time(day)')
        colorbar = plt.colorbar(graph1)
        colorbar.set_label('Temperature(℃)')
        plt.title("reference simulation")
        plt.savefig("permafrost_reference_4.{}.png".format(key))
    # getting the predicted data
    if mode == "predict":
        interp2 = LinearNDInterpolator(list(zip(data_plot[:,1], data_plot[:,0])), data_plot[:,3])
        Z = interp2(X, T)
        graph2 = plt.pcolormesh(T, X, Z)
        plt.gca().invert_yaxis()
        plt.ylabel('space(m)')
        plt.xlabel('time(day)')
        colorbar = plt.colorbar(graph2)
        colorbar.set_label('Temperature(℃)')
        plt.title("predicted outcome")
        plt.savefig("permafrost_predict_4.{}.png".format(key))


# function for saving and plotting the data for fixed depth, temperature against time
def plot_data_fixed_depth(key=10, experiment=3, depth=0.075, num_coords=54750, num_space = 150):
    # recover indexing for time
    a = int((key+1) % 10)
    b = int((key + 1 - a)/10)
    index_a = a - 1
    index_b = b - 1
    # restructure the space data
    space_coords = dataset["T"].coords["z"].values[0:num_space]
    space_list = list(np.round(space_coords, decimals=3))
    space_str = [str(a) for a in space_list]
    iter_list = [n for n in range(num_space)]
    dict_list = list(zip(space_str, iter_list))
    dict_space = dict(dict_list)
    key_space = dict_space['{}'.format(depth)]
    #load the data
    data_plot = np.genfromtxt("permafrost_{}.{}.dat".format(experiment, key), delimiter=' ')
    # constructing the grid
    x = data_plot[:,1]
    t = data_plot[:,0]
    # recover the real time axis
    time = dataset["T"].coords["time"].values[(122+index_a*366):(487+index_a*366)]
    # collect the temperature data
    tem_ref = data_plot[:,2]
    tem_pre = data_plot[:,3]
    list_tem_ref = []
    list_tem_pre = []
    for n in range(365):
        list_tem_ref.append(tem_ref[n*num_space+key_space])
        list_tem_pre.append(tem_pre[n*num_space+key_space])
    # making the plot
    plt.figure(figsize=(8,6))
    plt.plot(time, list_tem_ref, label='reference', linestyle='-', color='blue')
    plt.plot(time, list_tem_pre, label='predicted', linestyle='-', color='red')
    plt.xlabel('Time')
    plt.ylabel('Temperature(℃)')
    plt.legend()
    plt.savefig("temperature{}_depth{}_{}.png".format(experiment, depth, key))


# experiment 1, predicting temperature profile of 12 meters and 1 year with input being parameters and boundary conditions
# implement the training and testing
model1, data1 = train_DeepONet(nx=378, num_epochs=30000)
# plot the results
plot_data_heatmap(model=model1, data=data1, key=50, mode = "reference")
plot_data_heatmap(model=model1, data=data1, key=50, mode = "predict")


# experiment 2, predicting temperature profile of 12 meters and 1 year with input being only parameters
# implement the training and testing
model2, data2 = train_DeepONet(nx=13, num_epochs=50000, func_train_data = generate_train_data_without_boundary, func_test_data = generate_test_data_without_boundary)
# plot the results
plot_data(model=model2, data=data2, key=90, mode = "reference")
plot_data(model=model2, data=data2, key=90, mode = "predict")


# plotting the temperature evolution over time
plot_data_fixed_depth(key=40, experiment=1, depth=0.075, num_space = 100)
plot_data_fixed_depth(key=40, experiment=1, depth=0.375, num_space = 100)
plot_data_fixed_depth(key=40, experiment=1, depth=1.475, num_space = 100)
plot_data_fixed_depth(key=40, experiment=1, depth=4.95, num_space = 100)
plot_data_fixed_depth(key=20, experiment=1, depth=7.35, num_space = 150)
plot_data_fixed_depth(key=20, experiment=1, depth=9.55, num_space = 150)
