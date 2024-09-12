import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import deepxde as dde
from deepxde.backend import tf
from scipy.interpolate import LinearNDInterpolator


# function for orgnising the data of refernce simulations to be input into DeepONet 
def get_data(filename, time_span = 24*3600):
    data = np.load(filename)
    u0 = data["u0"].astype(np.float32)
    xt = data["xt"].astype(np.float32)
    num_xt = xt.shape[0]
    for n in range(num_xt):
        xt[n,0] = xt[n,0]/time_span
    u = data["u"].astype(np.float32) 
    return (u0, xt), u


# function for training the DeepONet and testing the model
def train_DeepONet(nx=4, train_data = "heat_train_nonlinear_1h_params.npz", test_data = "heat_test_nonlinear_1h_params.npz", num_epochs=40000, time_span = 24*3600):
    x_train, y_train = get_data(train_data, time_span)
    x_test, y_test = get_data(test_data, time_span)
    data = dde.data.TripleCartesianProd(x_train, y_train, x_test, y_test)

    net = dde.maps.DeepONetCartesianProd(
        [nx, 512, 512, 512], [2, 512, 512, 512, 512], "relu", "Glorot normal"
    )

    model = dde.Model(data, net)
    model.compile(
        "adam",
        lr=1e-3,
        decay=("inverse time", 1, 1e-4),
        metrics=["mean l2 relative error"],
    )
    callbacks = [dde.callbacks.EarlyStopping(monitor='loss_test', patience=8000)]
    # dde.callbacks.ModelCheckpoint(filepath='best_model', save_better_only=True, period=1000, monitor='test loss')
    losshistory, train_state = model.train(epochs=num_epochs, batch_size=None, callbacks= callbacks)
    return model, data


# function for maiking prediction and save the data for plotting
def save_data_DeepONet(test_data = "heat_test_nonlinear_1h_params.npz", file_prefix = "test_heat_nonlinear_1h_params", key=10, num_cord = 18921, model=None, data = None):
    data_x = data.test_x
    data_y = data.test_y
    y_pred = model.predict(data_x)
    data_unscaled = np.load(test_data)
    xt_arr = data_unscaled["xt"].astype(np.float32)
    y_predict_arr = y_pred[key].reshape(num_cord, 1)
    y_true_arr = data_y[key].reshape(num_cord, 1)
    np.savetxt((file_prefix + "_{}".format(key) + ".dat"), np.hstack((xt_arr, y_true_arr, y_predict_arr)))
    

# function for plotting the data
def plot_data(mode = "reference", file_prefix = "heat_nonlinear_1h_params", key=10):
    data = np.genfromtxt((file_prefix + "_{}".format(key) + ".dat"), delimiter=' ')
    # constructing the grid
    x = data[:,0]
    t = data[:,1]
    X = np.linspace(min(x), max(x))
    T = np.linspace(min(t), max(t))
    X, T = np.meshgrid(X, T)
    # for getting the refernce data
    if mode == "reference":
        interp1 = LinearNDInterpolator(list(zip(data[:,0], data[:,1])), data[:,2]) 
        Z = interp1(X, T)
        graph1 = plt.pcolormesh(X, T, Z)
        plt.ylabel('space(m)')
        plt.xlabel('time(s)')
        colorbar = plt.colorbar(graph1)
        colorbar.set_label('Temperature(℃)')
        plt.title("Reference simulation")
        plt.savefig(file_prefix + "_reference{}".format(key) + ".png")
    elif mode == "predict":
        interp2 = LinearNDInterpolator(list(zip(data[:,0], data[:,1])), data[:,2]) 
        Z = interp2(X, T)
        graph2 = plt.pcolormesh(X, T, Z)
        plt.ylabel('space(m)')
        plt.xlabel('time(s)')
        colorbar = plt.colorbar(graph2)
        colorbar.set_label('Temperature(℃)')
        plt.title("Predicted outcome")
        plt.savefig(file_prefix + "_predicted{}".format(key) + ".png")
    else:
        return NotImplementedError("mode can only be set as reference or predict")


# experiment 1: non-linear heat transfer of 1 hour, input of branch network as linear initial condition samples
# implement the training and testing rpocedure
model1, data1 = train_DeepONet(nx=21, train_data = "heat_train_nonlinear_1h.npz", test_data = "heat_test_nonlinear_1h.npz", num_epochs=50000, time_span = 3600)
# save and plot the data from the model
# case 1
save_data_DeepONet(test_data = "heat_test_nonlinear_1h.npz", file_prefix = "test_heat_nonlinear_1h", key=10, num_cord = 18921, model=model1, data = data1)
plot_data(mode = "reference", file_prefix = "heat_nonlinear_1h", key=10)
plot_data(mode = "predict", file_prefix = "heat_nonlinear_1h", key=10)
# case 2
save_data_DeepONet(test_data = "heat_test_nonlinear_1h.npz", file_prefix = "test_heat_nonlinear_1h", key=50, num_cord = 18921, model=model1, data = data1)
plot_data(mode = "reference", file_prefix = "heat_nonlinear_1h", key=50)
plot_data(mode = "predict", file_prefix = "heat_nonlinear_1h", key=50)
# case 3
save_data_DeepONet(test_data = "heat_test_nonlinear_1h.npz", file_prefix = "test_heat_nonlinear_1h", key=90, num_cord = 18921, model=model1, data = data1)
plot_data(mode = "reference", file_prefix = "heat_nonlinear_1h", key=90)
plot_data(mode = "predict", file_prefix = "heat_nonlinear_1h", key=90)


# experiment 2: non-linear heat transfer of 24 hours, input of branch network as different parameters in the PDE set up
# implement the training and testing rpocedure
model2, data2 = train_DeepONet(nx=4, train_data = "heat_train_nonlinear_1h_params.npz", test_data = "heat_test_nonlinear_1h_params.npz", num_epochs=35000, time_span = 24*3600)
# save and plot the data from the model
# case 1
save_data_DeepONet(test_data = "heat_test_nonlinear_1h_params.npz", file_prefix = "test_heat_nonlinear_1h_params", key=10, num_cord = 18921, model=model2, data = data2)
plot_data(mode = "reference", file_prefix = "heat_nonlinear_1h_params", key=10)
plot_data(mode = "predict", file_prefix = "heat_nonlinear_1h_params", key=10)
# case 2
save_data_DeepONet(test_data = "heat_test_nonlinear_1h_params.npz", file_prefix = "test_heat_nonlinear_1h_params", key=50, num_cord = 18921, model=model2, data = data2)
plot_data(mode = "reference", file_prefix = "heat_nonlinear_1h_params", key=50)
plot_data(mode = "predict", file_prefix = "heat_nonlinear_1h_params", key=50)
# case 3
save_data_DeepONet(test_data = "heat_test_nonlinear_1h_params.npz", file_prefix = "test_heat_nonlinear_1h_params", key=90, num_cord = 18921, model=model2, data = data2)
plot_data(mode = "reference", file_prefix = "heat_nonlinear_1h_params", key=90)
plot_data(mode = "predict", file_prefix = "heat_nonlinear_1h_params", key=90)
