import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator

data = np.genfromtxt("test_sin_large.dat", delimiter=' ')
interp = LinearNDInterpolator(list(zip(data[:,0], data[:,1])), data[:,3])
x = data[:,0]
t = data[:,1]
X = np.linspace(min(x), max(x))
T = np.linspace(min(t), max(t))
X, T = np.meshgrid(X, T)
Z = interp(X, T)
plt.pcolormesh(X, T, Z)
plt.show()
plt.savefig('heat_analytic_sin1.2.png')
