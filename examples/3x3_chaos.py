#%%
from .ctrnn import CTRNN
import numpy as np
import matplotlib.pyplot as plt


#%%

W = [
    [5.422, -0.24, 0.535],
    [-0.018, 4.59, -2.25],
    [2.75, 1.21, 3.885]
]

N = len(W)
timesteps = 5000
T = 2*.3e-6
dt = T/timesteps
tau = 1e-8 # fc=1/tau=100MHz

x0 = np.random.rand(N)
width = np.array([15, 15, 15, 15])*np.pi * 1e-2
gain = 1.0 * np.ones_like(x0)

# x0 = np.array([-0.5, 0.5, 0.25, -0.25])
# width = np.array([1, 1, 1, 1])*np.pi
# gain = 25 * np.ones_like(x0)

network_params = {
    'number of neurons': N,
    'number of inputs': 0,
    'number of outputs': N,
    'integration mode': 'RK4',
    'time step': dt,
    'decay constant': tau,
    #'activation': ['Lorentzian', x0, width],
    #'activation': ['Sin', x0, width],
    'activation': 'sigmoid'
    'randomize weights': False,
    'weight matrix': W,
    'gains': gain,
}

net = CTRNN(**network_params)

time_vector = []
output_vector = []
state_vector = []

for i in range(timesteps):
    net.step()

    time_vector.append(net.t_seconds)
    state_vector.append(net.state_vector)
    output_vector.append(net.output_vector)


#%%

# plot the activation function of each neuron
plt.figure(figsize=(16,10))

dummy_x = np.linspace(0, 1, 1000)
functions = []

for x in dummy_x:
    functions.append(net.activation_function(x))

functions = np.array(functions)

# print(functions)


for i in range(N):
    plt.plot(dummy_x, functions[:,i], label=f'Neuron {i+1}\n  x0 = {x0[i]:.2e}\n  width = {width[i]:.2e}')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Activation Functions')
plt.legend(loc='right')
plt.show()



# Plot the state and output over time on subplots
plt.figure(figsize=(16,8))
plt.subplot(1, 2, 1)
plt.plot([t*1e6 for t in time_vector], state_vector)
plt.xlabel('Time (µs)')
plt.ylabel('Neuron State')

plt.subplot(1, 2, 2)
plt.plot([t*1e6 for t in time_vector], output_vector)
plt.xlabel('Time (µs)')
plt.ylabel('Neuron Output')

plt.title('Network State Over Time')
plt.show()

#%%

state_vector = np.array(state_vector)   
# make a high resolution 3d plot
plt.subplot(111, projection='3d')
cmap = plt.get_cmap('jet')
plt.scatter(state_vector[:,3], state_vector[:,1], state_vector[:,2], c=np.array([cmap(i/timesteps) for i in range(timesteps)]))
plt.show()



# %%
