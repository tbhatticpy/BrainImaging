#%% Import modules


import numpy as np
import matplotlib.pyplot as plt


import sys
utilities_path = 'C:/Users/Talha/PycharmProjects/BrainImaging/Exercise6/'      # replace path
sys.path.append(utilities_path)   

from cross_functions import cplv
from kuramoto import KuramotoFast
from typing import List
import tqdm

try:                                         # import cupy if you have it installed
    import cupy as cp
    use_cuda = True
except:
    use_cuda = False

plt.style.use('default')



#%% Task 1 - Single node Kuramoto oscillator (10 points)


""" Today, we're gonna simulate some data with the hierarchical (2-layer) Kuramoto model.

This model simulates N nodes, each consisting of M locally coupled oscillators.

The local coupling strength K controls how strongly the oscillators within a node 
    are coupled to each other. K can take a different value for each node and is set in a parameter k_list.
    
The long-range coupling between nodes is controlled by the parameter weight_matrix 
     (an array with shape N x N).

How long it takes to simulate data will depend on your computer's computational power.
If you have a NVidea graphics card, you can try to install CUDA toolkit and CUPY
    which may speed up computations considerably. This is completely voluntary though.
    If your system is slow, use the minimum amount of nodes and oscillators recommended in the exercises,
    if you have a fast system, feel free to use more.


For Task 1, we will start with a single node.
Initialize KuramotoFast with a single node consisting of at least 100 oscillators.
Set the sampling rate to 200 Hz.
Set the mean frequency for the node to 10 Hz.
With frequency_spread, you can control how much the frequencies of the 
   individual oscillators in the node vary around the mean. Set it to 1.
   
Also, we'll set the noise_scale to 1, the local coupling k to 1.

With one node, the weight_matrix will be a 1x1 matrix with the value 1.

Now simulate 1.5 sec of activity with model.simulate().

Plot the real part, imaginary part, and amplitude of the node time series in one plot.
Add a legend and label the x-axis correctly.

"""

# Parameters for the simulation
n_nodes = 1
n_oscillators = 100
sampling_rate = 200  # Hz
node_frequencies = [10]  # Mean frequency for the node in Hz
frequency_spread = 1  # Frequency spread
noise_scale = 1  # Noise scale
k_list = [1]  # Local coupling strength
weight_matrix = np.array([[1]])  # 1x1 weight matrix for a single node
time = 1.5  # Duration of the simulation in seconds

# Initialize the KuramotoFast model
model = KuramotoFast(n_nodes = n_nodes, n_oscillators = n_oscillators, k_list = k_list,
                     weight_matrix = weight_matrix, node_frequencies = node_frequencies,
                     sampling_rate = sampling_rate, frequency_spread = frequency_spread,
                     noise_scale = noise_scale,use_cuda=use_cuda)

data_simulated = model.simulate(time=time)

# Extract real, imaginary parts, and amplitude
real_part = np.real(data_simulated[0])
imag_part = np.imag(data_simulated[0])
amplitude = np.abs(data_simulated[0])

# Time vector for plotting
time_vector = np.linspace(0, time, len(real_part))

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(time_vector, real_part, label='Real Part')
plt.plot(time_vector, imag_part, label='Imaginary Part')
plt.plot(time_vector, amplitude, label='Amplitude')

plt.xlabel('Time (s)')
plt.ylabel('Value')
plt.title('Kuramoto Model - Single Node Time Series')
plt.legend()
plt.grid(True)

plt.show()





#%% Task 2 - Explore the effect of noise and frequency spread. ( 5 points)

"""
Set both the noise and frequency_spread parameters to 0 and simulate again 1.5 sec. 
How does the simulated data look like now, and why?
What happens when you use larger values for these parameters?
"""


# Adjust parameters: set noise_scale and frequency_spread to 0
frequency_spread_zero = 0
noise_scale_zero = 0

# Reinitialize the model with new parameters
model_zero = KuramotoFast(
    n_nodes=n_nodes,
    n_oscillators=n_oscillators,
    sampling_rate=sampling_rate,
    k_list=k_list,
    weight_matrix=weight_matrix,
    node_frequencies=node_frequencies,
    frequency_spread=frequency_spread_zero,
    noise_scale=noise_scale_zero,
    use_cuda=False
)

# Simulate the data
data_simulated_zero = model_zero.simulate(time=time)

# Extract real, imaginary parts, and amplitude
real_part_zero = np.real(data_simulated_zero[0])
imag_part_zero = np.imag(data_simulated_zero[0])
amplitude_zero = np.abs(data_simulated_zero[0])

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(time_vector, real_part_zero, label='Real Part (Noise=0, Spread=0)')
plt.plot(time_vector, imag_part_zero, label='Imaginary Part (Noise=0, Spread=0)')
plt.plot(time_vector, amplitude_zero, label='Amplitude (Noise=0, Spread=0)')

plt.xlabel('Time (s)')
plt.ylabel('Value')
plt.title('Kuramoto Model - Single Node Time Series (No Noise, No Frequency Spread)')
plt.legend()
plt.grid(True)

plt.show()




#%% Task 3 (4 bonus points)

"""
Let's think more about what we observed in Task 2.

Which other quantity could we plot to visualize the node's oscillatory behaviour? 

Plot this quantity over time for noise = 0 and spread = 0, 
   and when either noise or spread is 4 and the other is 0.
   
"""


# Define a function to calculate the phase of the oscillation
def calculate_phase(data):
    return np.angle(data)


# Scenarios to simulate
scenarios = {
    "Noise=0, Spread=0": (0, 0),
    "Noise=4, Spread=0": (4, 0),
    "Noise=0, Spread=4": (0, 4)
}

# Plotting
plt.figure(figsize=(12, 8))

for i, (label, (noise, spread)) in enumerate(scenarios.items(), 1):
    # Reinitialize the model with current parameters
    model = KuramotoFast(
        n_nodes=n_nodes,
        n_oscillators=n_oscillators,
        sampling_rate=sampling_rate,
        k_list=k_list,
        weight_matrix=weight_matrix,
        node_frequencies=node_frequencies,
        frequency_spread=spread,
        noise_scale=noise,
        use_cuda=False
    )

    # Simulate the data
    data_simulated = model.simulate(time=time)

    # Calculate phase
    phase = calculate_phase(data_simulated[0])

    # Plotting the phase
    plt.subplot(3, 1, i)
    plt.plot(time_vector, phase, label=label)
    plt.xlabel('Time (s)')
    plt.ylabel('Phase (radians)')
    plt.title(f'Phase over Time ({label})')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

#%% Task 4. Assess phase-locking in a multi-node Kuramoto model (15 points).


"""

Now let's simulate multiple nodes. Use at least 10 nodes. 
If you have a good CPU, or a GPU and cupy, feel free to use more nodes.

Choose the node frequencies so that they are Gaussian-distributed around 10 Hz.

Choose values for the weight matrix so that 
   - the first N/2 nodes are coupled strongly to each other,
   - the next N/2 nodes are also coupled strongly to each other
   - the other connections should be 0 or very small.
  
Simulate at least 20 seconds data with the model.

Compute and plot the phase locking value (PLV) between all nodes' time series. 
   Label all 3 axes.

Is the PLV among the nodes that were set to be coupled in the weight matrix notably larger than among non-connected ones?
   (should be at least 2 times larger on average)
   
If not, try modifying the parameters until you get the desired result for PLV.

You can modify: K, L , frequency_spread (within nodes), 
    the standard variation of node's mean frequencies,
    noise_scale, time, n_oscillators per node

"""

# Parameters
n_nodes = 10  # At least 10 nodes
n_oscillators = 100  # Oscillators per node
sampling_rate = 200  # Hz
time = 20  # Simulation time in seconds
mean_frequency = 10  # Hz
frequency_std = 0.5  # Standard deviation for Gaussian-distributed frequencies
frequency_spread = 1  # Frequency spread within nodes
noise_scale = 0.5  # Noise level
k_list = [1] * n_nodes  # Local coupling strength

# Generate Gaussian-distributed node frequencies around 10 Hz
node_frequencies = np.random.normal(loc=mean_frequency, scale=frequency_std, size=n_nodes)

# Create the weight matrix
weight_matrix = np.zeros((n_nodes, n_nodes))
strong_coupling_value = 0.8  # Strong coupling value

# Strong coupling within the first N/2 nodes and the next N/2 nodes
for i in range(n_nodes // 2):
    for j in range(n_nodes // 2):
        if i != j:
            weight_matrix[i, j] = strong_coupling_value

for i in range(n_nodes // 2, n_nodes):
    for j in range(n_nodes // 2, n_nodes):
        if i != j:
            weight_matrix[i, j] = strong_coupling_value

# Initialize the model
model = KuramotoFast(
    n_nodes=n_nodes,
    n_oscillators=n_oscillators,
    sampling_rate=sampling_rate,
    k_list=k_list,
    weight_matrix=weight_matrix,
    node_frequencies=node_frequencies,
    frequency_spread=frequency_spread,
    noise_scale=noise_scale,
    use_cuda=False
)

# Simulate the data
data_simulated = model.simulate(time=time)

# Compute PLV between all nodes
plv_matrix = cplv(data_simulated)

# Plotting the PLV matrix
plt.figure(figsize=(10, 8))
plt.imshow(np.abs(plv_matrix), cmap='hot', interpolation='nearest')
plt.colorbar(label='PLV')
plt.title('Phase Locking Value (PLV) Between Nodes')
plt.xlabel('Node Index')
plt.ylabel('Node Index')
plt.grid(False)
plt.show()












#%% Task 5 (4 bonus points)

"""
Describe which changes to which parameters in Task 4 move the results closer
     in the desired direction and why.
"""





#%% Task 6: Simulate single spiking neuron (14 points)


""" 
Now, we'll do something different and simulate a single spiking neuron, 
    rather than create idealized oscillations. 
    
Our implementation is based on the model from Izhikevich (2003, www.izhikevich.com).

We want to evaluate the model's evolution over 1 sec time (at steps of 1 msec).   

This model works with two state variables: U and V, and 4 parameters: a,b,c,d.

U is the membrane potential, and V is a feedback variable that represents 
   the activation/inactivation of K+ and Na+ ionic currents. 
      
There is also an external current I which we will set 7 mA for the first 400 msec,
  then to 0 for 200 msec, and then again to 7 mA.

The evolution of U and V from time point t to t+1 is determined as follows:
    
V (t+1) = 140 + 0.04 * V^2 + 6 * V + I - U 
U (t+1) = U + a * (b * V - U)

The initial value for V is -65 mV, and (b * V) for U.

In order to simulate the "reset" of a neuron after initiating an action potential,  
  if V goes above a threshold V_thr = 30 mV, then 
  V will be reset to c, and U will be reset to (U + d).

Implement this and evaluate the evolution of V over 1 sec.

Plot both V and I as a function of time.

Ideally, plot them in two vertically stacked panels of one figure using plt.subplots(),
   or both in the same panel, but with separate y-axes 
   (for this, you can use ax2 = ax.twinx(), set the limits so that the time series don't overlap).

Describe the behaviour this neuron exhibits.
 
What about this model is different from real neurons?                                                     

"""


# Parameters
a = 0.02
b = 0.2
c = -65
d = 8
v_thr = 30

# Initial values
v = -65  # Initial membrane potential (mV)
u = b * v  # Initial recovery variable

# Time settings
T = 1000  # Total time in ms
dt = 1    # Time step in ms
time = np.arange(0, T + dt, dt)

# External current I
I = np.zeros_like(time)
I[:400] = 7  # 7 mA for the first 400 ms
I[600:] = 7  # 7 mA from 600 ms onwards

# Arrays to store V and U over time
V = np.zeros_like(time, dtype=float)
U = np.zeros_like(time, dtype=float)

V[0] = v
U[0] = u

# Simulation loop
for t in range(1, len(time)):
    v = V[t-1]
    u = U[t-1]
    current = I[t-1]

    # Update rules
    v_new = v + dt * (0.04 * v**2 + 5 * v + 140 - u + current)
    u_new = u + dt * (a * (b * v - u))

    # Spike condition and reset
    if v_new >= v_thr:
        v_new = c
        u_new = u + d

    V[t] = v_new
    U[t] = u_new

# Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Plot membrane potential V
ax1.plot(time, V, label='Membrane Potential (V)', color='b')
ax1.set_ylabel('Membrane Potential (mV)')
ax1.set_title('Neuron Model Simulation (Izhikevich, 2003)')
ax1.grid()
ax1.legend()

# Plot external current I
ax2.plot(time, I, label='External Current (I)', color='r')
ax2.set_xlabel('Time (ms)')
ax2.set_ylabel('Current (mA)')
ax2.grid()
ax2.legend()

plt.tight_layout()
plt.show()




#%% Task 7: Simulate other neuron types (6 points)

"""
Now, go to https://www.izhikevich.org/publications/spikes.htm.

Change the parameters of your model so that you produce behaviour typical of
- fast-spiking neurons
- chattering neurons

"""


# Define parameters for fast-spiking and chattering neurons
neuron_types = {
    "Fast-Spiking": {"a": 0.1, "b": 0.2, "c": -65, "d": 2},
    "Chattering": {"a": 0.02, "b": 0.2, "c": -50, "d": 2}
}

# Time settings
T = 1000  # Total time in ms
dt = 1    # Time step in ms
time = np.arange(0, T + dt, dt)

# External current I (same for both)
I = np.zeros_like(time)
I[:400] = 7  # 7 mA for the first 400 ms
I[600:] = 7  # 7 mA from 600 ms onwards

# Plotting setup
fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

for idx, (neuron_type, params) in enumerate(neuron_types.items()):
    a, b, c, d = params["a"], params["b"], params["c"], params["d"]
    v_thr = 30

    # Initial values
    v = -65
    u = b * v

    # Arrays to store V and U over time
    V = np.zeros_like(time, dtype=float)
    U = np.zeros_like(time, dtype=float)

    V[0] = v
    U[0] = u

    # Simulation loop
    for t in range(1, len(time)):
        v = V[t-1]
        u = U[t-1]
        current = I[t-1]

        # Update rules
        v_new = v + dt * (0.04 * v**2 + 5 * v + 140 - u + current)
        u_new = u + dt * (a * (b * v - u))

        # Spike condition and reset
        if v_new >= v_thr:
            v_new = c
            u_new = u + d

        V[t] = v_new
        U[t] = u_new

    # Plotting membrane potential
    axs[idx].plot(time, V, label=f'{neuron_type} Neuron (V)', color='b')
    axs[idx].plot(time, I, label='External Current (I)', color='r', linestyle='--')
    axs[idx].set_ylabel('Membrane Potential (mV)')
    axs[idx].set_title(f'{neuron_type} Neuron Simulation')
    axs[idx].grid()
    axs[idx].legend()

axs[-1].set_xlabel('Time (ms)')
plt.tight_layout()
plt.show()








