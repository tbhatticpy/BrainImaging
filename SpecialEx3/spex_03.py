import numpy as np
import seaborn as sb
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd

# functional network indices
networks = np.array(
      [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  2,  2,
        2,  5,  5,  5,  5,  5,  5,  5,  5,  6,  6,  6,  6,  6,  6,  6,  6,
        6,  6,  6,  6,  6,  7,  7,  7,  3,  3,  3,  3,  3,  3,  4,  4,  4,
        4,  4,  8,  8,  9,  9,  9,  9, 13, 13, 13, 13, 13, 13, 13, 14, 14,
       14, 14, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11,
       11, 12, 12, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16,  0,  0,
        0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,
        5,  5,  5,  5,  5,  5,  6,  6,  6,  6,  7,  7,  7,  3,  3,  3,  3,
        3,  3,  4,  4,  4,  4,  4,  8,  8,  8,  8,  9,  9,  9,  9, 13, 13,
       13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 10, 10, 10, 10,
       10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12,
       12, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16])

# functional network names
netw_names = np.array(
    ['ContA', 'ContB', 'ContC',
     'DefaultA', 'DefaultB', 'DefaultC',
     'DorsAttnA', 'DorsAttnB',
     'LimbicA_TempPole', 'LimbicB_OFC',
     'SalVentAttnA', 'SalVentAttnB',
     'SomMotA', 'SomMotB',
     'TempPar',
     'VisCent', 'VisPeri'])

#%% Task 1: Compare graph strength for two populations  (7 points)

"""
Load the edge data from all 52 subjects. 
The first 25 subjects have been diagnosed with a condition, let's say it is MCI, 
   the other 27 are healthy controls.
For each frequency band:
  Compute the graph strength (mean edge strength) for each subject. 
  Plot the distribution of subjects' values in each group with a violinplot (with dots).
  Run an appropiate test to see if the distributions of values
    are significantly different between the two groups.
  
"""

bands = ['delta', 'theta', 'alpha', 'beta', 'low-gamma', 'high-gamma']
total_subjects = 52
group_labels = ['MCI']*25 + ['Healthy']*27
results = []

for band in bands:
    subject_strength = []

    for subject_idx in range(total_subjects):
        file_path = f"iPLV/{band}/S{subject_idx}.npy"
        matrix = np.load(file_path)
        strength = matrix[np.triu_indices_from(matrix, k=1)].mean() #self connections exluded
        subject_strength.append(strength)
    df_band = pd.DataFrame({
        'SubjectID': np.arange(total_subjects),
        'Group': group_labels,
        'Strength': subject_strength,
        'Band': band
    })
    results.append((df_band))

df_all = pd.concat(results, axis=0, ignore_index=True)

for band in bands:
    df_band = df_all[df_all['Band']==band]
    plt.figure(figsize=(6,6))
    sb.violinplot(data=df_band, x='Group', y='Strength', cut=0, inner=None)
    sb.stripplot(data=df_band, x='Group', y='Strength', color='black', size=4, alpha=0.7)
    plt.title(f'Graph Strength by Group for {band.capitalize()} Band')
    plt.ylabel('Mean Edge Strength')
    plt.xlabel('Group')
    plt.tight_layout()
    plt.show()

    mci_values = df_band[df_band['Group']=='MCI']['Strength']
    hc_values = df_band[df_band['Group']=='Healthy']['Strength']
    stat, p_val = stats.mannwhitneyu(mci_values, hc_values, alternative='two-sided')
    print(f"{band.capitalize()} Band:")
    print(f"  Mann-Whitney U: statistic={stat}, p-value={p_val:.5f}")
    if p_val < 0.05:
        print("Significant difference between MCI and Healthy groups")
    else:
        print("No significant difference")

#%% Task 2: Compare node strengths in functional networks  (8 points)

"""
Compute the node strength for each subject and frequency band.  
For each frequency band and functional network:
    Compute the average node strength for this network. 
    Run a test to see if values differ significantly between groups.
    Make a heatmap plot that shows the significant differences. 
    Label axes for frequency bands and networks, center z-axis on 0.

"""
node_strength_data = []
for band in bands:
    for subject_idx in range(total_subjects):
        file_path = f"iPLV/{band}/S{subject_idx}.npy"
        matrix = np.load(file_path)
        np.fill_diagonal(matrix, 0)
        node_strengths = matrix.sum(axis=1)
        for node_idx, n_str in enumerate(node_strengths):
            node_strength_data.append({
                'SubjectID': subject_idx,
                'Group': group_labels[subject_idx],
                'Band': band,
                'Node': node_idx,
                'NodeStrength': n_str
            })

df_node_strength = pd.DataFrame(node_strength_data)
# compute average node strength for each band
average_strength_data = []

for (band, subject_id), sub_df in df_node_strength.groupby(['Band', 'SubjectID']):
    group_label = sub_df['Group'].iloc[0]  # MCI or Healthy
    for net_id in range(len(netw_names)):
        # find which rows in sub_df correspond to net_id
        mask = (networks == net_id)
        network_nodes = sub_df[sub_df['Node'].isin(np.where(mask)[0])]
        if len(network_nodes) > 0:
            avg_str = network_nodes['NodeStrength'].mean()
        else:
            avg_str = np.nan
        average_strength_data.append({
            'SubjectID': subject_id,
            'Group': group_label,
            'Band': band,
            'NetworkID': net_id,
            'NetworkName': netw_names[net_id],
            'AvgNodeStrength': avg_str
        })

df_network_strength = pd.DataFrame(average_strength_data)

# comparing groups for each freq band and network
n_bands = len(bands)
n_networks = len(netw_names)
diff_matrix = np.zeros((n_bands, n_networks))
pval_matrix = np.ones((n_bands, n_networks))
for b_idx, band in enumerate(bands):
    for net_id in range(n_networks):
        df_sub = df_network_strength[
            (df_network_strength['Band'] == band) &
            (df_network_strength['NetworkID'] == net_id)
            ]
        mci_vals = df_sub[df_sub['Group'] == 'MCI']['AvgNodeStrength']
        hc_vals = df_sub[df_sub['Group'] == 'Healthy']['AvgNodeStrength']
        # compute difference in means
        mean_mci = np.mean(mci_vals)
        mean_hc = np.mean(hc_vals)
        diff_matrix[b_idx, net_id] = mean_mci - mean_hc

        # Mann-Whitney test
        if len(mci_vals) > 0 and len(hc_vals) > 0:
            stat, pval = stats.mannwhitneyu(mci_vals, hc_vals, alternative='two-sided')
            pval_matrix[b_idx, net_id] = pval
        else:
            pval_matrix[b_idx, net_id] = np.nan

annot_matrix = np.empty_like(pval_matrix, dtype=object)
annot_matrix[:] = ''
annot_matrix[pval_matrix < 0.05] = '*'

plt.figure(figsize=(12, 5))
ax = sb.heatmap(
    diff_matrix,
    center=0,
    cmap='bwr',
    annot=annot_matrix,
    fmt='',
    xticklabels=netw_names,
    yticklabels=bands,
    cbar_kws={'label': 'Difference in Mean Node Strength (MCI - Healthy)'}
)
ax.set_xlabel('Functional Network')
ax.set_ylabel('Frequency Band')
ax.set_title('Differences in Node Strength by Network and Band')
plt.tight_layout()
plt.show()

#%% Task 3: Get number of significant edges. ( 10 points)

"""
  
For each frequency band and edge, test if the edge strength values differ 
  significantly between groups.
For each frequency band, get the number of significant edges. 
In which band(s) is the number of significant edges higher 
   than would be expected by chance? 
Is the overal number of significant edges higher than expected by chance?   
   
"""

adjacency_data = {band: [] for band in bands}
test_file = f"iPLV/{bands[0]}/S0.npy"
temp_matrix = np.load(test_file)
N = temp_matrix.shape[0]
for band in bands:
    for subject_idx in range(total_subjects):
        file_path = f"iPLV/{band}/S{subject_idx}.npy"
        mat = np.load(file_path)
        adjacency_data[band].append(mat)

# for each band Mann–Whitney U (MCI vs Healthy) test for each edge
alpha = 0.05
band_significant_counts = {}
overall_significant_count = 0
total_edges_all_bands = 0

for band in bands:
    n_significant = 0
    # upper triangle edges
    total_edges = N * (N - 1) // 2
    for i in range(N):
        for j in range(i + 1, N):
            # gather MCI vs Healthy edge values
            edge_vals_mci = []
            edge_vals_hc = []
            for s_idx in range(total_subjects):
                edge_val = adjacency_data[band][s_idx][i, j]
                if group_labels[s_idx] == 'MCI':
                    edge_vals_mci.append(edge_val)
                else:
                    edge_vals_hc.append(edge_val)
            if len(edge_vals_mci) > 0 and len(edge_vals_hc) > 0:
                stat, pval = stats.mannwhitneyu(edge_vals_mci, edge_vals_hc, alternative='two-sided')
                if pval < alpha:
                    n_significant += 1

    band_significant_counts[band] = n_significant
    overall_significant_count += n_significant
    total_edges_all_bands += total_edges
    print(f"Band '{band}': {n_significant} significant edges out of {total_edges} (p < 0.05).")

# binomial test to check significant edges in each band

print("\nChecking if significant edges in each band")
for band in bands:
    count_sig = band_significant_counts[band]
    n_edges = N * (N - 1) // 2
    pval_binom = stats.binomtest(count_sig, n_edges, 0.05, alternative='greater')
    print(f"Band '{band}': {count_sig} / {n_edges} edges, "
          f"binomial test p-value={pval_binom.pvalue:.5f}. "
          f"\nNumber of significant edges higher than would be expected by chance? {pval_binom.pvalue < 0.05}")


print(f"Total significant edges across all bands = {overall_significant_count}, out of {total_edges_all_bands} total edges tested")
overall_pval_binom = stats.binomtest(overall_significant_count, total_edges_all_bands,0.05, alternative='greater')
print(f"Overall binomial test p-value={overall_pval_binom.pvalue:.5f}. "
      f"\nOverall number of significant edges higher than expected by chance? {overall_pval_binom.pvalue < 0.05}")


#%% Task 4: Connected spiking-neuron model (18 points)

"""
Now, we will pick up the Ihzikevich model that we also used in Ex. 6 and 
    simulate a neuronal population consisting of a large number (N=1000) 
    of interacting neurons. We will also add weights between the neurons, 
    and random external input.
    
The model will run for 1000 time points of 1 msec each.    
       
The 1000 neurons will be partially of excitatory (E) and inhibitory (I) type.
  Choose a ratio E/I that is likely to produce oscillation-like behaviour.
  (Check the lectures!)

The E and I neurons can have different values for a,b,c,d parameters.
All E neurons will have the same value, and all I neurons will 
   have the same value for each parameter.
   
   
Both the initial values and the updated values at each time point will 
  be calculated as before, only for each neuron separately.

The 1000x1000 weight matrix W will represent the effect that each neuron has on all the others.
  (We assume that each neuron is connected to each)
For each neuron, the weights for inputs coming from the E-neurons should be drawn from a uniform [0,0.5] distribution
   and for inputs coming from the I-neurons drawn from a uniform [-1,0] distribution.
   
The equation for the external input current as well the 
   equation that adds the currents from other cells are already in the code.

Write to a variable whenever a neuron is firing: which neuron and which time point.
Then make a plot of all neurons firing over time where each firing is a small dot.
Time should be on x-axis, neurons on y-axis.

The output should show oscillation-like behaviour in the collective firing
   after a short while! 
   
If you don't get oscillation-like firing, run again a few times. 
If not, check your code if it updates values correctly.
You can also try to change the parameters.
   
What is the approximate oscillation frequency in your plot? 
    (may vary from run to run)

"""

## given parameters

simulation_time = 1000
dt = 1
# Excitatory neurons                 Inhibitory neurons
Ne = 800 ;                             Ni = 200
N_tot = Ne + Ni
exc_indices = np.arange(0, Ne)
inh_indices = np.arange(Ne, N_tot)

a1 = .02    ;                        a2 = .025
b1 = .25    ;                        b2 = .34
c1 = -60    ;                        c2 = -65
d1 = 6      ;                        d2 = 2

v0    = -65.5
v_thr =  35.5


# create arrays for parameters
a = np.zeros(N_tot)
b = np.zeros(N_tot)
c = np.zeros(N_tot)
d = np.zeros(N_tot)
a[exc_indices] = a1;  a[inh_indices] = a2
b[exc_indices] = b1;  b[inh_indices] = b2
c[exc_indices] = c1;  c[inh_indices] = c2
d[exc_indices] = d1;  d[inh_indices] = d2

# initial values of v and u for each neuron
v = np.full(N_tot, v0, dtype=float)
u = b * v

# set the matrix W of all-to-all synaptic weights
W = np.zeros((N_tot, N_tot), dtype=float)
for idx in exc_indices:
    W[:, idx] = np.random.uniform(0, 0.5, size=N_tot)
for idx in inh_indices:
    W[:, idx] = np.random.uniform(-1, 0, size=N_tot)


# initializations

spike_times = []  # (t, neuron_index)
example_inh_neuron = Ne  # to record membrane voltage of 800th index inhibitory neuron
v_record = np.zeros(simulation_time)

# loop through simulation time
for t in range(simulation_time):

    I = np.hstack((5*np.random.randn(Ne),2*np.random.randn(Ni)))
    # check which neurons will fire an action potential
    fired = np.where(v >= v_thr)[0]
    if len(fired) > 0:
        for neuron_id in fired:
            spike_times.append((t, neuron_id))
        # update membrane variables for neurons that fired
        v[fired] = c[fired]
        u[fired] += d[fired]
    # add currents from neurons that fired
    if len(fired) > 0:
        I += np.sum(W[:,fired], axis=1)
  # update membrane voltage
    v += (0.04 * v ** 2 + 5 * v + 140 - u + I) * dt
    u += (a * (b * v - u)) * dt
    v_record[t] = v[example_inh_neuron]
spike_times = np.array(spike_times)
mean_firing_rate = len(spike_times) / (N_tot * (simulation_time/1000))

print("Mean firing rate: %.2f Hz" % mean_firing_rate)

plt.figure(figsize=(10,5))
plt.scatter(spike_times[:,0], spike_times[:,1], s=2, marker='.', color='k')
plt.title('Population Firing Raster Plot')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron')
plt.xlim([0, simulation_time])
plt.ylim([-1, N_tot+1])
plt.tight_layout()
plt.show()

# fast spiking voltage of inhibitory neuron
plt.figure(figsize=(10,5))
time_array = np.arange(simulation_time)
plt.plot(time_array, v_record, color='b')
plt.title('Inhibitory Neuron Voltage Trace')
plt.xlabel('Time (ms)')
plt.ylabel('mV')
plt.ylim([-90, 50])
plt.grid(True)
plt.tight_layout()
plt.show()


#%% Task 5: Alter firing frequency in the model (7 points)

"""
Now, change the parameters of the model so that you get approximately the
   double firing frequency as before in your firing pattern.

"""
spike_times = []
for t in range(simulation_time):
    I = np.hstack((10*np.random.randn(Ne),2*np.random.randn(Ni))) # doubled the excitatory input
    fired = np.where(v >= v_thr)[0]
    if len(fired) > 0:
        for neuron_id in fired:
            spike_times.append((t, neuron_id))
        v[fired] = c[fired]
        u[fired] += d[fired]
    if len(fired) > 0:
        I += np.sum(W[:,fired], axis=1)
    v += (0.04 * v ** 2 + 5 * v + 140 - u + I) * dt
    u += (a * (b * v - u)) * dt

spike_times = np.array(spike_times)
mean_firing_rate = len(spike_times) / (N_tot * (simulation_time/1000))
print("Mean firing rate with double excitatory input: %.2f Hz" % mean_firing_rate)

plt.figure(figsize=(10,5))
plt.scatter(spike_times[:,0], spike_times[:,1], s=2, marker='.', color='k')
plt.title('Population Firing Raster Plot (doubled excitatory input)')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron')
plt.xlim([0, simulation_time])
plt.ylim([-1, N_tot+1])
plt.tight_layout()
plt.show()