import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import spearmanr, ttest_ind
import matplotlib.pyplot as plt
from collections import defaultdict

#Task 0:

patient_data = pd.read_csv('Patient_data.csv', delimiter=';', header=None)
patient_data = patient_data.to_numpy()

#print(patient_data)

control_data = pd.read_csv('Controls_data.csv', delimiter=';', header=None)
control_data = control_data.to_numpy()

#print(control_data)

print(patient_data.shape)


#Task 1:

parcels = patient_data.shape[0]
#print(parcels)

t_stats = np.zeros(parcels)
p_vals = np.zeros(parcels)

for i in range(parcels):
    t_stat, p_val = ttest_ind(patient_data[i,:],control_data[i,:], equal_var=False)
    t_stats[i] = t_stat
    p_vals[i] = p_val

diff = p_vals < 0.05
diff_sum = np.sum(diff)
print(f'Significant difference in {diff_sum} parcels')

larger_act = (patient_data.mean(axis=1) > control_data.mean(axis=1)) & diff
larger_act = np.sum(larger_act)

print(f'Activity larger in patients in {larger_act} parcels')

#Task 2:

with open('parcel_names.txt', 'r') as file:
    names_list = [line.strip() for line in file.readlines()]

parcel_network = defaultdict(list) #key = network name, value = list of indices of parcels

for i, parcel in enumerate(names_list):
    network_name = parcel.split('_')[2] #Format: 7Networks_LH_Cont_Cing_1__Left, Network name: Cont
    parcel_network[network_name].append(i)

#print(network)

counts = {network_key : len(indices) for network_key, indices in parcel_network.items()}

#print(counts)

for network_name, count in counts.items():
    print(f'Network: {network_name}, Number of parcels: {count}')


#Task 3:
patient_dictionary = {}
control_dictionary = {}

for network in parcel_network:
    patient_dictionary[network] = np.zeros(patient_data.shape[1])

for network in parcel_network:
    control_dictionary[network] = np.zeros(control_data.shape[1])

for network, indices in parcel_network.items():
    patient_dictionary[network] = patient_data[indices, :].mean(axis=0)
    control_dictionary[network] = control_data[indices, :].mean(axis=0)

test_stat_net = {}
net_p = {}

for network in parcel_network:
    t_stat, p_val = ttest_ind(
        patient_dictionary[network],
        control_dictionary[network],
        equal_var=False
    )
    test_stat_net[network] = t_stat
    net_p[network] = p_val

sig_nets = {}
for network, p_val in net_p.items():
    if p_val < 0.05:
        sig_nets[network] = p_val

print("\nSignificant networks (p < 0.05):")
for network, p_val in sig_nets.items():
    mean_patient = patient_dictionary[network].mean()
    mean_control = control_dictionary[network].mean()
    if mean_patient > mean_control:
        print(f"Network: {network} - Brain activity is significantly larger in patients (p = {p_val:.4f}).")
    else:
        print(f"Network: {network} - Brain activity is significantly smaller in patients (p = {p_val:.4f}).")

#Task 4:

net_names = list(parcel_network.keys())
mean_patient_activity = []
std_patient_activity = []
mean_control_activity = []
std_control_activity = []

for network in net_names:
    mean_patient_activity.append(patient_dictionary[network].mean())
    std_patient_activity.append(patient_dictionary[network].std())
    mean_control_activity.append(control_dictionary[network].mean())
    std_control_activity.append(control_dictionary[network].std())

plt.figure(figsize=(10, 6))
x = np.arange(len(net_names))
width = 0.35

plt.bar(x - width/2, mean_patient_activity, width, yerr=std_patient_activity, capsize=5, label='Patients', alpha=0.7)
plt.bar(x + width/2, mean_control_activity, width, yerr=std_control_activity, capsize=5, label='Controls', alpha=0.7)
plt.xticks(x, net_names, rotation=45, ha='right')
plt.xlabel('Network')
plt.ylabel('Mean Brain Activity')
plt.title('Mean Brain Activity per Network in Each Cohort')
plt.legend()
plt.tight_layout()
plt.show()

network_list = []
brain_activity_list = []
cohort_list = []

for network in net_names:
    for activity in patient_dictionary[network]:
        network_list.append(network)
        brain_activity_list.append(activity)
        cohort_list.append('Patients')

    for activity in control_dictionary[network]:
        network_list.append(network)
        brain_activity_list.append(activity)
        cohort_list.append('Controls')

violin_data = pd.DataFrame({
    'Network': network_list,
    'Brain Activity': brain_activity_list,
    'Cohort': cohort_list
})

plt.figure(figsize=(10, 6))
sns.violinplot(data=violin_data, x='Network', y='Brain Activity', hue='Cohort', split=True, inner='quartile')
plt.xticks(rotation=45, ha='right')
plt.title('Distribution of Subjects\' per Network in Violinplots')
plt.xlabel('Network')
plt.ylabel('Brain Activity')
plt.tight_layout()
plt.show()

#Task 5

symptom_scores = pd.read_csv('symptom_scores.txt', header=None, names=['Score'])['Score'].to_numpy()

plt.figure(figsize=(10, 6))
plt.hist(symptom_scores, bins=20, edgecolor='black', alpha=0.7)
plt.title('Distribution of Patients\' Symptom Scores')
plt.xlabel('Symptom Score')
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#Task 6

parcel_corr = np.zeros(parcels)
parcel_p = np.zeros(parcels)

for i in range(parcels):
    corr, p_val = spearmanr(symptom_scores, patient_data[i, :])
    parcel_corr[i] = corr
    parcel_p[i] = p_val

sig_parcels = []

for index, p_val in enumerate(parcel_p):
    if p_val <0.05:
        sig_parcels.append(index)

print(f"Number of parcels with significant correlations: {len(sig_parcels)}")

no_sig_nets = defaultdict(int)
for idx in sig_parcels:
    for network, indices in parcel_network.items():
        if idx in indices:
            no_sig_nets[network] += 1

most_significant_network = max(no_sig_nets, key=no_sig_nets.get)
print(f"Network with most significant correlations: {most_significant_network} ({no_sig_nets[most_significant_network]} significant parcels)")

#Task 7

net_corr = {}
net_p = {}


for network, indices in parcel_network.items():
    network_mean_activity = np.mean(patient_data[indices, :], axis=0)
    corr, p_val = spearmanr(symptom_scores, network_mean_activity)
    net_corr[network] = corr
    net_p[network] = p_val

print("Correlations for all networks:")
for network in net_corr:
    print(f"Network: {network}, Correlation: {net_corr[network]}, p-value: {net_p[network]}")

sig_net_corr = {}

for network in net_corr:
    if net_p[network] < 0.05:
        sig_net_corr[network] = (net_corr[network], net_p[network])

print("\nSignificant correlations at the network level:")
for network, (corr, p_val) in sig_net_corr.items():
    print(f"Network: {network}, Correlation: {corr}, p-value: {p_val}")

#Task 8

for network in sig_net_corr:
    network_mean_activity = patient_data[parcel_network[network], :].mean(axis=0)

    plt.figure(figsize=(10, 6))

    plt.scatter(symptom_scores, network_mean_activity, label="Data Points", color='blue', alpha=0.6)
    plt.title(f'Relationship Between Symptom Scores and {network} Mean Activity')
    plt.xlabel('Symptom Scores')
    plt.ylabel(f'{network} Mean Brain Activity')

    regression_plot = sns.regplot(
        x=symptom_scores,
        y=network_mean_activity,
        scatter=False,
        color='red',
        line_kws={"linestyle": "--"}
    )

    plt.legend()
    plt.grid(visible=True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()
