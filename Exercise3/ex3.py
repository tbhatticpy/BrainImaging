import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nilearn import connectome, maskers, plotting, datasets
plt.style.use('default')


current_dir = os.path.dirname(os.path.abspath(__file__))

data_dir = os.path.join(current_dir, 'nilearn_data')

data = datasets.fetch_adhd(n_subjects=40, data_dir=data_dir)


df = pd.read_csv('nilearn_data/adhd/ADHD200_40subs_motion_parameters_and_phenotypics.csv')
#rint(df.head())

patients_indices = df[df['adhd'] == 1].index.tolist() #Where 'adhd' column is 1
controls_indices = df[df['adhd'] == 0].index.tolist() #Where 'adhd' column is 0
print("Patients indices:", patients_indices)
print("Controls indices:", controls_indices)

atlas = datasets.fetch_atlas_msdl()
labels = atlas.labels[1:]
coords = atlas.region_coords

#print(labels)
#print(coords)

masker = maskers.NiftiMapsMasker(maps_img=atlas.maps,
                                 standardize="zscore_sample",
                                 standardize_confounds="zscore_sample")

time_series = masker.fit_transform(data.func[0]) #Had to add fit before transform according to nilearn documentation

#print('Time series of the first subject: ', time_series)
print("Time series shape: ", time_series.shape)

#Task 1

#corr_matrix = np.corrcoef(time_series.T)

conn_measure = connectome.ConnectivityMeasure(kind='correlation')
correlation_matrix = conn_measure.fit_transform([time_series])[0]
#print(correlation_matrix)
print("Shape of the Correlation Matrix: ", correlation_matrix.shape)


plt.figure(figsize=(10, 8))
plt.imshow(correlation_matrix, interpolation='nearest', cmap='RdBu_r', vmin=-1, vmax=1)
plt.colorbar()
plt.title("Functional Connectivity Matrix (Matplotlib)")
plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=90, fontsize=8)
plt.yticks(ticks=np.arange(len(labels)), labels=labels, fontsize=8)
plt.tight_layout()
plt.show()


#Task 2

def threshold_and_plot(correlation_matrix, coords, percentile, title):
  n_regions = correlation_matrix.shape[0]
  upper_tri_idx = np.triu_indices(n_regions, k=1)
  upper_tri = correlation_matrix[upper_tri_idx]

  # Calculate the threshold value
  threshold_value = np.percentile(np.abs(upper_tri), 100 - percentile)
  print(f"Threshold value for top {percentile}% strongest connections: {threshold_value}")

  # Create the thresholded matrix
  threshold_matrix = np.zeros_like(correlation_matrix)
  strong_connections = np.abs(correlation_matrix) >= threshold_value
  threshold_matrix[strong_connections] = correlation_matrix[strong_connections]

  # Ensure the matrix is symmetric and zero out the diagonal
  threshold_matrix = np.triu(threshold_matrix, k=1)
  threshold_matrix += threshold_matrix.T
  np.fill_diagonal(threshold_matrix, 0)

  num_connections = np.count_nonzero(threshold_matrix) // 2
  total_possible_connections = n_regions * (n_regions - 1) // 2
  proportion_retained = num_connections / total_possible_connections * 100
  print(f"Number of connections retained: {num_connections}")
  print(f"Proportion of connections retained: {proportion_retained:.2f}%")

  plotting.plot_connectome(
    threshold_matrix,
    coords,
    node_color='auto',
    node_size=50,
    edge_cmap='RdBu_r',
    edge_vmin=-1, edge_vmax=1,
    title=title
  )
  plt.show()

  return threshold_matrix


threshold_matrix_5 = threshold_and_plot(correlation_matrix, coords, percentile=5, title="Top 5% Strongest Connections")

threshold_matrix_10 = threshold_and_plot(correlation_matrix, coords, percentile=10, title="Top 10% Strongest Connections")

threshold_matrix_30 = threshold_and_plot(correlation_matrix, coords, percentile=30, title="Top 30% Strongest Connections")


#Task 3

degrees = np.sum(threshold_matrix_30 != 0, axis=1)
max_degree = np.max(degrees)
max_degree_nodes = np.where(degrees == max_degree)[0]
print(f"All degrees, {degrees}")
print(f"Maximum degree: {max_degree}")
print("Nodes with maximum degree:", max_degree_nodes)

max_degree_labels = [labels[idx] for idx in max_degree_nodes]

print("Labels of nodes with maximum degree:", max_degree_labels)

for node_index in max_degree_nodes:
    node_connections = correlation_matrix[node_index, :].copy()
    node_connections[node_index] = 0

    abs_connections = np.abs(node_connections)
    sorted_conn_indices = np.argsort(-abs_connections)
    top_8_indices = sorted_conn_indices[:8]

    node_threshold_matrix = np.zeros_like(correlation_matrix)

    for idx in top_8_indices:
        node_threshold_matrix[node_index, idx] = correlation_matrix[node_index, idx]
        node_threshold_matrix[idx, node_index] = correlation_matrix[idx, node_index]

    node_label = labels[node_index]
    title = f"Top 8 connections of node: {node_label}"

    node_colors = ['lightgray'] * correlation_matrix.shape[0]
    node_colors[node_index] = 'red'

    plotting.plot_connectome(
        node_threshold_matrix,
        coords,
        node_color=node_colors,
        node_size=50,
        edge_cmap='RdBu_r',
        edge_vmin=-1, edge_vmax=1,
        title=title
    )
    plt.show()

# Task 4

time_series_list = []
correlation_matrices = []

conn_measure = connectome.ConnectivityMeasure(kind='correlation')

for func_file in data.func:
    time_series = masker.transform(func_file)
    time_series_list.append(time_series)

    correlation_matrix = conn_measure.fit_transform([time_series])[0]
    correlation_matrices.append(correlation_matrix)

correlation_matrices = np.array(correlation_matrices)
print("Shape of correlation_matrices:", correlation_matrices.shape)

assert correlation_matrices.shape[0] == df.shape[0], "Mismatch in number of subjects"

patients_matrices = correlation_matrices[patients_indices]
controls_matrices = correlation_matrices[controls_indices]

mean_correlation_matrix_all = np.mean(correlation_matrices, axis=0)

mean_correlation_matrix_patients = np.mean(patients_matrices, axis=0)

mean_correlation_matrix_controls = np.mean(controls_matrices, axis=0)

plt.figure(figsize=(10, 8))
plt.imshow(mean_correlation_matrix_all, interpolation='nearest', cmap='RdBu_r', vmin=-1, vmax=1)
plt.colorbar()
plt.title("Group Mean Connectivity Matrix - All Subjects")
plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=90, fontsize=8)
plt.yticks(ticks=np.arange(len(labels)), labels=labels, fontsize=8)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
plt.imshow(mean_correlation_matrix_patients, interpolation='nearest', cmap='RdBu_r', vmin=-1, vmax=1)
plt.colorbar()
plt.title("Group Mean Connectivity Matrix - Patients")
plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=90, fontsize=8)
plt.yticks(ticks=np.arange(len(labels)), labels=labels, fontsize=8)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
plt.imshow(mean_correlation_matrix_controls, interpolation='nearest', cmap='RdBu_r', vmin=-1, vmax=1)
plt.colorbar()
plt.title("Group Mean Connectivity Matrix - Controls")
plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=90, fontsize=8)
plt.yticks(ticks=np.arange(len(labels)), labels=labels, fontsize=8)
plt.tight_layout()
plt.show()

difference_matrix = mean_correlation_matrix_patients - mean_correlation_matrix_controls

plt.figure(figsize=(10, 8))
max_diff = np.max(np.abs(difference_matrix))
plt.imshow(difference_matrix, interpolation='nearest', cmap='RdBu_r', vmin=-max_diff, vmax=max_diff)
plt.colorbar()
plt.title("Difference in Connectivity (Patients - Controls)")
plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=90, fontsize=8)
plt.yticks(ticks=np.arange(len(labels)), labels=labels, fontsize=8)
plt.tight_layout()
plt.show()


# Task 5

def threshold_mean_matrix(mean_matrix, percentile):
    n_regions = mean_matrix.shape[0]
    upper_tri_idx = np.triu_indices(n_regions, k=1)
    upper_tri = mean_matrix[upper_tri_idx]

    threshold_value = np.percentile(np.abs(upper_tri), 100 - percentile)
    print(f"Threshold value for top {percentile}% strongest connections: {threshold_value}")

    threshold_matrix = np.zeros_like(mean_matrix)
    strong_connections = np.abs(mean_matrix) >= threshold_value
    threshold_matrix[strong_connections] = mean_matrix[strong_connections]

    threshold_matrix = np.triu(threshold_matrix, k=1)
    threshold_matrix += threshold_matrix.T
    np.fill_diagonal(threshold_matrix, 0)

    return threshold_matrix


percentile = 30

threshold_matrix_patients = threshold_mean_matrix(mean_correlation_matrix_patients, percentile)
threshold_matrix_controls = threshold_mean_matrix(mean_correlation_matrix_controls, percentile)

degrees_patients = np.sum(threshold_matrix_patients != 0, axis=1)
degrees_controls = np.sum(threshold_matrix_controls != 0, axis=1)

degree_difference = degrees_patients - degrees_controls

abs_degree_difference = np.abs(degree_difference)
sorted_indices = np.argsort(-abs_degree_difference)
top_7_indices = sorted_indices[:7]

print("\nDegrees in Patients:", degrees_patients)
print("Degrees in Controls:", degrees_controls)
print("Degree Difference (Patients - Controls):", degree_difference)
print("\nTop 7 parcels with largest degree differences:")
for idx in top_7_indices:
    print(f"Parcel: {labels[idx]}, Degree Difference: {degree_difference[idx]}")

import networkx as nx


def compute_betweenness_centrality(threshold_matrix):
    G = nx.from_numpy_array(threshold_matrix)
    betweenness = nx.betweenness_centrality(G, weight='weight', normalized=True)
    betweenness_array = np.array([betweenness[i] for i in range(len(betweenness))])
    return betweenness_array


betweenness_patients = compute_betweenness_centrality(threshold_matrix_patients)
betweenness_controls = compute_betweenness_centrality(threshold_matrix_controls)

betweenness_difference = betweenness_patients - betweenness_controls

abs_betweenness_difference = np.abs(betweenness_difference)
sorted_indices_bc = np.argsort(-abs_betweenness_difference)
top_7_indices_bc = sorted_indices_bc[:7]

print("\nBetweenness Centrality in Patients:", betweenness_patients)
print("Betweenness Centrality in Controls:", betweenness_controls)
print("Betweenness Centrality Difference (Patients - Controls):", betweenness_difference)
print("\nTop 7 parcels with largest betweenness centrality differences:")
for idx in top_7_indices_bc:
    print(f"Parcel: {labels[idx]}, Betweenness Centrality Difference: {betweenness_difference[idx]}")
