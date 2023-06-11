import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform

# Read the data from the Excel sheet
data = pd.read_excel('Book1.xlsx', sheet_name='Sheet1', names=['id', 'x', 'y'])
depot = pd.read_excel('Book1.xlsx', sheet_name='Sheet2', names=['id', 'x', 'y'])

# Assign names to the matrices
matrix_1_name = 'A'
matrix_2_name = 'B'
matrix_3_name = 'C'

# Print the data and depot dataframes
print(matrix_1_name)
print(data)

print(matrix_2_name)
print(depot)

# Add depot value to data
data2 = pd.concat([depot, data], ignore_index=True)  # Concatenate dataframes
data2 = data2[['id', 'x', 'y']]  # Reorder columns
data2['id'] = range(1, len(data2) + 1)  # Reset id sequence starting from 1

# Print the updated data and depot dataframes
print(matrix_3_name)
print(data2)

# Calculate the pairwise similarity between data points using the Gaussian kernel
similarity = rbf_kernel(data2[['x', 'y']], gamma=1.0/len(data2))

# Construct the affinity matrix using the pairwise similarity
affinity_matrix = np.zeros((len(data2), len(data2)))
for i in range(len(data2)):
    for j in range(len(data2)):
        affinity_matrix[i,j] = similarity[i,j]

# Print the affinity matrix
print(affinity_matrix)

# Perform Normalized Spectral Clustering on the affinity matrix
n_clusters = 4 # Choose the number of clusters
sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=0, n_init=54)
labels = sc.fit_predict(affinity_matrix)

# Print the cluster labels
print(labels)

# Apply k-means clustering algorithm to the transformed matrix
kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=54)
kmeans.fit(labels.reshape(-1, 1))

# Print the cluster labels
print(kmeans.labels_)

# Assign each data point to the nearest cluster
cluster_labels = kmeans.predict(labels.reshape(-1, 1))
print(cluster_labels)

# Add cluster labels as a new column to data2 dataframe
data2['cluster'] = cluster_labels

# Group the data by cluster labels
grouped_data = data2.groupby('cluster')

# Print the clustered data separately
for cluster_label, cluster_data in grouped_data:
    print(f"Cluster {cluster_label}:")
    print(cluster_data)
    print()

# Calculate the pairwise distances between points in each cluster
for cluster_label, cluster_data in grouped_data:
    print(f"Pairwise distances for cluster {cluster_label}:")
    pairwise_distances = pdist(cluster_data[['x', 'y']])
    pairwise_distances_matrix = squareform(pairwise_distances)
    print(pairwise_distances_matrix)
    print()