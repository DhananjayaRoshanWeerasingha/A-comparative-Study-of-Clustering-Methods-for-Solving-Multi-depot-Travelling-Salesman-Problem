import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import haversine_distances
from sklearn.metrics import silhouette_score

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


# Calculate the distance matrix between data points
distance_matrix = haversine_distances(np.radians(data[['y', 'x']].values),
                                      np.radians(depot[['y', 'x']].values))
# Define the scaling factor
sigma = 0.1

# Calculate the degree of membership using a Gaussian membership function
membership_matrix = np.exp(-distance_matrix**2 / (2 * sigma**2))

print(membership_matrix)

# Initialize an empty array for the updated depots
updated_depots = np.zeros((len(depot), 2))


# Loop over the clusters
for j in range(len(depot)):
    # Calculate the sum of the membership values for cluster j
    sum_wj = np.sum(membership_matrix[:, j])
    
    # Calculate the weighted average of the data points in cluster j
    if sum_wj > 0:
        updated_depots[j, :] = np.sum(membership_matrix[:, j][:, np.newaxis] * data[['y', 'x']].values, axis=0) / sum_wj
    else:
        # If the sum of the membership values is 0, set the updated depot to the original depot
        updated_depots[j, :] = depot[['y', 'x']].iloc[j].values

print(updated_depots)

# Define a function to calculate the fuzzy c-means objective function
def fuzzy_c_means_objective_function(data, updated_depots, membership_matrix):
    # Initialize the objective function value
    obj = 0

    # Loop over the data points
    for i in range(len(data)):
        # Calculate the sum of the weighted distances between the data point and each updated depot
        sum_wd = 0
        for j in range(len(updated_depots)):
            sum_wd += membership_matrix[i, j] * haversine_distances(np.radians(data.iloc[i][['y', 'x']].values.reshape(1, -1)), np.radians(updated_depots[j, :].reshape(1, -1)))

        # Add the sum of weighted distances to the objective function value
        obj += sum_wd

    return obj

# Call the function to calculate the objective function value
obj = fuzzy_c_means_objective_function(data, updated_depots, membership_matrix)

print(obj)

# Determine cluster membership
cluster_membership = np.argmax(membership_matrix, axis=1)

# Add the cluster membership column to the data dataframe
data['cluster'] = cluster_membership

# Print the updated data dataframe
print(data)

# Calculate the silhouette score for the fuzzy c-means clustering
silhouette_avg = silhouette_score(data[['y', 'x']], cluster_membership)
print("The average silhouette score is :", silhouette_avg)

# Interpretation:
# A silhouette score close to 1 indicates that the data points are well-clustered and separated from other clusters.
# A silhouette score close to -1 indicates that the data points are poorly clustered and may belong to another cluster.
# A silhouette score close to 0 indicates that the data points are on the boundary between two clusters.