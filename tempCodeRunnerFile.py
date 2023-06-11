import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import haversine_distances

# Read the data from the Excel sheet
data = pd.read_excel('Book1.xlsx', sheet_name='Sheet1', names=['id', 'x', 'y'])
depot = pd.read_excel('Book1.xlsx', sheet_name='Sheet2', names=['id', 'x', 'y'])

# Assign names to the matrices
matrix_1_name = 'A'
matrix_2_name = 'B'

# Print the data and depot dataframes
print(matrix_1_name)
print(data)

print(matrix_2_name)
print(depot)

# Define a function to calculate Euclidean distance
def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# Initialize the number of clusters and maximum number of iterations
k = 4
max_iterations = 50

# Loop until convergence or maximum number of iterations is reached
for iteration in range(max_iterations):
    print("Iteration", iteration + 1)
    
    # Calculate the distance between each data point and each depot
    distances = []
    for i in range(len(data)):
        row = []
        for j in range(len(depot)):
            d = euclidean_distance(data['x'][i], data['y'][i], depot['x'][j], depot['y'][j])
            row.append(d)
        distances.append(row)

    # Assign each data point to the cluster whose depot is closest to it
    clusters = np.argmin(distances, axis=1)

    # Add a new column to the data dataframe to indicate the cluster assignment
    data['cluster'] = clusters

    # Group the data by cluster and calculate the mean of each group
    cluster_means = data.groupby('cluster').mean()

    # Create a new dataframe to store the updated depot locations
    updated_depot = pd.DataFrame(columns=['id', 'x', 'y'])

    # Loop through the clusters and add the mean location to the updated depot dataframe
    for i in range(len(cluster_means)):
        id = i + 1
        x = cluster_means['x'][i]
        y = cluster_means['y'][i]
        new_row = pd.DataFrame({'id': id, 'x': x, 'y': y}, index=[0])
        updated_depot = pd.concat([updated_depot, new_row], ignore_index=True)

    # Check if the updated depots are the same as the previous iteration
    if (depot == updated_depot).all().all():
        print("Converged after", iteration + 1, "iterations.")
        break

    # Update the depots for the next iteration
    depot = updated_depot.copy()

# Print the final clusters and depots
print("Final clusters:")
print(data)
print("Final depots:")
print(depot)

cluster_sizes = data.groupby('cluster').size()
print("Cluster sizes:")
print(cluster_sizes)

print("Cluster centroids:")
print(cluster_means)

# Group the data by cluster
grouped_data = data.groupby('cluster')

# Loop over the groups and print the data points assigned to each depot
for name, group in grouped_data:
    # Get the x and y coordinate values of the assigned depot
    depot_x, depot_y = depot.loc[name, ['x', 'y']]

    # Concatenate the data x and y values with the depot x and y values
    cluster_points = np.concatenate((group[['x', 'y']].values, [[depot_x, depot_y]]))

    # Print the data points assigned to the current depot
    print(f"\nData points assigned to Depot {name} (x={depot_x}, y={depot_y}):")
    print(cluster_points)

    # Calculate the distance matrix between the cluster points
    distance_matrix = haversine_distances(np.radians(cluster_points[:, :2]))

    # Print the distance matrix
    print(f"\nDistance matrix for cluster {name}:")
    print(distance_matrix)
