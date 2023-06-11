import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform

# Create instance of AntColonyOptimizer
n_ants = 6
n_iterations = 1
decay_factor = 0.95
alpha = 1
beta = 2
q = 1

class AntColonyOptimizer:
    def __init__(self, n_ants, n_iterations, decay_factor, alpha, beta, q):
        self.n_ants = n_ants # number of ants
        self.n_iterations = n_iterations  # number of iterations
        self.decay_factor = decay_factor  # pheromone decay factor
        self.alpha = alpha  # pheromone exponent
        self.beta = beta  # heuristic exponent
        self.q = q  # pheromone intensity
        self.distances = None  # distance matrix
        self.pheromone = None  # pheromone matrix
        self.best_solution = None  # best solution found
        self.best_fitness = np.inf  # best fitness found

    def fit(self, distances):
        self.distances = distances
        n_cities = distances.shape[0]
        self.pheromone = np.ones((n_cities, n_cities))  # initialize pheromone matrix

        for i in range(self.n_iterations):
            solutions = self._generate_solutions()
            self._update_pheromone(solutions)
            best_solution, best_fitness = self._get_best_solution(solutions)
            if best_fitness < self.best_fitness:
                self.best_solution = best_solution
                self.best_fitness = best_fitness
            self.pheromone *= self.decay_factor  # decay pheromone

    def _generate_solutions(self):
        solutions = []
        for ant in range(self.n_ants):
            visited_cities = [0]  # start from city 0
            while len(visited_cities) < self.distances.shape[0]:
                unvisited_cities = set(range(self.distances.shape[0])) - set(visited_cities)
                current_city = visited_cities[-1]
                next_city = self._choose_next_city(current_city, unvisited_cities)
                visited_cities.append(next_city)
            solutions.append(visited_cities)
        return solutions

    def _choose_next_city(self, current_city, unvisited_cities):
        pheromone = self.pheromone[current_city, list(unvisited_cities)]
        distance = self.distances[current_city, list(unvisited_cities)]
        heuristic = 1 / distance
        probabilities = np.power(pheromone, self.alpha) * np.power(heuristic, self.beta)
        probabilities /= np.sum(probabilities)
        next_city = list(unvisited_cities)[np.random.choice(range(len(probabilities)), p=probabilities)]
        return next_city

    def _update_pheromone(self, solutions):
        for i in range(self.n_ants):
            path = solutions[i]
            path_length = sum([self.distances[path[j-1], path[j]] for j in range(1, len(path))])
            for j in range(1, len(path)):
                self.pheromone[path[j-1], path[j]] += self.q / path_length

    def _get_best_solution(self, solutions):
        best_solution = min(solutions, key=lambda x: sum([self.distances[x[j-1], x[j]] for j in range(1, len(x))]))
        best_fitness = sum([self.distances[best_solution[j-1], best_solution[j]] for j in range(1, len(best_solution))])
        return best_solution, best_fitness

# Read the data from the Excel sheet
data = pd.read_excel('Book20.xlsx', sheet_name='Sheet1', names=['id', 'x', 'y'])
depot = pd.read_excel('Book20.xlsx', sheet_name='Sheet2', names=['id', 'x', 'y'])

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
n_clusters = 2 # Choose the number of clusters
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
    #print()
    # create instance of AntColonyOptimizer
    aco = AntColonyOptimizer(n_ants, n_iterations, decay_factor, alpha, beta, q)
    # run ant colony optimization algorithm
    aco.fit(pairwise_distances_matrix)
    # display best solution and best fitness found by the algorithm
    print("Best solution:", aco.best_solution)
    print("Best fitness:", aco.best_fitness)

    