import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import haversine_distances
from sklearn.metrics import silhouette_score

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

grouped_data = data.groupby('cluster')

# Loop over the groups and print the data points assigned to each depot
for name, group in grouped_data:
    # Get the x and y coordinate values of the assigned depot
    depot_x, depot_y = updated_depots[name]

    # Concatenate the data x and y values with the depot x and y values
    cluster_points = np.concatenate((group[['x', 'y']].values, [[depot_x, depot_y]]))

    print(f"Data points assigned to Depot {name} (x={depot_x}, y={depot_y}):")
    print(cluster_points)
    print("\n")

 # Calculate the distance matrix between the cluster points
    distance_matrix = haversine_distances(np.radians(cluster_points[:, 0:2]))
    print(f"Distance matrix for cluster {name}:")
    print(distance_matrix)
    print("\n")
    # create instance of AntColonyOptimizer
    aco = AntColonyOptimizer(n_ants, n_iterations, decay_factor, alpha, beta, q)
    # run ant colony optimization algorithm
    aco.fit(distance_matrix)
    # display best solution and best fitness found by the algorithm
    print("Best solution:", aco.best_solution)
    print("Best fitness:", aco.best_fitness)    
                                