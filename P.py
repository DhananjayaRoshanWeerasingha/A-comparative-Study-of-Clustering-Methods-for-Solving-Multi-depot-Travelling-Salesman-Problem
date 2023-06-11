import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import haversine_distances

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

# Define a function to calculate Euclidean distance
def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# Initialize the number of clusters and maximum number of iterations
k = 2
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
    #print(f"\nDistance matrix for cluster {name}:")
    #print(distance_matrix)
    # create instance of AntColonyOptimizer
    aco = AntColonyOptimizer(n_ants, n_iterations, decay_factor, alpha, beta, q)
    # run ant colony optimization algorithm
    aco.fit(distance_matrix)
    # display best solution and best fitness found by the algorithm
    print("Best solution:", aco.best_solution)
    print("Best fitness:", aco.best_fitness)