import pandas as pd
import numpy as np

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

# Calculate the square root of the Euclidean distance between each depot and each data point
distances = np.zeros((len(depot), len(data)))
for i in range(len(depot)):
    for j in range(len(data)):
        distances[i][j] = np.sqrt((depot.iloc[i]['x'] - data.iloc[j]['x'])**2 + (depot.iloc[i]['y'] - data.iloc[j]['y'])**2)

print(distances)

# Assign data points to the nearest depot
assignments = {}
for j in range(len(data)):
    nearest_depot = np.argmin(distances[:, j])
    if nearest_depot in assignments:
        assignments[nearest_depot].append(data.iloc[j]['id'])
    else:
        assignments[nearest_depot] = [data.iloc[j]['id']]

print(assignments)

# Calculate the new depot locations
new_depots = []
for depot_id, assigned_data_points in assignments.items():
    assigned_data = data[data['id'].isin(assigned_data_points)]
    new_depot_x = assigned_data['x'].mean()
    new_depot_y = assigned_data['y'].mean()
    new_depots.append({'id': depot_id, 'x': new_depot_x, 'y': new_depot_y})

# Convert the new depots to a pandas dataframe
new_depot_df = pd.DataFrame(new_depots, columns=['id', 'x', 'y'])

# Print the new depot locations
print(new_depot_df)

# Recalculate the distance between each data point and the new obtained depot points
for i in range(len(new_depots)):
    for j in range(len(data)):
        distances[i][j] = np.sqrt((new_depots[i]['x'] - data.iloc[j]['x'])**2 + (new_depots[i]['y'] - data.iloc[j]['y'])**2)

print(distances)

# Recalculate the distance between each data point and the new obtained depot points
for i in range(len(new_depots)):
    for j in range(len(data)):
        distances[i][j] = np.sqrt((new_depots[i]['x'] - data.iloc[j]['x'])**2 + (new_depots[i]['y'] - data.iloc[j]['y'])**2)

# Assign data points to the nearest depot
assignments2 = {}
for j in range(len(data)):
    nearest_depot = np.argmin(distances[:, j])
    if nearest_depot in assignments2:
        assignments2[nearest_depot].append(data.iloc[j]['id'])
    else:
        assignments2[nearest_depot] = [data.iloc[j]['id']]

print( assignments2)

# Print the x and y coordinates of the assigned data point
for i in range(len(assignments2)):
    if i in assignments2:
        for j in assignments2[i]:
            nearest_depot = i
            print("Assigned data point (id={}): ({}, {}) to depot {}".format(
                data.iloc[j-1]['id'], data.iloc[j-1]['x'], data.iloc[j-1]['y'], nearest_depot))
            
#Calculate the pairwise distances between points in each cluster 
if len(assignments2) > 0:
    for idx in range(len(assignments2)-1):
        Cluster = data.iloc[assignments2[idx]]
        num_points = len(Cluster)
        Cluster_distances = np.zeros((num_points, num_points))
        depot_point = depot.iloc[idx]

         # add depot point as the first point in the cluster
        Cluster = pd.concat([pd.DataFrame([[depot_point['x'], depot_point['y']]], columns=['x', 'y']), Cluster], ignore_index=True)

        for i in range(num_points):
            for j in range(num_points):
                point_i = Cluster.iloc[i]
                point_j = Cluster.iloc[j]
                distance = np.sqrt((point_i['x'] - point_j['x'])**2 + (point_i['y'] - point_j['y'])**2)
                Cluster_distances[i-1][j-1] = distance

        print(f"Cluster {idx+1} distances:")
       # print(Cluster_distances)
        # create instance of AntColonyOptimizer
        aco = AntColonyOptimizer(n_ants, n_iterations, decay_factor, alpha, beta, q)
        # run ant colony optimization algorithm
        aco.fit(Cluster_distances)
        # display best solution and best fitness found by the algorithm
        print("Best solution:", aco.best_solution)
        print("Best fitness:", aco.best_fitness)


    Cluster = data.iloc[assignments2[0]]
    num_points = len(Cluster)
    Cluster_distances = np.zeros((num_points, num_points))

    for i in range(num_points):
        for j in range(num_points):
            point_i = Cluster.iloc[i]
            point_j = Cluster.iloc[j]
            distance = np.sqrt((point_i['x'] - point_j['x'])**2 + (point_i['y'] - point_j['y'])**2)
            Cluster_distances[i][j] = distance

    print(f"Cluster {len(assignments2)} distances:")
    #print(Cluster_distances)
            
    # create instance of AntColonyOptimizer
    aco = AntColonyOptimizer(n_ants, n_iterations, decay_factor, alpha, beta, q)
    # run ant colony optimization algorithm
    aco.fit(Cluster_distances)
    # display best solution and best fitness found by the algorithm
    print("Best solution:", aco.best_solution)
    print("Best fitness:", aco.best_fitness)
    solution=aco.best_solution
    solution_new=[]   
else:
    print("assignments2 is empty.")
    