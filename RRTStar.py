import numpy as np
import pybullet as p
import pybullet_data
import random
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import heapq

class RRTStar:
    def __init__(self, robot, gamma_rrt_star=1.0, eta=0.1, max_iterations=10000, goal_threshold=0.15, goal_bias=0.9):
        self.robot = robot
        self.tree = []
        self.gamma_rrt_star = gamma_rrt_star
        self.eta = eta
        self.max_iterations = max_iterations
        self.goal = None
        self.node_index = 0 
        self.goal_threshold = goal_threshold
        self.goal_bias = goal_bias  # Probability of sampling the goal

        # Initialize plot
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.plot_initialized = False

    def plan(self, start_pos, goal_pos):
        self.goal = goal_pos
        start_node = {'id': self.node_index, 'config': self.robot.get_joint_pos(), 'ee_pos': start_pos, 'cost': 0, 'parent_index': None}
        self.node_index += 1
        V = [start_node]
        E = []

        # # Use a priority queue to prioritize nodes closer to the goal
        # priority_queue = []
        # heapq.heappush(priority_queue, (self.distance_to_goal(start_node), start_node))

        for i in range(self.max_iterations):
            print('Iteration %d' % i)

            if random.random() < self.goal_bias:
                xrand = self.goal  # Bias towards the goal
            else:
                xrand = self.random_sample()

            xnearest_index = self.nearest_neighbor(V, xrand)
            xnearest = V[xnearest_index]
            xnew_config = self.steer(xnearest['ee_pos'], xrand)
            self.robot.reset_joint_pos(xnew_config)

            if not self.robot.in_collision():
                xnew_pos = self.robot.ee_position()
                xnew_cost = xnearest['cost'] + np.linalg.norm(xnew_pos - xnearest['ee_pos'])
                xnew = {'id': self.node_index, 'config': xnew_config, 'ee_pos': xnew_pos, 'cost': xnew_cost, 'parent_index': xnearest['id']}
                self.node_index += 1
                Xnear_indices = self.near_neighbors(V, xnew_pos)
                V.append(xnew)
                xmin = xnearest
                cmin = xnew_cost

                for xnear_index in Xnear_indices:
                    xnear = V[xnear_index]
                    new_cost = xnear['cost'] + np.linalg.norm(xnear['ee_pos'] - xnew['ee_pos'])
                    if new_cost < cmin and not self.robot.in_collision():
                        xmin = xnear
                        cmin = new_cost

                xnew['parent_index'] = xmin['id']
                E.append((xmin['id'], xnew['id']))

                # Visualize the current state of the tree
                self.visualize_tree(V, E, goal_pos)

                # Check if the goal is reached
                if self.is_goal_reached(xnew['ee_pos']):
                    print("Goal reached!")
                    self.tree = V
                    return self.reconstruct_path(xnew)

                # Rewiring the Tree
                for xnear_index in Xnear_indices:
                    xnear = V[xnear_index]
                    new_cost = xnew['cost'] + np.linalg.norm(xnew['ee_pos'] - xnear['ee_pos'])
                    if new_cost < xnear['cost'] and not self.robot.in_collision():
                        xparent_id = xnear['parent_index']
                        
                        # Remove the old edge from the parent to xnear.
                        E = [(parent, child) for parent, child in E if not (parent == xparent_id and child == xnear['id'])]
                        
                        E.append((xnew['id'], xnear['id']))
                        xnear['parent_index'] = xnew['id']
                        xnear['cost'] = new_cost

        self.tree = V
        return self.reconstruct_path()
    
    def is_goal_reached(self, ee_pos):
        return np.linalg.norm(ee_pos - self.goal) < self.goal_threshold

    def steer(self, start_pos, target_pos):
        direction = target_pos - start_pos
        distance = np.linalg.norm(direction)
        if distance <= self.eta:
            return self.robot.inverse_kinematics(target_pos)
        else:
            direction = (direction / distance) * self.eta
            new_target_pos = start_pos + direction
            return self.robot.inverse_kinematics(new_target_pos)

    def nearest_neighbor(self, V, target_pos):
        tree_positions = [node['ee_pos'] for node in V]
        tree_kdtree = KDTree(tree_positions)
        _, nearest_index = tree_kdtree.query(target_pos)
        return nearest_index

    def near_neighbors(self, V, target_pos):
        tree_positions = [node['ee_pos'] for node in V]
        tree_kdtree = KDTree(tree_positions)
        card_V = len(V)
        dimension = len(target_pos)
        radius = min(self.gamma_rrt_star * (np.log(card_V) / card_V) ** (1 / dimension), self.eta)
        indices = tree_kdtree.query_ball_point(target_pos, radius)
        return indices

    def random_sample(self):
        x = np.random.uniform(-1, 1)
        y = np.random.uniform(-1, 1)
        z = np.random.uniform(0, 1)  # Added z-axis for 3D space
        return np.array([x, y, z])

    def distance_to_goal(self, node):
        return np.linalg.norm(node['ee_pos'] - self.goal)

    def reconstruct_path(self, goal_node=None):
        if not self.tree:
            return []  # Return an empty list if the tree is empty

        path = []
        current_node = goal_node if goal_node else self.tree[-1]
        
        while current_node is not None:
            path.insert(0, current_node)
            parent_index = current_node['parent_index']
            current_node = next((node for node in self.tree if node['id'] == parent_index), None)

        self.visualize_tree(self.tree, self.tree_edges(self.tree), self.goal, path)
        return path

    def tree_edges(self, V):
        E = []
        for node in V:
            if node['parent_index'] is not None:
                parent = next((n for n in V if n['id'] == node['parent_index']), None)
                if parent:
                    E.append((parent['id'], node['id']))
        return E

    def visualize_tree(self, V, E, goal_pos, path=None):
        self.ax.clear()
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)

        for (parent_id, child_id) in E:
            parent = next(node for node in V if node['id'] == parent_id)
            child = next(node for node in V if node['id'] == child_id)
            self.ax.plot([parent['ee_pos'][0], child['ee_pos'][0]], [parent['ee_pos'][1], child['ee_pos'][1]], 'k-')

        self.ax.plot([node['ee_pos'][0] for node in V], [node['ee_pos'][1] for node in V], 'bo')
        self.ax.plot(goal_pos[0], goal_pos[1], 'ro')  # Plot the goal position

        # Draw the final path in yellow
        if path:
            for i in range(len(path) - 1):
                self.ax.plot([path[i]['ee_pos'][0], path[i+1]['ee_pos'][0]], [path[i]['ee_pos'][1], path[i+1]['ee_pos'][1]], 'y-', linewidth=2)

        self.ax.set_title('RRT* Tree')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        plt.draw()
        plt.pause(0.001)
