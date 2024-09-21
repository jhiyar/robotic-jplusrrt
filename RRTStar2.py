import numpy as np
import pybullet as p
import pybullet_data
import random
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import heapq

class RRTStar:
    def __init__(self, robot, gamma_rrt_star=1.0, eta=0.03, max_iterations=10000, goal_threshold=0.08, goal_direction_probability=0.5, with_visualization=False):
        self.robot = robot
        self.tree = []
        self.gamma_rrt_star = gamma_rrt_star # radius
        self.eta = eta # step size
        self.max_iterations = max_iterations
        self.goal = None
        self.node_index = 0 
        self.goal_threshold = goal_threshold
        self.goal_direction_probability = goal_direction_probability
        self.with_visualization = with_visualization

        if with_visualization:
            plt.ion()
            self.fig, self.ax = plt.subplots()
            self.ax.set_xlim(-1, 1)
            self.ax.set_ylim(-1, 1)
            self.plot_initialized = False

    def plan(self, start_pos, goal_pos):
        self.goal = goal_pos
        goal_config = self.get_goal_config(goal_pos)
        start_node = {'id': self.node_index, 'config': self.robot.get_joint_pos(), 'ee_pos': start_pos, 'cost': 0, 'parent_index': None}
        self.node_index += 1
        V = [start_node]
        E = []

        for i in range(self.max_iterations):
            print('Iteration %d' % i)

            if random.random() < self.goal_direction_probability:
                xrand = goal_config
            else:
                xrand = self.random_sample()

            xnearest_index = self.nearest_neighbor(V, xrand)
            xnearest = V[xnearest_index]
            xnew_config = self.steer(xnearest['config'], xrand)
            
            self.robot.reset_joint_pos(xnew_config)

            #Rewiring the tree
            if not self.robot.in_collision():
                xnew_pos = self.robot.ee_position()
                xnew_cost = xnearest['cost'] + np.linalg.norm(xnew_config - xnearest['config'])
                xnew = {'id': self.node_index, 'config': xnew_config, 'ee_pos': xnew_pos, 'cost': xnew_cost, 'parent_index': xnearest['id']}
                self.node_index += 1
                Xnear_indices = self.near_neighbors(V, xnew_config)
                V.append(xnew)
                xmin = xnearest
                cmin = xnew_cost

                for xnear_index in Xnear_indices:
                    xnear = V[xnear_index]
                    new_cost = xnear['cost'] + np.linalg.norm(xnear['config'] - xnew['config'])
                    if new_cost < cmin and not self.robot.in_collision():
                        xmin = xnear
                        cmin = new_cost

                xnew['parent_index'] = xmin['id']
                E.append((xmin['id'], xnew['id']))

                if self.with_visualization:
                    self.visualize_tree(V, E, goal_pos)

                if self.is_goal_reached(xnew['config']):
                    print("Goal reached!")
                    self.tree = V
                    return self.reconstruct_path(xnew)

                # Finding the Minimum Cost Path to the New Node
                for xnear_index in Xnear_indices:
                    xnear = V[xnear_index]
                    new_cost = xnew['cost'] + np.linalg.norm(xnew['config'] - xnear['config'])
                    if new_cost < xnear['cost'] and not self.robot.in_collision():
                        xparent_id = xnear['parent_index']
                        E = [(parent, child) for parent, child in E if not (parent == xparent_id and child == xnear['id'])]
                        E.append((xnew['id'], xnear['id']))
                        xnear['parent_index'] = xnew['id']
                        xnear['cost'] = new_cost

        self.tree = V
        return None

    
    def get_goal_config(self,goal_pos):
        for _ in range(10000): 
            # Generate a random orientation
            random_orientation = [random.uniform(-np.pi, np.pi), 0, 0]
            goal_config = self.robot.inverse_kinematics(goal_pos, random_orientation)
            self.robot.reset_joint_pos(goal_config)
            if not self.robot.in_collision():
                print("Found collision-free goal configuration:", goal_config)
                return goal_config
        
        raise RuntimeError("Failed to find a collision-free initial configuration for the goal tree.")


    
    def is_goal_reached(self, config):
        self.robot.reset_joint_pos(config)
        ee_pos = self.robot.ee_position()
        return np.linalg.norm(ee_pos - self.goal) < self.goal_threshold

    def steer(self, start_config, target_config):
        # print("Steering : %s" % start_config , " to %s" % target_config)
        direction = target_config - start_config

        # check the direction and compare with eta, 

        distance = np.linalg.norm(direction)
        if distance <= self.eta:
            return target_config
        else:
            direction = (direction / distance) * self.eta
            new_config = start_config + direction
            return new_config

    def nearest_neighbor(self, V, target_config):
        tree_configs = [node['config'] for node in V]
        tree_kdtree = KDTree(tree_configs)
        _, nearest_index = tree_kdtree.query(target_config)
        return nearest_index

    def near_neighbors(self, V, target_config):
        tree_configs = [node['config'] for node in V]
        
        # Tip: check the time it takes to create KDTree every time, can we create it once ? 
        tree_kdtree = KDTree(tree_configs)
        
        card_V = len(V)
        dimension = len(target_config)
        radius = min(self.gamma_rrt_star * (np.log(card_V) / card_V) ** (1 / dimension), self.eta)
        indices = tree_kdtree.query_ball_point(target_config, radius)
        return indices

    def random_sample(self):
        lower_limits, upper_limits = self.robot.joint_limits()
        return np.random.uniform(lower_limits, upper_limits)

    def distance_to_goal(self, node):
        self.robot.reset_joint_pos(node['config'])
        ee_pos = self.robot.ee_position()
        return np.linalg.norm(ee_pos - self.goal)

    def reconstruct_path(self, goal_node=None):
        if not self.tree:
            return []

        path = []
        current_node = goal_node if goal_node else self.tree[-1]
        
        while current_node is not None:
            path.insert(0, current_node)
            parent_index = current_node['parent_index']
            current_node = next((node for node in self.tree if node['id'] == parent_index), None)

        if self.with_visualization:
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
        self.ax.plot(goal_pos[0], goal_pos[1], 'ro')

        if path:
            for i in range(len(path) - 1):
                self.ax.plot([path[i]['ee_pos'][0], path[i + 1]['ee_pos'][0]],
                             [path[i]['ee_pos'][1], path[i + 1]['ee_pos'][1]], 'y-', linewidth=2)

        self.ax.set_title('RRT* Tree')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        plt.draw()
        plt.pause(0.001)
