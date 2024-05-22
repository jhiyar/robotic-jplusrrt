import numpy as np
import pybullet as p
import pybullet_data
import random


#Bidirectional Inverse Kinematics RRT
class BIKRRT:
    def __init__(self, robot, goal_direction_probability=0.5):
        self.robot = robot
        self.start_tree = []
        self.goal_tree = []
        self.goal_direction_probability = goal_direction_probability
        self.goal = None  # goal is a numpy array [x, y, z] of the goal position

    def plan(self, start_pos, goal_pos):
        self.goal = goal_pos

        # Initialize both trees with start and goal configurations
        self.start_tree.append({'config': self.robot.get_joint_pos(), 'ee_pos': start_pos, 'parent_index': None})
        self.goal_tree.append({'config': self.robot.inverse_kinematics(goal_pos), 'ee_pos': goal_pos, 'parent_index': None})

        while True:
            # Grow the start tree
            if random.random() < self.goal_direction_probability:
                success = self.extend_tree(self.start_tree, self.goal_tree[-1]['ee_pos'])
            else:
                success = self.random_sample(self.start_tree) is not None

            if success and self.check_connection():
                break

            # Grow the goal tree
            if random.random() < self.goal_direction_probability:
                success = self.extend_tree(self.goal_tree, self.start_tree[-1]['ee_pos'])
            else:
                success = self.random_sample(self.goal_tree) is not None

            if success and self.check_connection():
                break

        return self.reconstruct_path()

    def extend_tree(self, tree, target_pos, step_size=0.05):
        nearest_index = self.nearest_neighbor(tree, target_pos)
        nearest_node = tree[nearest_index]

        new_config = self.step_towards(nearest_node['config'], target_pos, step_size)
        self.robot.reset_joint_pos(new_config)

        if not self.robot.in_collision():
            new_ee_pos = self.robot.ee_position()
            node = {'config': new_config, 'ee_pos': new_ee_pos, 'parent_index': nearest_index}
            tree.append(node)
            return True

        return False

    def nearest_neighbor(self, tree, target_ee_pos):
        """Find the nearest node in the tree to q_rand."""
        closest_distance = np.inf
        closest_index = None
        for i, node in enumerate(tree):
            distance = np.linalg.norm(node['ee_pos'] - target_ee_pos)
            if distance < closest_distance:
                closest_distance = distance
                closest_index = i
        return closest_index

    def step_towards(self, q_near, target_pos, step_size=0.05):
        """Take a small step from q_near towards target_pos."""
        direction = target_pos - self.robot.ee_position()
        distance = np.linalg.norm(direction)  # Euclidean distance
        if distance <= step_size:
            return self.robot.inverse_kinematics(target_pos)
        else:
            direction = (direction / distance) * step_size
            target_pos = self.robot.ee_position() + direction
            return self.robot.inverse_kinematics(target_pos)

    def random_sample(self, tree, attempts=100):
        for _ in range(attempts):
            # Generate a random end-effector position
            x = np.random.uniform(-1, 1)
            y = np.random.uniform(-1, 1)
            z = np.random.uniform(0, 1)  # Added z-axis for 3D space
            target_ee_pos = np.array([x, y, z])

            try:
                q_rand = self.robot.inverse_kinematics(target_ee_pos)
            except:
                continue

            self.robot.reset_joint_pos(q_rand)

            if not self.robot.in_collision():
                ee_pos = self.robot.ee_position()

                # Find the nearest node in the tree to the new end-effector position
                nearest_index = self.nearest_neighbor(tree, ee_pos)
                q_near = tree[nearest_index]['config']
                q_new = self.step_towards(q_near, target_ee_pos)
                new_ee_pos = self.robot.ee_position()  # Get the new end-effector position after moving

                if q_new is not None and not self.robot.in_collision():
                    node = {'config': q_new, 'ee_pos': new_ee_pos, 'parent_index': nearest_index}
                    tree.append(node)
                    return True  # Indicate success

        return False

    def check_connection(self):
        """Check if the start and goal trees are connected."""
        for start_node in self.start_tree:
            for goal_node in self.goal_tree:
                if np.linalg.norm(start_node['ee_pos'] - goal_node['ee_pos']) < 0.05:
                    self.connection = (start_node, goal_node)
                    return True
        return False

    def reconstruct_path(self):
        """Reconstruct the path from start to goal."""
        if not self.start_tree or not self.goal_tree:
            return []  # Return an empty list if either tree is empty

        path = []

        # Traverse from the connection point back to the start
        node = self.connection[0]
        while node is not None:
            path.insert(0, node)
            parent_index = node['parent_index']
            node = self.start_tree[parent_index] if parent_index is not None else None

        # Traverse from the connection point to the goal
        node = self.connection[1]
        while node is not None:
            path.append(node)
            parent_index = node['parent_index']
            node = self.goal_tree[parent_index] if parent_index is not None else None

        return path