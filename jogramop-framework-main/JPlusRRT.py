import numpy as np
import random 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class JPlusRRT:
    def __init__(self, robot, goal_direction_probability=0.5,step_size=0.2, with_visualization=False):
        self.robot = robot
        self.tree = []
        self.goal_direction_probability = goal_direction_probability
        self.goal = None  
        self.step_size = step_size  
        self.with_visualization = with_visualization
        self.closest_node_index = None  # Track the closest node to the goal


        if with_visualization:
            # Initialize plot
            plt.ion()
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.ax.set_xlim(-1, 1)
            self.ax.set_ylim(-1, 1)
            self.ax.set_zlim(0, 1)

    def plan(self, start_config, goal_pos):
        self.goal = goal_pos

        self.start_config = start_config

        # Add the initial configuration as the first node in the tree
        full_pose = self.robot.end_effector_pose()
        start_ee_pos = full_pose[:3, 3]
        
        initial_node = {'config': start_config, 'ee_pos': start_ee_pos, 'parent_index': None}
        self.tree.append(initial_node)

        self.closest_node_index = 0

        while not self.is_goal_reached():
            if random.random() <= self.goal_direction_probability:
                # Try to move towards the goal and update the robot's state
                print("Moving towards goal")
                success = self.move_towards_goal()
            else:
                # Sample a new position and update the robot's state
                print("Sample a new position")
                success = self.random_sample() is not None

            # After updating the robot's state, check for collisions
            if not success:
                print("Collision detected, searching for another point...")
                continue

            if self.with_visualization:
                self.visualize_tree()
            
        return self.reconstruct_path()

    def nearest_neighbor(self, target_config):
        """Find the nearest node in the tree to the given configuration."""
        closest_distance = np.inf
        closest_index = None
        for i, node in enumerate(self.tree):
            distance = np.linalg.norm(node['config'] - target_config)  # Compare configurations, not ee_pos
            if distance < closest_distance:
                closest_distance = distance
                closest_index = i
        return closest_index

    def step_towards(self, q_near, q_rand, step_size=0.05):
        """Take a small step from q_near towards q_rand."""
        direction = q_rand - q_near
        distance = np.linalg.norm(direction)  # Euclidean distance 
        if distance <= step_size:
            return q_rand
        else:
            q_new = q_near + (direction / distance) * step_size  # a new position that is exactly one step size closer towards q_rand, without exceeding the step size limit.
            return q_new

    def random_sample(self, attempts=100):
        lower_limits, upper_limits = self.robot.arm_joint_limits().T
        for _ in range(attempts):
            q_rand = np.random.uniform(lower_limits, upper_limits)
            self.robot.reset_arm_joints(q_rand)

            if not self.robot.in_collision():
                nearest_index = self.nearest_neighbor(q_rand) if self.tree else None
                if nearest_index is not None:
                    q_near = self.tree[nearest_index]['config']
                    q_new = self.step_towards(q_near, q_rand)
                else:
                    q_new = q_rand

                if q_new is not None and not self.robot.in_collision():
                    full_pose = self.robot.end_effector_pose()
                    new_ee_pos = full_pose[:3, 3] 
                    node = {'config': q_new, 'ee_pos': new_ee_pos, 'parent_index': nearest_index}
                    self.tree.append(node)

                    # Update the closest node to the goal if this one is closer
                    if self.closest_node_index is None or np.linalg.norm(new_ee_pos - self.goal) < np.linalg.norm(self.tree[self.closest_node_index]['ee_pos'] - self.goal):
                        self.closest_node_index = len(self.tree) - 1

                    return True

        return False
    
    def move_towards_goal(self):
        if self.closest_node_index is None:
            print("No valid starting node")
            return False  # No valid node to start from

        # Use the configuration from the closest node to the goal
        closest_node = self.tree[self.closest_node_index]
        self.robot.reset_arm_joints(closest_node['config'])

        full_pose = self.robot.end_effector_pose()
        current_ee_pos = full_pose[:3, 3] 

        goal_pos = self.goal

        direction_vector = goal_pos - current_ee_pos
        direction_vector /= np.linalg.norm(direction_vector)

        desired_ee_velocity = direction_vector * self.step_size

        J = self.robot.get_jacobian()
        J = J[:, :-2]

        J_pseudo_inverse = np.linalg.pinv(J)
        joint_velocities = J_pseudo_inverse @ desired_ee_velocity

        # Introduce a random perturbation to help escape local minima
        perturbation = np.random.uniform(-0.01, 0.01, size=joint_velocities.shape)
        joint_velocities += perturbation

        current_joint_positions = self.robot.arm_joints_pos()
        new_joint_positions = current_joint_positions + joint_velocities

        print("Current Joint Positions:", current_joint_positions)
        print("Joint Velocities:", joint_velocities)
        print("New Joint Positions:", new_joint_positions)

        # Temporarily set the robot to the new positions to check for collisions
        self.robot.reset_arm_joints(new_joint_positions)
        if self.robot.in_collision():
            print("Collision detected, skipping this node.")
            return False  # Move results in a collision, revert changes
        else:
            # Successful move towards goal without collision, update the tree
            parent_index = self.closest_node_index
            full_pose = self.robot.end_effector_pose()
            new_ee_pos = full_pose[:3, 3]
            node = {'config': new_joint_positions, 'ee_pos': new_ee_pos, 'parent_index': parent_index}

            print("New EE Position:", new_ee_pos)
            print("New node found, adding to the tree", node)

            self.tree.append(node)

            # print(new_ee_pos , self.goal , self.tree[self.closest_node_index]['ee_pos'])

            # Update the closest node to the goal if this one is closer
            if np.linalg.norm(new_ee_pos - self.goal) < np.linalg.norm(self.tree[self.closest_node_index]['ee_pos'] - self.goal):
                self.closest_node_index = len(self.tree) - 1

            return True


    def is_goal_reached(self):
        """
        Checks if the current end effector position is sufficiently close to the goal.
        
        Returns:
            bool: True if the end effector is close to the goal, False otherwise.
        """
        full_pose = self.robot.end_effector_pose()
        current_ee_pos = full_pose[:3, 3] 
        goal_pos = self.goal
        distance_to_goal = np.linalg.norm(current_ee_pos - goal_pos)
        threshold = 0.05  # Meters

        return distance_to_goal <= threshold

    def reconstruct_path(self):
        """
        Reconstructs the path from the goal node back to the start node.
        
        Returns:
            list: The sequence of configurations forming the path from start to goal.
        """
        if not self.tree:
            return []  # Return an empty list if the tree is empty

        path = []
        # Start from the last added node which is assumed to be the goal or closest to the goal
        current_node_index = len(self.tree) - 1
        current_node = self.tree[current_node_index]

        while current_node is not None:
            path.insert(0, current_node)
            parent_index = current_node['parent_index']
            current_node = self.tree[parent_index] if parent_index is not None else None

        if self.with_visualization:
            self.visualize_tree(final=True, path=path) 
        
        return path

    def visualize_tree(self, final=False, path=None):
        if not self.with_visualization:
            return
                
        self.ax.clear()
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.set_zlim(0, 1)

        # Plot the start and goal positions
        start_ee_pos = self.tree[0]['ee_pos']  # Use the end-effector position of the first node in the tree
        self.ax.scatter(start_ee_pos[0], start_ee_pos[1], start_ee_pos[2], c='yellow', marker='o', s=100)
        self.ax.scatter(self.goal[0], self.goal[1], self.goal[2], c='green', marker='o', s=100)

        for node in self.tree:
            if node['parent_index'] is not None:
                parent_node = self.tree[node['parent_index']]
                self.ax.plot([node['ee_pos'][0], parent_node['ee_pos'][0]], 
                            [node['ee_pos'][1], parent_node['ee_pos'][1]], 
                            [node['ee_pos'][2], parent_node['ee_pos'][2]], 'b-')
                self.ax.scatter([node['ee_pos'][0]], [node['ee_pos'][1]], [node['ee_pos'][2]], c='blue', marker='o')

        if final and path:
            for i in range(len(path) - 1):
                self.ax.plot([path[i]['ee_pos'][0], path[i + 1]['ee_pos'][0]],
                            [path[i]['ee_pos'][1], path[i + 1]['ee_pos'][1]],
                            [path[i]['ee_pos'][2], path[i + 1]['ee_pos'][2]], 'orange', linewidth=2)

        plt.draw()
        plt.pause(0.01)

