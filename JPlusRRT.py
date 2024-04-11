import numpy as np
import random 

class JPlusRRT:
    def __init__(self, robot, goal_direction_probability=0.05):
        self.robot = robot
        self.tree = []
        self.goal_direction_probability = goal_direction_probability
        self.goal = None #goal is a numpy array [x, y, z] of the goal position

    # def plan(self, start_pos, goal_pos):
    #     self.goal = goal_pos
    #     # Initialize the tree with the start position. Assume start_pos is the robot's current configuration.
    #     start_node = {'config': start_pos, 'parent_index': None}
    #     self.tree.append(start_node)

    #     while not self.is_goal_reached():
    #         # Decide whether to sample towards the goal or randomly in the configuration space
    #         if random.random() < self.goal_direction_probability:
    #             # Directly use the goal as the target, and convert it to configuration space using inverse kinematics
    #             q_rand = self.robot.inverse_kinematics(self.goal[:3])  # Assuming self.goal contains position only
    #         else:
    #             # Sample a random configuration directly in the configuration space
    #             q_rand = self.random_sample()


    #         # Find the nearest node in the tree to the sampled configuration
    #         nearest_index = self.nearest_neighbor(q_rand)
    #         q_near = self.tree[nearest_index]['config']

    #         # Take a step towards q_rand from q_near to get q_new
    #         q_new = self.step_towards(q_near, q_rand)

    #         # Temporarily set the robot to the new configuration to check for collisions
    #         self.robot.reset_joint_pos(q_new)
    #         if not self.robot.in_collision():
    #             # If no collision, add q_new to the tree
    #             node = {'config': q_new, 'parent_index': nearest_index}
    #             self.tree.append(node)
    #         else:
    #             print("Collision detected, searching for another point...")
    #             continue

    #     return self.reconstruct_path()

    def plan(self, start_pos, goal_pos):
        self.goal = goal_pos
        # Start position is now assumed to be set directly in the robot, 
        # so we start planning from the robot's current state

        while not self.is_goal_reached():
            if random.random() < self.goal_direction_probability:
                # Try to move towards the goal and update the robot's state
                success = self.move_towards_goal()
            else:
                # Sample a new position and update the robot's state
                success = self.random_sample() is not None

            # After updating the robot's state, check for collisions without passing new_pos
            if not success or self.robot.in_collision():
                print("Collision detected, searching for another point...")
                continue
            
        return self.reconstruct_path()


    def nearest_neighbor(self, q_rand):
        """Find the nearest node in the tree to q_rand."""
        closest_distance = np.inf
        closest_index = None
        for i, node in enumerate(self.tree):
            distance = np.linalg.norm(node['config'] - q_rand)
            if distance < closest_distance:
                closest_distance = distance
                closest_index = i
        return closest_index
    
    def step_towards(self, q_near, q_rand, step_size=0.05):
        """Take a small step from q_near towards q_rand."""
        direction = q_rand - q_near
        distance = np.linalg.norm(direction) # Euclidean distance 
        if distance <= step_size:
            return q_rand
        else:
            q_new = q_near + (direction / distance) * step_size #  a new position that is exactly one step size closer towards q_rand, without exceeding the step size limit.
            return q_new

    
    def random_sample(self, attempts=100):
        lower_limits, upper_limits = self.robot.joint_limits()
        for _ in range(attempts):
            q_rand = np.random.uniform(lower_limits, upper_limits)
            if self.tree:
                nearest_index = self.nearest_neighbor(q_rand)
                q_near = self.tree[nearest_index]['config']
                q_new = self.step_towards(q_near, q_rand)
                self.robot.reset_joint_pos(q_new)
                if not self.robot.in_collision():
                    node = {'config': q_new, 'parent_index': nearest_index}
                    self.tree.append(node)
                    return True  # Indicate success
            else:
                # If the tree is empty, initialize it with the start position
                self.robot.reset_joint_pos(q_rand)
                if not self.robot.in_collision():
                    node = {'config': q_rand, 'parent_index': None}
                    self.tree.append(node)
                    return True
        return False

    
    def move_towards_goal(self):
        current_ee_pos = self.robot.ee_position()
        goal_pos = self.goal

        direction_vector = goal_pos - current_ee_pos
        direction_vector /= np.linalg.norm(direction_vector)

        step_size = 0.01
        desired_ee_velocity = direction_vector * step_size

        J = self.robot.get_jacobian()
        J_pseudo_inverse = np.linalg.pinv(J)

        joint_velocities = J_pseudo_inverse.dot(desired_ee_velocity)
        current_joint_positions = self.robot.get_joint_pos()
        new_joint_positions = current_joint_positions + joint_velocities

        lower_limits, upper_limits = self.robot.joint_limits()
        new_joint_positions = np.clip(new_joint_positions, lower_limits, upper_limits)

        # Temporarily set the robot to the new positions to check for collisions
        self.robot.reset_joint_pos(new_joint_positions)
        if self.robot.in_collision():
            return False  # Move results in a collision, revert changes
        else:
            # Successful move towards goal without collision, update the tree
            parent_index = len(self.tree) - 1 if self.tree else None
            node = {'config': new_joint_positions, 'parent_index': parent_index}
            self.tree.append(node)
            return True

    def is_goal_reached(self):
        """
        Checks if the current end effector position is sufficiently close to the goal.
        
        Returns:
            bool: True if the end effector is close to the goal, False otherwise.
        """
        # Get the current position of the end effector
        current_ee_pos = self.robot.ee_position()
        
        # Assuming self.goal is a numpy array [x, y, z] representing the goal position
        goal_pos = self.goal
        
        # Calculate the Euclidean distance between the current end effector position and the goal
        distance_to_goal = np.linalg.norm(current_ee_pos - goal_pos)
        
        # Define a threshold for how close the end effector needs to be to the goal to consider it reached
        threshold = 0.05  # Meters
        
        # Check if the distance to the goal is less than or equal to the threshold
        if distance_to_goal <= threshold:
            return True  # The goal is considered reached
        else:
            return False  # The goal is not reached

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
            # Prepend the configuration to the path
            path.insert(0, current_node)
            # Move to the parent node
            parent_index = current_node['parent_index']
            current_node = self.tree[parent_index] if parent_index is not None else None

        return path