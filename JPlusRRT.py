import numpy as np
import random 

class JPlusRRT:
    def __init__(self, robot, goal_direction_probability=0.05):
        self.robot = robot
        self.tree = []
        self.goal_direction_probability = goal_direction_probability
        self.goal = None #goal is a numpy array [x, y, z] of the goal position

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


    def random_sample(self, attempts=100):
        lower_limits, upper_limits = self.robot.joint_limits()
        for _ in range(attempts):
            random_config = np.random.uniform(lower_limits, upper_limits)
            self.robot.reset_joint_pos(random_config)
            if not self.robot.in_collision():
                parent_index = len(self.tree) - 1 if self.tree else None
                node = {'config': random_config, 'parent_index': parent_index}
                self.tree.append(node)
                return True  # Indicate success
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