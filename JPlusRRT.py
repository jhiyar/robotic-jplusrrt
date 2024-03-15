import time
import random
import numpy as np

class JPlusRRT:
    def __init__(self, robot, goal_direction_probability=0.2, timeout=10):
        self.robot = robot
        self.tree = []  # Tree initialized with start configuration
        self.goal_direction_probability = goal_direction_probability
        self.timeout = timeout  # Timeout in seconds
        self.goal = None  # Goal position

    def add_configuration(self, qstart):
        # Add the start configuration to the RRT tree
        self.tree.append({'config': qstart, 'parent_index': None})

    def extend_randomly(self):
        # Step 1: Generate a random configuration within the robot's joint limits.
        lower_limits, upper_limits = self.robot.joint_limits()
        random_config = np.random.uniform(lower_limits, upper_limits)

        # Step 2: Find the nearest node in the tree to the random configuration
        nearest_node_index = self.find_nearest_node(random_config)
        nearest_node = self.tree[nearest_node_index]['config']

        # Step 3: Attempt to extend towards the random configuration
        # Calculate the direction of extension in joint space
        direction = random_config - nearest_node
        norm = np.linalg.norm(direction)
        if norm == 0:
            return False  # Random config is essentially the same as nearest node

        # Limit the extension to a maximum step size in joint space
        max_step_size = 0.1  # Adjust based on your robot's characteristics
        if norm > max_step_size:
            direction = (direction / norm) * max_step_size

        new_config = nearest_node + direction

        # Check if the new configuration is within joint limits and not in collision
        if self.in_collision(new_config) or not self.in_joint_limits(new_config):
            return False  # Extension is not feasible

        # If feasible, add the new configuration to the tree
        self.tree.append({'config': new_config, 'parent_index': nearest_node_index})
        return True

    def in_collision(self, config):
        # Temporarily apply the configuration to the robot and check for collisions
        self.robot.reset_joint_pos(config)
        return self.robot.in_collision()

    def in_joint_limits(self, config):
        lower_limits, upper_limits = self.robot.joint_limits()
        return np.all(config >= lower_limits) and np.all(config <= upper_limits)

    def find_nearest_node(self, random_config):
        # Calculate the nearest node in the tree to the given configuration
        # This could be based on the Euclidean distance in joint space
        distances = [np.linalg.norm(node['config'] - random_config) for node in self.tree]
        nearest_node_index = np.argmin(distances)
        return nearest_node_index
    
    def GetRandomGrasp(self, gc):
        """
        Selects a random grasp from a predefined set or generates a grasp configuration.
        
        :param gc: Goal configuration, potentially used to influence grasp selection.
        :return: A randomly selected or generated grasp configuration.
        """
        # Example: Assume grasps are defined by end effector positions and orientations
        # For simplicity, this returns a fixed grasp or could randomly select from a predefined list
        
        # Placeholder for an actual grasp selection logic based on gc
        random_grasp = {'position': [0.7, 0.0, 0.6], 'orientation': [0, 0, 0, 1]}  # Example grasp
        
        return random_grasp
    
    def ComputeTargetPose(self, grasp):
        target_pose = grasp['position']  # Directly use the position part
        return target_pose  # Make sure this is a list or a NumPy array

    def BuildSolutionPath(self, goal_node_index):
        path = []
        current_index = goal_node_index

        while current_index is not None:
            node = self.tree[current_index]  # Retrieve the node using its index
            # Append the node's configuration as a dictionary to the path
            path.append({'config': node['config']})
            current_index = node['parent_index']  # Move to the parent node's index
            
            if isinstance(current_index, np.integer):  # Ensure index is a Python integer
                current_index = int(current_index)

        return path[::-1]






    def extend_to_goal(self, pobj, gc):
        # Placeholder for GetRandomGrasp and ComputeTargetPose methods
        # Assuming these can be somehow determined or are provided externally
        grasp = self.GetRandomGrasp(gc)  # Needs implementation
        ptarget = self.ComputeTargetPose(grasp)  # Needs implementation

        # Assuming qstart is the current robot configuration
        qstart = self.robot.get_joint_pos()
        qnear = qstart
        reached = False

        while not reached:
            pnear = self.robot.ee_position()  # Current end effector position
            delta_p = ptarget - pnear  # Vector towards the target
            J = self.robot.get_jacobian()  # Current Jacobian
            J_pinv = np.linalg.pinv(J)  # Pseudoinverse of the Jacobian

            # Limiting the step size in Cartesian space to a maximum value
            max_step_size = 0.05  # Define a suitable step size
            norm_delta_p = np.linalg.norm(delta_p)
            if norm_delta_p > max_step_size:
                delta_p = (delta_p / norm_delta_p) * max_step_size

            delta_q = J_pinv.dot(delta_p)  # Change in configuration space
            qnear += delta_q  # Update the configuration

            # Check for collisions and joint limits
            if self.robot.in_collision() or not self.in_joint_limits(qnear):
                return None  # Return None if in collision or outside joint limits

            self.robot.reset_joint_pos(qnear)  # Update the robot's joint positions
            
            if np.linalg.norm(delta_p) < 0.01:  # Threshold to consider as reached
                reached = True
            
            goal_node_index = len(self.tree) - 1

        return self.BuildSolutionPath(goal_node_index) 

    def prune_path(self, solution):
        pruned_path = [solution[0]]  # Always include the start configuration

        for i in range(2, len(solution)):
            # Check if you can go from the last added configuration directly to the current one without collision
            if self.direct_path_feasible(pruned_path[-1], solution[i]):
                # If the direct path is feasible, remove the last configuration because it's not needed
                pruned_path[-1] = solution[i]
            else:
                # If not, include the current configuration as a necessary waypoint
                pruned_path.append(solution[i-1])

        pruned_path.append(solution[-1])  # Always include the goal configuration
        return pruned_path

    def direct_path_feasible(self, start_config, end_config):
        # Assuming 'config' key contains the numeric configuration data
        start_config_array = np.array(start_config['config'])
        end_config_array = np.array(end_config['config'])
        
        # Simple linear interpolation example
        steps = 10  # Number of interpolation steps
        for step in range(1, steps + 1):
            interpolated_config = start_config_array + (end_config_array - start_config_array) * step / steps
            self.robot.reset_joint_pos(interpolated_config)
            if self.robot.in_collision():
                return False  # Collision detected, direct path is not feasible
        return True  # No collision detected, direct path is feasible


    def plan(self, qstart, pobj, gc):
        self.add_configuration(qstart)
        self.goal = gc
        start_time = time.time()

        while time.time() - start_time < self.timeout:
            self.extend_randomly()

            if random.random() < self.goal_direction_probability:
                solution = self.extend_to_goal(pobj, gc)
                if solution is not None:
                    return self.prune_path(solution)

        return None  # Return None if no solution is found within the timeout
