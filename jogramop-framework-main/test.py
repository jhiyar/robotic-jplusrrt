import numpy as np
import time

def move_to_goal(robot, start_config, goal_pos, step_size=0.1):
    """
    This function attempts to move the robot's end effector to the goal position using the Jacobian.
    """
    robot.reset_arm_joints(start_config)
    
    for i in range(100):  # Attempt to move for 100 iterations
        full_pose = robot.end_effector_pose()
        current_ee_pos = full_pose[:3, 3]

        # Calculate the direction vector and normalize it
        direction_vector = goal_pos - current_ee_pos
        distance_to_goal = np.linalg.norm(direction_vector)
        
        if distance_to_goal < 0.05:  # Goal threshold
            print("Goal reached!")
            break
        
        direction_vector /= np.linalg.norm(direction_vector)  # Normalize direction vector

        # Scale the desired velocity by step size
        desired_ee_velocity = direction_vector * step_size
        J = robot.get_jacobian()

        # Check if Jacobian is well-conditioned
        condition_number = np.linalg.cond(J)
        if condition_number > 1e12:
            print(f"Warning: Jacobian is poorly conditioned (condition number = {condition_number:.2e})")
            break

        J_pseudo_inverse = np.linalg.pinv(J)
        joint_velocities = J_pseudo_inverse @ desired_ee_velocity

        # Calculate new joint positions
        new_joint_positions = robot.arm_joints_pos() + joint_velocities[1:8]

        # Check joint limits before setting the new positions
        lower_limits, upper_limits = robot.arm_joint_limits().T
        new_joint_positions = np.clip(new_joint_positions, lower_limits, upper_limits)

        robot.reset_arm_joints(new_joint_positions)
        time.sleep(0.2)

        print(f"Iteration: {i}, Distance to Goal: {distance_to_goal:.4f}, Joint Positions: {new_joint_positions}, EE Position: {current_ee_pos}")

    return robot.arm_joints_pos()
if __name__ == '__main__':
    from scenario import Scenario

    # Initialize the robot object
    scenario_id = 12
    print(f'********** SCENARIO {scenario_id:03d} **********')
    s = Scenario(scenario_id)
    
    s.select_n_grasps(60)
    
    robot, sim = s.get_robot_and_sim(with_gui=True)

    goal_pos = s.grasp_poses[0][:3, 3]
    
    start_config = robot.arm_joints_pos()  # Get the current configuration

    # Execute the move_to_goal function
    final_joint_positions = move_to_goal(robot, start_config, goal_pos)

    # Print the final joint positions after attempting to reach the goal
    print("Final joint positions:", final_joint_positions)

    input()
