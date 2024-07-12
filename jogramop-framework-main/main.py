import numpy as np
from visualization import show_scenario
from scenario import Scenario
from JPlusRRT import JPlusRRT

def matrix_to_quaternion(matrix):
    """
    Convert a rotation matrix to a quaternion.
    """
    trace = np.trace(matrix)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (matrix[2, 1] - matrix[1, 2]) * s
        y = (matrix[0, 2] - matrix[2, 0]) * s
        z = (matrix[1, 0] - matrix[0, 1]) * s
    else:
        if matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
            s = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
            w = (matrix[2, 1] - matrix[1, 2]) / s
            x = 0.25 * s
            y = (matrix[0, 1] + matrix[1, 0]) / s
            z = (matrix[0, 2] + matrix[2, 0]) / s
        elif matrix[1, 1] > matrix[2, 2]:
            s = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
            w = (matrix[0, 2] - matrix[2, 0]) / s
            x = (matrix[0, 1] + matrix[1, 0]) / s
            y = 0.25 * s
            z = (matrix[1, 2] + matrix[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
            w = (matrix[1, 0] - matrix[0, 1]) / s
            x = (matrix[0, 2] + matrix[2, 0]) / s
            y = (matrix[1, 2] + matrix[2, 1]) / s
            z = 0.25 * s
    return np.array([x, y, z, w])

def move_robot_along_path(robot, sim, path):
    for config in path:
        robot.move_to(config)
        sim.step()  # Advance the simulation

def main():
    s = Scenario(11)
    s.select_n_grasps(30)
    robot, sim = s.get_robot_and_sim(with_gui=True)  # Enable GUI for visualization

    # Extract the goal pose from the selected grasp
    grasp_pose = s.grasp_poses[0]
    pos = grasp_pose[:3, 3]
    orn_matrix = grasp_pose[:3, :3]  # Extract orientation matrix

    # Convert orientation matrix to quaternion using the helper function
    orn_quat = matrix_to_quaternion(orn_matrix)

    # Print position and orientation for debugging
    print("Goal Position:", pos)
    print("Goal Orientation (Quaternion):", orn_quat)

    # Check if the quaternion has NaN values
    if np.any(np.isnan(orn_quat)):
        print("Invalid orientation quaternion. Please check the input rotation matrix.")
        return
    

    # Get IK solution for the goal position
    goal_joint_config = robot.inverse_kinematics(pos, orn_quat)
    input()

    if goal_joint_config is None:
        print("No IK solution found for the goal position.")
        return

    # Get the current joint configuration as the start configuration
    start_joint_config = robot.arm_joints_pos()


    # Plan the path using JPlusRRT
    # planner = JPlusRRT(robot, start_joint_config, goal_joint_config)
    # path = planner.plan()

    # print("path" , path)


    # if path is None:
    #     print("No valid path found to the goal.")
    #     return

    # # Move the robot along the planned path
    # move_robot_along_path(robot, sim, path)
    
    # Visualize the scenario
    # show_scenario(s)
    # input()  # Wait for user input to close the visualization

if __name__ == "__main__":
    main()

