import numpy as np
from visualization import show_scenario
from scenario import Scenario


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


def main():
    s = Scenario(11)
    s.select_n_grasps(10)
    robot, sim = s.get_robot_and_sim(with_gui=True)  # Enable GUI for visualization

    # Try multiple grasps to find a reachable goal
    for i in range(len(s.grasp_poses)):
        grasp_pose = s.grasp_poses[i]
        pos = grasp_pose[:3, 3]
        orn_matrix = grasp_pose[:3, :3]  # Extract orientation matrix

        # Convert orientation matrix to quaternion using the helper function
        orn_quat = matrix_to_quaternion(orn_matrix)

        # Print position and orientation for debugging
        print(f"Trying grasp {i + 1}/{len(s.grasp_poses)}")
        print("Goal Position:", pos)
        print("Goal Orientation (Quaternion):", orn_quat)

        # Check if the quaternion has NaN values
        if np.any(np.isnan(orn_quat)):
            print("Invalid orientation quaternion. Skipping this grasp.")
            continue

        # Get IK solution for the goal position
        goal_joint_config = robot.inverse_kinematics(pos, orn_quat)

        if goal_joint_config is None:
            print("No IK solution found for this goal position.")
            continue

        # If a valid IK solution is found, move the robot and exit the loop
        print("IK solution found. Moving the robot.")
        robot.move_to(goal_joint_config)
        input()
        # Step the simulation to visualize the movement
        # for _ in range(100):
        #     sim.step()  # Advance the simulation multiple times to visualize the movement

        break  # Exit the loop if a valid solution is found
    else:
        print("No valid IK solution found for any of the grasps.")

    # Visualize the scenario
    # show_scenario(s)
    # input()  # Wait for user input to close the visualization


if __name__ == "__main__":
    main()

