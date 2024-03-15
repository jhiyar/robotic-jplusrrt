from robot import Robot
from JPlusRRT import JPlusRRT
import numpy as np
import time 
from goal import Goal

if __name__ == '__main__':
    # Initialize the robot, potentially with a GUI to visualize the process
    robot = Robot(with_gui=True)
    
    # Define the start configuration for the robot
    # For simplicity, we use the robot's home configuration or current configuration
    qstart = robot.get_joint_pos()

    # Initialize the JPlusRRT planner
    # Adjust the goal direction probability and timeout as needed
    planner = JPlusRRT(robot, goal_direction_probability=0.2, timeout=30)

    # Define the goal configuration (position + orientation)
    # For simplicity, this example will just reuse the goal position and assume a default orientation
    # In practice, `gc` might need to include specific orientation values or other goal state details
    goal_position = np.array([0.7, 0.0, 0.6])  # Position part of the goal configuration
    goal_orientation = np.array([0, 0, 0, 1])  # Orientation part (e.g., a quaternion)


    for i in range(6):
        goal = Goal(i)
        robot.set_goal(goal)

    # Combine position and orientation into a single structure for gc
    # This combination depends on how your JPlusRRT and Robot classes expect goal configurations
    gc = np.concatenate([goal_position, goal_orientation])

    # Then, when calling the plan method, include this gc as an argument
    path = planner.plan(qstart, goal_position, gc)  # Adjusted to include gc

    # Check if a path was found
    if path:
        print("Path found. Moving the robot...")
        # Execute the path
        print(path)
        for node in path:
            print("Moving the robot to node: " + str(node))

            config = node['config']

            robot.reset_joint_pos(config)
            # Introduce a small delay if you want to visualize the movement step by step
            time.sleep(1)
        print("Path execution completed. Press Enter to finish.")
        input() 
    else:
        print("No path found within the given timeout.")

    # Disconnect the robot (and close the simulation) when done
    robot.disconnect()
