from robot import Robot
from JPlusRRT import JPlusRRT
import numpy as np
import time  # For adding delays between movements
from goal import Goal

if __name__ == '__main__':
    robot = Robot(with_gui=True)
    goal_position = np.array([0.7, 0.0, 0.6])  # Example goal

    for i in range(6):
        goal = Goal(i)
        robot.set_goal(goal)
    
    planner = JPlusRRT(robot, goal_direction_probability=0.15)
    start_position = robot.get_joint_pos()
    
    path = planner.plan(start_position, goal_position)
    
    if path:
        print("Moving the robot along the found path...")
        for node in path:
            if 'config' in node:  # Ensure 'config' key exists
                joint_positions = node['config']
                print("################################")
                print(joint_positions)
                print("################################")
                robot.reset_joint_pos(joint_positions)  # Move the robot to each position in the path
                # time.sleep(.3)  # Wait a bit to see the movement
        print("Path execution completed. Press Enter to finish.")
        input() 
    else:
        print("No path found.")

    
    robot.disconnect()
