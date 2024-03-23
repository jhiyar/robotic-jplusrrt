from robot import Robot
from JPlusRRT import JPlusRRT
import numpy as np
import time  # For adding delays between movements
from goal import Goal
# from util import move_to_joint_pos,move_to_ee_pose

if __name__ == '__main__':
    robot = Robot(with_gui=True)
    # goal_position = np.array([0.7, 0.0, 0.6])  # Example goal 0.7, 0.3, 0.6
    goal_position = np.array([0.7, 0.3, 0.6]) 
    # goal_position = np.array([0.7, 0.0, 0.2]) 



    for i in range(6):
        goal = Goal(i)
        robot.set_goal(goal)

    # move_to_ee_pose(robot.robot_id, goal_position)


    planner = JPlusRRT(robot, goal_direction_probability=0.9)
    start_position = robot.get_joint_pos()
    
    path = planner.plan(start_position, goal_position)
    
    if path:
        print("Moving the robot along the found path...")
        for node in path:
            if 'config' in node:  # Ensure 'config' key exists
                joint_positions = node['config']
                # move_to_joint_pos(robot.robot_id, joint_positions)
                robot.reset_joint_pos(joint_positions)  # Move the robot to each position in the path
                # time.sleep(.3)  # Wait a bit to see the movement
        print("Path execution completed. Press Enter to finish.")
        input() 
    else:
        print("No path found.")


    robot.disconnect()
