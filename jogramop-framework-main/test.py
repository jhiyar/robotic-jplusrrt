from util import SCENARIO_IDS
from scenario import Scenario
from RRTStar import RRTStar  
import numpy as np
import time
import random

def interpolate_path(path, step_size=0.05):
    """Interpolate between path nodes to generate finer steps."""
    interpolated_path = []
    for i in range(len(path) - 1):
        start = path[i]['config']
        end = path[i + 1]['config']
        distance = np.linalg.norm(end - start)
        num_steps = int(np.ceil(distance / step_size))
        
        for j in range(num_steps):
            interp_config = start + (end - start) * (j / num_steps)
            interpolated_path.append(interp_config)
    
    interpolated_path.append(path[-1]['config'])  # Ensure the last point is added
    return interpolated_path

def main():
    scenario_id = 21
    print(f'********** SCENARIO {scenario_id:03d} **********')

  
    
    # Load the scenario
    s = Scenario(scenario_id)
    
    # Select the grasp poses
    s.select_n_grasps(60)
    
    # Get the robot and simulation environment
    robot, sim = s.get_robot_and_sim(with_gui=False)
    
    # Get the initial configuration of the robot's joints
    start_config = robot.arm_joints_pos()
    
    # Define the goal position (using the first grasp pose)
    goal_pos = s.grasp_poses[0][:3, 3]
    
    # Create the RRTStar planner
    planner = RRTStar(robot, goal_bias=0.1, with_visualization=True)
    
    # Run the planner to find a path
    print("Running RRT* planner...")
    path = planner.plan(start_config, goal_pos)

    robot, sim = s.get_robot_and_sim(with_gui=True)
    
    # If a path is found, execute it
    if path:
        print("Path found! Executing path ... path length:", len(path))
        
        # Interpolate the path to get smaller steps
        interpolated_path = interpolate_path(path, step_size=0.5)
        
        for node in path:
            # Move the robot to each configuration along the path
            robot.move_to(node['config'])
            time.sleep(0.2) # Delay to simulate the robot's motion
        
        print("Path execution complete.")
    else:
        print("No path found.")
    
    # Wait for the user to continue/exit
    input('Press Enter to exit')

if __name__ == '__main__':
    main()
