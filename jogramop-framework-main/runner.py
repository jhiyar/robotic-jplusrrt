from util import SCENARIO_IDS
from scenario import Scenario
from RRTStar import RRTStar  
import numpy as np
import time

def find_ik(robot, grasp_pose):
    """
    Attempt to find an inverse kinematics (IK) solution for the given grasp pose.
    Returns the joint configuration if successful, otherwise returns None.
    """
    ik_solution = robot.inverse_kinematics(grasp_pose)
    if ik_solution:
        return ik_solution
    else:
        return None

def run_planner(planner, robot, start_config, goal_pos):
    """
    Runs the planner and attempts to find a path to the goal position.
    Returns the path if successful, otherwise returns None.
    """
    path = planner.plan(start_config, goal_pos)
    if path:
        print(f"Path found! Path length: {len(path)}")
        return path
    else:
        return None

def execute_path(robot, path):
    """
    Executes the planned path by moving the robot along the nodes in the path.
    """
    for node in path:
        robot.move_to(node['config'])
        time.sleep(0.2)  # Simulate the robot's motion

def main():
    scenario_id = 21
    num_runs = 100
    print(f'********** SCENARIO {scenario_id:03d} **********')
    
    # Load the scenario
    s = Scenario(scenario_id)
    
    # Select the grasp poses
    s.select_n_grasps(60)
    
    # Get the robot and simulation environment
    robot, sim = s.get_robot_and_sim(with_gui=True)
    
    # Get the initial configuration of the robot's joints
    start_config = robot.arm_joints_pos()

    # Initialize planners (assuming RRTStar for now, but you can add more planners here)
    planners = [RRTStar(robot)]  # Add other planners here if needed

    # Run 100 times for each planner
    for planner in planners:
        print(f"Running planner {planner.__class__.__name__}")
        
        for run in range(num_runs):
            print(f"Run {run + 1} / {num_runs}")
            
            grasp_found = False
            for grasp_pose in s.grasp_poses:
                # Try to find an IK solution for the current grasp pose
                ik_solution = find_ik(robot, grasp_pose)
                
                if ik_solution is not None:
                    print(f"Grasp found at pose: {grasp_pose[:3, 3]}")
                    grasp_found = True
                    break
                else:
                    print("No valid IK found for this grasp, trying next grasp.")
            
            if grasp_found:
                # Define the goal position from the selected grasp pose
                goal_pos = grasp_pose[:3, 3]
                
                # Run the planner
                path = run_planner(planner, robot, start_config, goal_pos)
                
                if path:
                    # Execute the path if one is found
                    execute_path(robot, path)
                else:
                    print("No path found.")
            else:
                print("No valid grasp found, skipping to the next run.")
    
    print("All runs completed.")

if __name__ == '__main__':
    main()
