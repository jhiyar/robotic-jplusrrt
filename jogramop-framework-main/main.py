from util import SCENARIO_IDS
from scenario import Scenario
from RRTStar import RRTStar  
from JPlusRRT import JPlusRRT
import numpy as np
import time

def run_rrt_star_100_times(planner, start_config, goal_pos, num_trials=100):
    successes = 0
    total_planning_time = 0
    
    for trial in range(num_trials):
        print(f"Running trial {trial + 1}/{num_trials}")
        
        # Start the timer
        start_time = time.time()
        
        # Run the planner
        path = planner.plan(start_config, goal_pos)
        
        # End the timer and calculate the planning time
        end_time = time.time()
        planning_time = end_time - start_time
        total_planning_time += planning_time
        
        # Check if the path was found
        if path:
            successes += 1
        print(f"Trial {trial + 1}: {'Success' if path else 'Failure'} | Planning Time: {planning_time:.4f} seconds")

    # Calculate success rate and average planning time
    success_rate = (successes / num_trials) * 100  # Success rate in percentage
    avg_planning_time = total_planning_time / num_trials  # Average planning time

    return success_rate, avg_planning_time


def main():
    scenario_id = 11
    print(f'********** SCENARIO {scenario_id:03d} **********')
    s = Scenario(scenario_id)
    
    s.select_n_grasps(60)
    
    robot, sim = s.get_robot_and_sim(with_gui=False)
    
    start_config = robot.arm_joints_pos()  # Get the start configuration of robot joints
    
    goal_pos = s.grasp_poses[0][:3, 3]  # Use the first grasp pose
    
    # Create the RRTStar planner
    planner = RRTStar(robot, goal_bias=0.5, with_visualization=False)
    
    # Run the RRT* algorithm 100 times and collect success rate and average planning time
    success_rate, avg_planning_time = run_rrt_star_100_times(planner, start_config, goal_pos, num_trials=100)
    
    # Print the results
    print(f"Success Rate: {success_rate:.2f}%")
    print(f"Average Planning Time: {avg_planning_time:.4f} seconds")
    
    input('Enter to continue')

if __name__ == '__main__':
    main()
