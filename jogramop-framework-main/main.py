from util import SCENARIO_IDS
from scenario import Scenario
from RRTStar import RRTStar  
from JPlusRRT import JPlusRRT
import numpy as np

# After finding a solution, consider the orientation of the grasp for grasp planning.
def main():
    scenario_id = 12
    print(f'********** SCENARIO {scenario_id:03d} **********')
    s = Scenario(scenario_id)
    
    s.select_n_grasps(60)
    
    robot, sim = s.get_robot_and_sim(with_gui=True)
    
    start_config = robot.arm_joints_pos()  # Get the start configuration of robot joints
    
    goal_pos = s.grasp_poses[0][:3, 3]  # use the first grasp pose
    
    planner = JPlusRRT(robot, goal_direction_probability=1, with_visualization=True)
    path = planner.plan(start_config, goal_pos)
    
    if path:
        print("Path found!")
        print(len(path))
        for node in path:
            # Adding safety checks before moving to configuration
            if np.isnan(node['config']).any() or np.isinf(node['config']).any():
                raise ValueError(f"Invalid configuration found in path: {node['config']}")
            robot.move_to(node['config'])
    else:
        print("No path found.")
    
    input('Enter to continue')

if __name__ == '__main__':
    main()