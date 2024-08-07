from util import SCENARIO_IDS
from scenario import Scenario
from RRTStar import RRTStar  
from JPlusRRT import JPlusRRT


# After finding a solution, consider the orientation of the grasp for grasp planning.

def main():
    scenario_id = 32
    print(f'********** SCENARIO {scenario_id:03d} **********')
    s = Scenario(scenario_id)
    
    s.select_n_grasps(60)
    
    robot, sim = s.get_robot_and_sim(with_gui=True)
    
    start_pos = robot.end_effector_pose()
    goal_pos = s.grasp_poses[0][:3, 3]  # use the first grasp pose
    
    rrt_star = RRTStar(robot, with_visualization=True)
    # rrt_star = JPlusRRT(robot, with_visualization=True)
    path = rrt_star.plan(start_pos, goal_pos)
    
    if path:
        print("Path found!")
        print(len(path))
        for node in path:
            robot.move_to(node['config'])
            # sim.step() 
            # robot.close()  
            # sim.step()
            # robot.open()
            # sim.step()
    else:
        print("No path found.")
    
    input('Enter to continue')

if __name__ == '__main__':
    main()
