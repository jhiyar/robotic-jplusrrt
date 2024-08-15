from util import SCENARIO_IDS
from scenario import Scenario
from RRTStar import RRTStar  
from JPlusRRT import JPlusRRT


# After finding a solution, consider the orientation of the grasp for grasp planning.

def main():
    scenario_id = 12
    print(f'********** SCENARIO {scenario_id:03d} **********')
    s = Scenario(scenario_id)
    
    s.select_n_grasps(60)
    
    robot, sim = s.get_robot_and_sim(with_gui=True)
    
    start_pos = robot.end_effector_pose()
    
    # joint_configuration = robot.arm_joints_pos # start configuration of robot joints

    goal_pos = s.grasp_poses[0][:3, 3]  # use the first grasp pose
    
    # planner = RRTStar(robot, with_visualization=True)
    planner = JPlusRRT(robot,goal_direction_probability=1, with_visualization=True)
    path = planner.plan(start_pos, goal_pos)
    
    if path:
        print("Path found!")
        print(len(path))
        for node in path:
            robot.move_to(node['config'])
    else:
        print("No path found.")
    
    input('Enter to continue')

    
    # goal_joint_config = robot.inverse_kinematics(goal_pos)
    # robot.move_to(goal_joint_config)

if __name__ == '__main__':
    main()
