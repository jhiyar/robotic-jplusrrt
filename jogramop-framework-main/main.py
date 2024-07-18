from util import SCENARIO_IDS
from scenario import Scenario
from visualization import show_scenario

def main():
    # for i in SCENARIO_IDS:
    print(f'********** SCENARIO {11:03d} **********')
    s = Scenario(12)
    
    # Select a random subset of grasps for better visualization
    s.select_n_grasps(30)
    
    # Visualize the scenario
    # show_scenario(s)
    
    # Get the robot and simulation environment
    robot, sim = s.get_robot_and_sim(with_gui=True)
    
    # Retrieve IK solutions for the selected grasps
    ik_solutions = s.get_ik_solutions()
    
    if len(ik_solutions) == 0:
        print('No valid IK solutions found for the selected grasps.')
        return
    
    # Execute the grasp for the first valid IK solution
    for ik_solution in ik_solutions:
        # Reset robot joints to the IK solution
        # print(len(ik_solutions))
        robot.reset_arm_joints(ik_solution)
        
        # Check for collisions and execute the grasp if there are none
        if not robot.in_collision():
            # Perform the grasp
            execute_grasp(robot, ik_solution, sim)

            # grasp_pos = robot.end_effector_pose()
            # grasp_joint_config = robot.inverse_kinematics(grasp_pos)
            # robot.move_to(grasp_joint_config)
            # robot.close()

            # grasp_pos_lift = robot.end_effector_pose()
            # grasp_pos_lift[2] += 0.1  # Move 10 cm down to the object
            # grasp_joint_config_lift = robot.inverse_kinematics(grasp_pos_lift)
            # robot.move_to(grasp_joint_config_lift)
            # # robot.open()

            print("Grasp completed")
            input()
            continue
    else:
        print('No valid grasp could be executed due to collisions.')

def execute_grasp(robot, initial_ik_solution, sim):
    # Move the robot to the pre-grasp position
    pre_grasp_pos = robot.end_effector_pose()
    pre_grasp_pos[2] += 0.1  # Move 10 cm above the object
    pre_grasp_joint_config = robot.inverse_kinematics(pre_grasp_pos)
    
    if pre_grasp_joint_config is not None and pre_grasp_joint_config.size > 0:
        robot.move_to(pre_grasp_joint_config)
    
        # Lower the robot to the grasp position
        grasp_pos = robot.end_effector_pose()
        grasp_pos[2] -= 0.1  # Move 10 cm down to the object
        grasp_joint_config = robot.inverse_kinematics(grasp_pos)
        
        if grasp_joint_config is not None and grasp_joint_config.size > 0:
            robot.move_to(grasp_joint_config)
            
            # Close the gripper to grasp the object
            robot.close()
            
            # Lift the object
            lift_pos = robot.end_effector_pose()
            lift_pos[2] += 0.1  # Move 10 cm up with the object
            lift_joint_config = robot.inverse_kinematics(lift_pos)
            
            if lift_joint_config is not None and lift_joint_config.size > 0:
                robot.move_to(lift_joint_config)
                
                # Optionally, move the object to a new location
                # new_pos = robot.end_effector_pose()
                # new_pos[0] += 0.2  # Move 20 cm to the right
                # new_joint_config = robot.inverse_kinematics(new_pos)
                # if new_joint_config is not None and new_joint_config.size > 0:
                #     robot.move_to(new_joint_config)
                
                # Open the gripper to release the object
                robot.open()
            else:
                print('Failed to find IK solution for lifting the object.')
        else:
            print('Failed to find IK solution for grasping the object.')
    else:
        print('Failed to find IK solution for pre-grasp position.')

if __name__ == '__main__':
    main()
