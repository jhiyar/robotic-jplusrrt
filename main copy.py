from robot import Robot
from goal import Goal


if __name__ == '__main__':
    # create robot (which also contains simulation environment)
    robot = Robot(with_gui=True)

    # show all the goals (you would just use one at a time)
    for i in range(6):
        goal = Goal(i)
        robot.set_goal(goal)  # only serves visualization purposes

    print('---------------------')
    print('the joint limits of the robot are:')
    print('lower: ', robot.joint_limits()[0])
    print('upper: ', robot.joint_limits()[1])

    print('---------------------')
    input('hit enter to continue')

    # some exemplary usage of functions
    joint_pos = [-0.4, 0.4, 0.4, -1.7, 0.0, 1.57, 0.75]
    print('resetting arm joints to:', joint_pos)
    robot.reset_joint_pos(joint_pos)
    goal = Goal(1)
    print('robot is in collision:', robot.in_collision())
    print('end effector position:', robot.ee_position())
    print('EE distance to goal:', goal.distance(robot.ee_position()))
    print('is the goal reached?', goal.reached(robot.ee_position()))
    print('jacobian:', robot.get_jacobian().shape)
    print(robot.get_jacobian())

    print('---------------------')
    input('hit enter to close simulation')
    robot.disconnect()
