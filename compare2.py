from robot import Robot
from JPlusRRT11 import JPlusRRT
from IKRRT import IKRRT
from BIKRRT import BIKRRT
from RRTStar2 import RRTStar
from BIKRRTOptimized import BIKRRTOptimized

import numpy as np
import time
from goal import Goal


# Function to calculate path length
def calculate_path_length(path):
    length = 0
    for i in range(1, len(path)):
        node1 = path[i - 1]['ee_position']
        node2 = path[i]['ee_position']
        length += np.linalg.norm(np.array(node2) - np.array(node1))
    return length


# Function to run the planner and return results
def run_planner(planner_class, robot, start_position, goal_position, num_trials=10):
    success_count = 0
    total_planning_time = 0
    total_path_length = 0

    for _ in range(num_trials):
        planner = planner_class(robot, goal_direction_probability=0.9, with_visualization=False)

        start_time = time.time()
        path = planner.plan(start_position, goal_position)
        planning_time = time.time() - start_time

        if path:
            success_count += 1
            path_length = calculate_path_length(path)
            total_path_length += path_length
        else:
            path_length = None

        total_planning_time += planning_time

    success_rate = success_count / num_trials
    avg_planning_time = total_planning_time / num_trials
    avg_path_length = total_path_length / success_count if success_count > 0 else None

    return success_rate, avg_planning_time, avg_path_length


if __name__ == '__main__':
    # Initialize the robot and goal
    robot = Robot(with_gui=False)
    goal_position = np.array([0.7, 0.0, 0.6])
    
    start_position = np.array(robot.ee_position())

    # Dictionary to store results
    results = {
        "JPlusRRT": {},
        "RRTStar": {},
        "IKRRT": {},
        "BIKRRT": {},
        "BIKRRTOptimized": {}
    }

    # Run each planner and store the results
    results["JPlusRRT"]["success_rate"], results["JPlusRRT"]["avg_planning_time"], results["JPlusRRT"]["avg_path_length"] = run_planner(JPlusRRT, robot, start_position, goal_position)
    results["RRTStar"]["success_rate"], results["RRTStar"]["avg_planning_time"], results["RRTStar"]["avg_path_length"] = run_planner(RRTStar, robot, start_position, goal_position)
    results["IKRRT"]["success_rate"], results["IKRRT"]["avg_planning_time"], results["IKRRT"]["avg_path_length"] = run_planner(IKRRT, robot, start_position, goal_position)
    results["BIKRRT"]["success_rate"], results["BIKRRT"]["avg_planning_time"], results["BIKRRT"]["avg_path_length"] = run_planner(BIKRRT, robot, start_position, goal_position)
    results["BIKRRTOptimized"]["success_rate"], results["BIKRRTOptimized"]["avg_planning_time"], results["BIKRRTOptimized"]["avg_path_length"] = run_planner(BIKRRTOptimized, robot, start_position, goal_position)

    # Print the results
    for planner, data in results.items():
        print(f"Results for {planner}:")
        print(f"  Success Rate: {data['success_rate']*100:.2f}%")
        print(f"  Average Planning Time: {data['avg_planning_time']:.4f} seconds")
        if data['avg_path_length'] is not None:
            print(f"  Average Path Length: {data['avg_path_length']:.4f}")
        else:
            print(f"  Average Path Length: No successful paths found")
        print("-" * 40)

    robot.disconnect()
