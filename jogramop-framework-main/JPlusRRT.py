import numpy as np
import random

class JPlusRRT:
    def __init__(self, robot, start_config, goal_config, max_iter=1000, step_size=0.1):
        self.robot = robot
        self.start_config = start_config
        self.goal_config = goal_config
        self.max_iter = max_iter
        self.step_size = step_size
        self.tree = {tuple(start_config): None}

    def distance(self, config1, config2):
        return np.linalg.norm(np.array(config1) - np.array(config2))

    def random_config(self):
        return [random.uniform(joint[0], joint[1]) for joint in self.robot.arm_joint_limits()]

    def nearest_neighbor(self, random_config):
        return min(self.tree.keys(), key=lambda x: self.distance(x, random_config))

    def new_config(self, nearest_config, random_config):
        direction = np.array(random_config) - np.array(nearest_config)
        norm = np.linalg.norm(direction)
        step = direction / norm * self.step_size
        return tuple(np.array(nearest_config) + step)

    def plan(self):
        for _ in range(self.max_iter):
            random_config = self.random_config()
            nearest_config = self.nearest_neighbor(random_config)
            new_config = self.new_config(nearest_config, random_config)
            if not self.robot.in_collision():
                self.tree[new_config] = nearest_config
                if self.distance(new_config, self.goal_config) < self.step_size:
                    self.tree[self.goal_config] = new_config
                    return self.reconstruct_path()
        return None

    def reconstruct_path(self):
        path = []
        config = self.goal_config
        while config is not None:
            path.append(config)
            config = self.tree[config]
        path.reverse()
        return path
