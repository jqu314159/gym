import gym
from gym import spaces
import numpy as np
import math
from gym.envs.custom.game_env import PyGame2D

class CustomEnv(gym.Env):
    #metadata = {'render.modes' : ['human']}
    def __init__(self):

        self.pygame = PyGame2D()
        self.action_space = spaces.Box(
            low = np.array([
              self.pygame.car.min_angular_acceleration_action,
              self.pygame.car.min_acceleration_action
            ]),
            high = np.array([
              self.pygame.car.max_angular_acceleration_action,
              self.pygame.car.max_acceleration_action
            ]),
            shape=(2,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
          np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
          np.array([10, 10, 10, 10, 10, 10, 10, 10, 10,
            (self.pygame.car.screen_width // 100) + 1,
            (self.pygame.car.screen_height // 100) + 1,
            (self.pygame.car.screen_width // 100) + 1,
            (self.pygame.car.screen_height // 100) + 1,
            1,
            1]
          ),
          dtype=np.float
        )
	#last 5 is current position(nomolize) , goal position , current angle

    def reset(self):
        del self.pygame
        self.pygame = PyGame2D()
        obs = self.pygame.observe()
        return obs

    def step(self, action: np.ndarray):
        self.pygame.action(action)
        obs = self.pygame.observe()
        reward = self.pygame.evaluate()
        done = self.pygame.is_done()
        return obs, reward, done, {}
        
    def double_reward_step(self, action: np.ndarray):
        self.pygame.action(action)
        obs = self.pygame.observe()
        reward , reward2= self.pygame.double_reward_evaluate()
        done = self.pygame.is_done()
        return obs, reward, reward2, done, {}

    def render(self, mode="human", close=False):
        self.pygame.view()

