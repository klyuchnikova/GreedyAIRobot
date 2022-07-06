from enum import Enum
import gym
import numpy as np
import random
from ... import game_vars
import game_env_funvtools
from gym import error, spaces, utils
from gym.utils import seeding

class GameMode(Enum):
    FRIENDLY = 'FRIENDLY'
    DEATHMATCH = 'DEATHMATCH'

class BubbleShooterEnv(gym.Env):
    metadata = {'render.modes': ['FRIENDLY', 'DEATHMATCH'],
                'step.modes': ['FRIENDLY', 'DEATHMATCH']}
    tools = game_env_funvtools

    def __init__(self, bots, mode = 'FRIENDLY'):
        super(BubbleShooterEnv, self).__init__()
        BubbleShooterEnv.tools.check_initial_vars_correctness_()
        BubbleShooterEnv.tools.check_initial_bots_correctness_(bots)

        if mode == 'FRIENDLY':
            self.mode = GameMode.FRIENDLY
        else:
            self.mode = GameMode.DEATHMATCH

        self.bots = dict.fromkeys(np.arange(game_vars.NUM_BOTS), bots)
        self.bots_coordinates = dict.fromkeys(np.arange(game_vars.NUM_BOTS), (0, 0))
        self.bots_score = dict.fromkeys(np.arange(game_vars.NUM_BOTS), 0)

        # we'll have a 2d matrix where 1 element is reserved for bots, second is for coins
        num_channels = 2
        self.observation_space = gym.spaces.Box(np.int32,
                                                num_channels,
                                                shape=(num_channels, game_vars.MAP_WIDTH, game_vars.MAP_HEIGHT))
        # action pool -> dx^2 + dy^2 <= r^2 => action can be done within radius
        self.action_space = gym.spaces.Discrete(game_vars.MINING_RADIUS  game_vars.ATTACK_RADIUS)
        self.done = False

    def step(self, action):
        pass

    def reset(self):
        self.ROUND_NUMBER = 1
        self.NUM_BOT_COINS = dict.fromkeys(list(range(self.NUM_BOTS)), 0)

    def render(self, mode='human', close=False):
        pass
