from enum import Enum

import gym
import numpy as np

import game_env_funvtools
from ... import game_vars


class GameMode(Enum):
    FRIENDLY = 'FRIENDLY'
    DEATHMATCH = 'DEATHMATCH'


class CoinMinerEnv(gym.Env):
    metadata = {'render.modes': ['FRIENDLY', 'DEATHMATCH'],
                'step.modes': ['FRIENDLY', 'DEATHMATCH']}
    tools = game_env_funvtools

    def __init__(self, bots, mode='FRIENDLY'):
        super(CoinMinerEnv, self).__init__()
        CoinMinerEnv.tools.check_initial_vars_correctness_()
        CoinMinerEnv.tools.check_initial_bots_correctness_(bots)

        if mode == 'FRIENDLY':
            self.mode = GameMode.FRIENDLY
            self.influence_mask = CoinMinerEnv.tools.count_influence_mask(game_vars.VIEW_RADIUS, game_vars.MINING_RADIUS)
        else:
            self.mode = GameMode.DEATHMATCH
            self.influence_mask = CoinMinerEnv.tools.count_influence_mask(game_vars.VIEW_RADIUS, game_vars.MINING_RADIUS, game_vars.ATTACK_RADIUS)

        self.bots = dict.fromkeys(np.arange(game_vars.NUM_BOTS), bots)

        self.ROUND_NUMBER = 1
        self.bots_score = dict.fromkeys(np.arange(game_vars.NUM_BOTS), 0)
        self.bots_alive = dict.fromkeys(np.arange(game_vars.NUM_BOTS), True)
        self.bots_died_this_round = set()
        self.NUM_BOTS_ALIVE = game_vars.NUM_BOTS

        self.bot_finished_step_order = dict.fromkeys(np.arange(game_vars.NUM_BOTS), -1)
        self.number_waiting_step = game_vars.NUM_BOTS
        self.finished_step_round = False

        # we'll have a 2d matrix where 1 element is reserved for bots, second is for coins
        num_channels = 2
        self.observation_space = gym.spaces.Box(np.int32,
                                                num_channels,
                                                shape=(num_channels, game_vars.MAP_WIDTH, game_vars.MAP_HEIGHT))
        # action pool -> dx^2 + dy^2 <= r^2 => action can be done within radius
        self.action_space = gym.spaces.Discrete(9)
        self.done = False

    def step(self, action):
        assert not self.done
        bot_id, bot_action = action['bot_id'], action['action']
        assert 0 <= bot_id < game_vars.NUM_BOTS
        assert self.bot_finished_step_order[bot_id] == -1
        if isinstance(bot_action, tuple) or isinstance(bot_action, list) or isinstance(bot_action, np.ndarray):
            assert -1 <= bot_action[0] <= 1
            assert -1 <= bot_action[0] <= 1
        else:
            bot_action = [0, 0]

        if self.bots_alive[bot_id]:
            self.observation_space = CoinMinerEnv.tools.step(self, bot_id, bot_action)
            self.bot_finished_step_order[bot_id] = self.NUM_BOTS_ALIVE - self.number_waiting_step
            self.number_waiting_step -= 1

        if CoinMinerEnv.tools.finished_round(self):
            self.next_round()

    def next_round(self):
        CoinMinerEnv.tools.next_round(self)
        self.ROUND_NUMBER += 1
        self.bot_finished_step_order = dict.fromkeys(np.arange(game_vars.NUM_BOTS), -1)
        self.number_waiting_step = self.NUM_BOTS_ALIVE
        self.finished_step_round = False
        self.done = CoinMinerEnv.tools.game_ended(self)
        self.send_responses()
        self.bots_died_this_round = set()

    def send_responses(self):
        # generate and send responses to bots which are still in the game ore just left
        if self.done:
            for bot_id in np.arange(game_vars.NUM_BOTS):
                if self.bots_alive[bot_id] or bot_id in self.bots_died_this_round:
                    yield {"bot_id": bot_id, "start message": "match over"}
        else:
            for bot_id in np.arange(game_vars.NUM_BOTS):
                if self.bots_alive[bot_id]:
                    response = {"bot_id": bot_id, "start message": "update", "data": {"round": self.ROUND_NUMBER,
                                                                                      **CoinMinerEnv.tools.scan_area_around(self, bot_id, self.influence_mask)}}
                    yield response
                elif bot_id in self.bots_died_this_round:
                    yield {"bot_id": bot_id, "start message": "match over"}
            for bot_id in np.arange(game_vars.NUM_BOTS):
                yield {"bot_id": bot_id, "start message": "match over"}

    def reset(self):
        self.ROUND_NUMBER = 1
        self.bots_score = dict.fromkeys(np.arange(game_vars.NUM_BOTS), 0)
        self.bots_alive = dict.fromkeys(np.arange(game_vars.NUM_BOTS), True)
        self.bots_died_this_round = set()
        self.NUM_BOTS_ALIVE = game_vars.NUM_BOTS

        self.bot_finished_step_order = dict.fromkeys(np.arange(game_vars.NUM_BOTS), -1)
        self.number_waiting_step = game_vars.NUM_BOTS
        self.finished_step_round = False
        self.done = False

        num_channels = 2
        self.observation_space = gym.spaces.Box(np.int32,
                                                num_channels,
                                                shape=(num_channels, game_vars.MAP_WIDTH, game_vars.MAP_HEIGHT))
        self.render_new_state()

    def render_new_state(self):
        self.tools.render_money(self, game_vars.COIN_START_SPAWN_VOLUME)
        self.tools.render_bot_coordinates(self)
