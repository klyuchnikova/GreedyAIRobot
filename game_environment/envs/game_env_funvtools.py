import numpy as np
import random
from gym import error, spaces, utils
from gym.utils import seeding
from ... import game_vars

def check_initial_vars_correctness_():
    assert game_vars.NUM_BOTS >= 1
    assert game_vars.NUM_ROUNDS >= 1
    assert game_vars.MATCH_MODE in ['FRIENDLY', 'DEATHMATCH']
    assert game_vars.VIEW_RADIUS >= game_vars.MINING_RADIUS
    assert game_vars.VIEW_RADIUS >= game_vars.ATTACK_RADIUS
    assert 1 <= game_vars.MAP_WIDTH <= 32767
    assert 1 <= game_vars.MAP_HEIGHT <= 32767

def check_initial_bots_correctness_(bots):
    assert len(bots) == game_vars.NUM_BOTS

def count_action_mask(VIEW_RADIUS, MINING_RADIUS, ATTACK_RADIUS = -1):
    # generates 3d square matrix with uneven side with bools for whether the corresponding ceil should be taken or not
    return []