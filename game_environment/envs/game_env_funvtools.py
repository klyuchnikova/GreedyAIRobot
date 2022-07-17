import numpy as np

from game_env import GameMode
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


def count_influence_mask(VIEW_RADIUS, MINING_RADIUS, ATTACK_RADIUS=-1):
    # generates 3d square matrix with uneven side with bools for whether the corresponding ceil should be taken or not
    side = int(max(np.sqrt(VIEW_RADIUS), np.sqrt(MINING_RADIUS), np.sqrt(ATTACK_RADIUS))) * 2 + 1
    influence_mask = np.zeros((side, side, 3))
    half_side = (side - 1) // 2
    for dx in range(-half_side, half_side + 1):
        for dy in range(-half_side, half_side + 1):
            if dx**2 + dy**2 <= VIEW_RADIUS:
                influence_mask[dx + half_side][dy + half_side][0] = 1
            if dx**2 + dy**2 <= MINING_RADIUS:
                influence_mask[dx + half_side][dy + half_side][1] = 1
            if dx**2 + dy**2 <= ATTACK_RADIUS:
                influence_mask[dx + half_side][dy + half_side][2] = 1
    return influence_mask


def finished_round(game_env):
    return game_env.number_waiting_step == 0


def game_ended(game_env):
    return game_env.number_waiting_step == 0 and game_env.ROUND_NUMBER == game_vars.NUM_ROUNDS


def step(game_env, bot_id, bot_action):
    # only if there's no one in the desired position the step is made
    old_coords = game_env.bots_coordinates[bot_id]
    new_coords = ((old_coords[0] + bot_action[0])%game_vars.MAP_WIDTH, (old_coords[1] + bot_action[1])%game_vars.MAP_HEIGHT)
    if game_env.observation_space[new_coords[0]][new_coords[1]][0] == -1:
        game_env.bots_coordinates[bot_id] = new_coords
        game_env.observation_space[old_coords[0]][old_coords[1]][0] = -1
        game_env.observation_space[new_coords[0]][new_coords[1]][0] = bot_id
    else:
        pass


def next_round(game_env):
    # 1. collect money
    # 2. death matches
    # 3. spawn of money

    alive_bots = [k for k, v in game_env.bots_alive.items() if v]
    alive_bots.sort(key=lambda x: game_env.bot_finished_step_order[x])

    for bot_id in alive_bots:
        new_coords = game_env.bots_coordinates[bot_id]
        game_env.bots_score[bot_id] += game_env.observation_space[new_coords[0]][new_coords[1]][1]
        game_env.observation_space[new_coords[0]][new_coords[1]][1] = 0

    if game_env.mode is GameMode.DEATHMATCH:
        for bot_id in alive_bots:
            x, y = game_env.bots_coordinates[bot_id]
            for sec_bot_id in alive_bots:
                x2, y2 = game_env.bots_coordinates[sec_bot_id]
                if sec_bot_id != bot_id and (x2 - x)**2 + (y2 - y)**2 <= game_vars.ATTACK_RADIUS:
                    # battle!
                    deathmatch_between(game_env, bot_id, sec_bot_id)

    if game_env.ROUND_NUMBER % game_vars.COIN_SPAWN_PERIOD == 0:
        render_money(game_env, game_vars.COIN_SPAWN_VOLUME)


def deathmatch_between(game_env, fir_id, sec_id):
    if game_env.bots_score[fir_id] < game_env.bots_score[sec_id]:
        winner, looser = sec_id, fir_id
    elif game_env.bots_score[fir_id] == game_env.bots_score[sec_id]:
        if np.random.randint(2, size=1) == 0:
            winner, looser = sec_id, fir_id
        else:
            winner, looser = fir_id, sec_id
    else:
        winner, looser = fir_id, sec_id
    game_env.bots_score[winner] += game_env.bots_score[looser]
    game_env.bots_alive[looser] = False
    game_env.bots_died_this_round.add(looser)
    x, y = game_env.bots_coordinates[looser]
    game_env.observation_space[x][y][0] = -1
    game_env.NUM_BOTS_ALIVE -= 1

def render_money(game_env, money_amount):
    map_size = game_vars.MAP_WIDTH*game_vars.MAP_HEIGHT
    new_money_coordinates = np.random.randint(map_size, size=money_amount)
    for coord in new_money_coordinates:
        x, y = coord%game_vars.MAP_WIDTH, coord//game_vars.MAP_WIDTH
        game_env.observation_space[x][y][1] += 1

def render_bot_coordinates(game_env):
    map_size = game_vars.MAP_WIDTH * game_vars.MAP_HEIGHT
    new_bot_coordinates = np.random.choice(map_size, replace = False, size=game_vars.NUM_BOTS)
    for i, coord in enumerate(new_bot_coordinates):
        x, y = coord % game_vars.MAP_WIDTH, coord // game_vars.MAP_WIDTH
        game_env.observation_space[x][y][0] = i

def scan_area_around(game_env, bot_id, view_mask):
    scan_area_half_size = len(view_mask)//2
    x,y = game_env.bots_coordinates[bot_id]
    coins_seen = []
    enemies_seen = []
    blocks_seen = []
    for dx in range(-scan_area_half_size, scan_area_half_size+1):
        for dy in range(-scan_area_half_size, scan_area_half_size+1):
            if view_mask[dx][dy][0]:
                enemy_id = game_env.observation_space[x+dx][y+dy][0]
                if enemy_id != -1 and enemy_id != bot_id:
                    enemies_seen.append({"x" : x+dx, "y" : y+dy, "NUM_BOT_COINS" : game_env.bots_score[enemy_id], "BOT_ID" : enemy_id})
                for coin in range(game_env.observation_space[x+dx][y+dy][1]):
                    coins_seen.append({"x" : x+dx, "y" : y+dy})
    return {"coin" : coins_seen, "bot" : enemies_seen}


def generate_map(observation_space):
    '''
    Return coin area, block area, robot area (with numbers)
    '''
    game_map = np.zeros((game_vars.MAP_WIDTH, game_vars.MAP_HEIGHT))
    for x in range(len(observation_space)):
        for y in range(len(observation_space[x])):

            game_map[x][y]
    all_pieces = np.sum(state[[govars.BLACK, govars.WHITE]], axis=0)
    empties = 1 - all_pieces

    empty_labels, num_empty_areas = ndimage.measurements.label(empties)

    black_area, white_area = np.sum(state[govars.BLACK]), np.sum(state[govars.WHITE])
    for label in range(1, num_empty_areas + 1):
        empty_area = empty_labels == label
        neighbors = ndimage.binary_dilation(empty_area)
        black_claim = False
        white_claim = False
        if (state[govars.BLACK] * neighbors > 0).any():
            black_claim = True
        if (state[govars.WHITE] * neighbors > 0).any():
            white_claim = True
        if black_claim and not white_claim:
            black_area += np.sum(empty_area)
        elif white_claim and not black_claim:
            white_area += np.sum(empty_area)

    return black_area, white_area


def batch_areas(batch_state):
    black_areas, white_areas = [], []

    for state in batch_state:
        ba, wa = areas(state)
        black_areas.append(ba)
        white_areas.append(wa)
    return np.array(black_areas), np.array(white_areas)

def str(state):
    board_str = ''

    size = state.shape[1]
    board_str += '\t'
    for i in range(size):
        board_str += '{}'.format(i).ljust(2, ' ')
    board_str += '\n'
    for i in range(size):
        board_str += '{}\t'.format(i)
        for j in range(size):
            if state[0, i, j] == 1:
                board_str += '○'
                if j != size - 1:
                    if i == 0 or i == size - 1:
                        board_str += '═'
                    else:
                        board_str += '─'
            elif state[1, i, j] == 1:
                board_str += '●'
                if j != size - 1:
                    if i == 0 or i == size - 1:
                        board_str += '═'
                    else:
                        board_str += '─'
            else:
                if i == 0:
                    if j == 0:
                        board_str += '╔═'
                    elif j == size - 1:
                        board_str += '╗'
                    else:
                        board_str += '╤═'
                elif i == size - 1:
                    if j == 0:
                        board_str += '╚═'
                    elif j == size - 1:
                        board_str += '╝'
                    else:
                        board_str += '╧═'
                else:
                    if j == 0:
                        board_str += '╟─'
                    elif j == size - 1:
                        board_str += '╢'
                    else:
                        board_str += '┼─'
        board_str += '\n'

    black_area, white_area = areas(state)
    done = game_ended(state)
    ppp = prev_player_passed(state)
    t = turn(state)
    if done:
        game_state = 'END'
    elif ppp:
        game_state = 'PASSED'
    else:
        game_state = 'ONGOING'
    board_str += '\tTurn: {}, Game State (ONGOING|PASSED|END): {}\n'.format('BLACK' if t == 0 else 'WHITE', game_state)
    board_str += '\tBlack Area: {}, White Area: {}\n'.format(int(black_area), int(white_area))
    return board_str
