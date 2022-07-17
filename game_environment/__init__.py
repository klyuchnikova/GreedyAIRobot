from gym.envs.registration import register

register(id = 'CoinMiner-v0',
         entry_point='gym_bubbleshooter.envs:CoinMinerEnv',
         )
