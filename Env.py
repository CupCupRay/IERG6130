import numpy as np

def env:
	MAX_Channel = 100
	Channels = np.zeros(MAX_Channel)

def opponent_strategy(env, mode=0):
	if mode=0:
		return np.random.choice(env.MAX_Channel)
	else if mode=1:
		return 0