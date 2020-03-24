import numpy as np
import Env

default_config = dict(
    # Environment hyper-parameters
    env_name = "Jamming Attack",
    max_iteration = 10,
    max_episode_length = 100000,
    evaluate_interval = 100,
    max_channel = 100,
    total_packet = 1000,
    # Training hyper-parameters
    gamma = 0.99,
    eps = 0.3,
    seed = 0,
)

class Agent(object):
    """docstring for Agent"""

    def __init__(self, config):
        super(Agent, self).__init__()
        self.__action_move_channel = 0
        self.__action_send_packet = 1
        self.__Max_channel = config["max_channel"]

    def random_policy(self):
        self.__action_move_channel = np.random.choice(self.__Max_channel)
        self.__action_send_packet = np.random.randint(2)

    def stay_policy(self):
        self.__action_move_channel = 0
        self.__action_send_packet = 1

    @property
    def act_c(self):
        return self.__action_move_channel

    @property
    def act_s(self):
        return self.__action_send_packet