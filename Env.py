import numpy as np


class Env(object):
    """docstring for env"""

    def __init__(self, max_channel, total_package, attack_mode=0):
        super(Env, self).__init__()
        self.__MAX_Channel = max_channel
        self.__Total_package = total_package
        self.__channels = np.zeros(self.__MAX_Channel)

        self.__current_state = 0  # Default current_state is 0
        self.__time = 0  # Reset time
        self.__attack_mode = attack_mode  # Set attacker's mode
        self.__attacked_channels = []
        self.__sent_packages = 0
        self.__ACK_sent = [False]

    def reset(self, attack_mode=0):
        self.__current_state = 0  # Default current_state is 0
        self.__time = 0  # Reset time
        # print("Current Time is ", self.__time)
        self.__attack_mode = attack_mode  # Set attacker's mode
        self.__attacked_channels = []
        self.__sent_packages = 0
        self.__ACK_sent = [False]

    def step(self, action):

        if (action > self.__MAX_Channel) or (action < 0):
            print("Error in action")
            return

        if self.__sent_packages == self.__Total_package:
            print("Error! No more step if already Done!")
            return

        self.__time += 1
        # print("Current Time is ", self.__time)

        # Opponent attack
        self.__opponent_attack(self.__attack_mode)

        # Agent changes the channel and sends package
        self.__current_state = action
        new_state = self.__current_state
        self.__send_package(new_state)

        # reward
        if self.__ACK_sent[self.__time - 1]:  # whether receive the ACK (sent from t - 1)
            self.__sent_packages += 1
            reward = 1
        else:
            reward = 0

        # done
        if self.__sent_packages == self.__Total_package:
            done = True
        else:
            done = False

        return new_state, reward, done, 0

    @property
    def act_dim(self):
        return self.__MAX_Channel

    @property
    def obs_dim(self):
        return self.__MAX_Channel

    @property
    def time(self):
        return self.__time

    def __attack(self):
        for i in range(len(self.__channels)):
            if i in self.__attacked_channels:
                self.__channels[i] = 1
            else:
                self.__channels[i] = 0

    def __send_package(self, current_state):
        if self.__channels[current_state] == 0:  # Successfully send package
            self.__ACK_sent.append(True)
        else:  # The channel has been occupied
            self.__ACK_sent.append(False)

    def __opponent_attack(self, mode=0):
        # Modify the self.__attacked_channels list
        if mode == 0:
            # No attack
            return
        elif mode == 1:
            # Randomly choose only ONE channel to attack
            self.__attacked_channels.clear()
            self.__attacked_channels.append(np.random.choice(self.__MAX_Channel))
            self.__attack()
            return
        elif mode == 2:
            # Randomly choose only HALF channels to attack
            self.__attacked_channels.clear()
            while len(self.__attacked_channels) < self.__MAX_Channel / 2:
                x = np.random.randint(0, self.__MAX_Channel)
                if x not in self.__attacked_channels:
                    self.__attacked_channels.append(x)
            self.__attack()
            return


if __name__ == '__main__':
    MAX_channel = 100
    Total_package = 1000
    Num_episode = 10
    Attack_mode = 0  # Can change the mode from 0 - 2


    class Agent(object):
        """docstring for Agent"""

        def __init__(self, arg):
            super(Agent, self).__init__()
            self.arg = arg


    # Main Loop
    test_env = Env(MAX_channel, Total_package)
    for mode in range(3):
        print("For attack mode = ", mode)
        for num in range(Num_episode):
            test_env.reset(mode)
            done = False
            while not done:
                action = np.random.choice(test_env.act_dim)
                state_new, new_reward, done, info = test_env.step(action)
            print("Episode ", num, ", the time cost is ", test_env.time)
