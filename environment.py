import numpy as np

default_config = dict(
    # Environment hyper-parameters
    env_name="Jamming Attack",
    max_iteration=10,
    max_episode_length=100000,
    evaluate_interval=100,
    max_channel=100,
    total_packet=1000,
    # Training hyper-parameters
    gamma=0.99,
    eps=0.3,
    seed=0,
)


class Env(object):
    """docstring for env"""

    def __init__(self, config):
        super(Env, self).__init__()
        self.__config = config

        self.__Max_channel = config["max_channel"]
        self.__Total_packet = config["total_packet"]
        self.__channels = np.zeros(self.__Max_channel)

        self.__current_state = 0  # Default current_state is 0
        self.__time = 0  # Reset time
        self.__attack_mode = 0  # Set attacker's mode to be default mode
        self.__attacked_channels = []
        self.__ACK_sent = [False]
        self.__last_round_send_packet = False
        self.__last_round_send_channel = -1  # Only for attacker mode = 5
        self.__attack_target = []  # Only for attacker mode = 3

        self.__sent_packets = 0
        self.__received_ACK = 0
        self.__PSR = 0.0
        self.__PDR = 0.0

    def reset(self, attack_mode=0):
        print("--------------------------------------------------------")
        print("Test", self.__config["env_name"], "with Attack Mode =", attack_mode)

        self.__current_state = 0  # Default current_state is 0
        self.__time = 0  # Reset time
        # print("Current Time is ", self.__time)
        self.__attack_mode = attack_mode  # Set attacker's mode
        self.__attacked_channels = []
        self.__ACK_sent = [False]
        self.__last_round_send_packet = False
        self.__last_round_send_channel = -1  # Only for attacker mode = 5
        self.__attack_target = []  # Only for attacker mode = 3

        self.__sent_packets = 0
        self.__received_ACK = 0
        self.__PSR = 0.0
        self.__PDR = 0.0

    def step(self, action_move_channel, action_send_packet):

        if (action_move_channel > self.__Max_channel) or (action_move_channel < 0):
            print("Error in action_move_channel")
            return

        if self.__received_ACK == self.__Total_packet:
            print("Error! No more step if already Done!")
            return

        self.__time += 1

        # Opponent attack
        self.__opponent_attack(self.__attack_mode)

        # Agent changes the channel and sends packet
        self.__current_state = action_move_channel
        new_state = self.__current_state

        # reward
        if self.__ACK_sent[self.__time - 1]:  # if receive the ACK (sent from t - 1)
            reward = 1
        elif self.__last_round_send_packet:  # else if not receive the ACK but sent packets in the previous step
            reward = -1
        else:
            reward = 0

        # Send the packet or not
        if action_send_packet == 1:
            self.__send_packet(new_state, True)
            self.__last_round_send_packet = True
            self.__last_round_send_channel = new_state
        else:
            self.__send_packet(new_state, False)
            self.__last_round_send_packet = False
            self.__last_round_send_channel = -1

        # done
        if self.__received_ACK == self.__Total_packet:
            end = True
        else:
            end = False

        self.__PSR = np.round(self.__sent_packets / self.__time, 3)
        # print("Totally send", self.__sent_packets, "packets. And cost time is", self.__time, ".")
        self.__PDR = np.round(self.__received_ACK / max(1, self.__sent_packets), 3)
        # print("Total", self.__received_ACK, "packets are received. And send", self.__sent_packets, "packets.")
        return new_state, reward, end, [self.__PSR, self.__PDR]

    @property
    def act_dim(self):
        return self.__Max_channel

    @property
    def obs_dim(self):
        return self.__Max_channel

    @property
    def time(self):
        return self.__time

    def __attack(self):
        for i in range(len(self.__channels)):
            if i in self.__attacked_channels:
                self.__channels[i] = 1
            else:
                self.__channels[i] = 0

    def __send_packet(self, current_state, flag):
        if self.__channels[current_state] == 0 and flag:  # Successfully send packet
            self.__sent_packets += 1
            self.__received_ACK += 1
            self.__ACK_sent.append(True)
        elif self.__channels[current_state] == 1 and flag:  # The channel has been occupied
            self.__sent_packets += 1
            self.__ACK_sent.append(False)
        else:  # Did not send the packet
            self.__ACK_sent.append(False)

    def __opponent_attack(self, mode=0):
        # Modify the self.__attacked_channels list
        if mode == 0:
            # No attack
            return
        elif mode == 1:
            # Constant jammer, focus on several channels to continuously attack
            if not self.__attacked_channels:
                rd = np.random.choice(self.__Max_channel)
                for also_attack in range(max(0, rd - 15), min(self.__Max_channel, rd + 15)):
                    self.__attacked_channels.append(also_attack)
                # print(self.__attacked_channels)
            self.__attack()
            return
        elif mode == 2:
            # Constant jammer, continuously randomly choose several channel to attack
            self.__attacked_channels.clear()
            while len(self.__attacked_channels) < self.__Max_channel / 4:
                x = np.random.randint(0, self.__Max_channel)
                if x not in self.__attacked_channels:
                    self.__attacked_channels.append(x)
            self.__attack()
            return
        elif mode == 3:
            # Random jammer, switch back and forth between sleep and active, when active it focus on several channels
            # to attack
            active_time = 10
            sleep_time = 10
            # np.random.seed(self.__seed)
            if not self.__attack_target:
                while len(self.__attack_target) < self.__Max_channel / 4:
                    x = np.random.randint(0, self.__Max_channel)
                    if x not in self.__attack_target:
                        self.__attack_target.append(x)

            if self.__time % (active_time + sleep_time) < active_time:  # when active
                self.__attacked_channels.clear()
                for i in range(len(self.__attack_target)):
                    self.__attacked_channels.append(self.__attack_target[i])
            else:
                self.__attacked_channels.clear()
            # print("In mode 3, the attacked channels are", self.__attacked_channels)
            self.__attack()
            return
        elif mode == 4:
            # Random jammer, switch back and forth between sleep and active, when active it will randomly choose
            # servel channels to attack
            active_time = 10
            sleep_time = 10
            if self.__time % (active_time + sleep_time) < active_time:  # when active
                self.__attacked_channels.clear()
                while len(self.__attacked_channels) < self.__Max_channel / 4:
                    x = np.random.randint(0, self.__Max_channel)
                    if x not in self.__attacked_channels:
                        self.__attacked_channels.append(x)
            else:
                self.__attacked_channels.clear()
            # print("In mode 4, the attacked channels are", self.__attacked_channels)
            self.__attack()
            return
        elif mode == 5:
            # Reactive jammer, which can passively listen and obtain the communication channel used by the agent,
            # and attack
            if self.__last_round_send_channel != -1:  # Sniff the channel that agent sent packet
                self.__attacked_channels.clear()
                for also_attack in range(max(0, self.__last_round_send_channel - 5),
                                         min(self.__Max_channel, self.__last_round_send_channel + 5)):
                    self.__attacked_channels.append(also_attack)
            # print("In mode 4, the attacked channels are", self.__attacked_channels)
            self.__attack()
            return


if __name__ == '__main__':

    class Agent(object):
        """docstring for Agent"""

        def __init__(self, config):
            super(Agent, self).__init__()
            self.__current_channel = 0
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


    # Main Loop
    test_env = Env(default_config)
    agent = Agent(default_config)

    for test_mode in range(6):
        for num in range(default_config["max_iteration"]):
            test_env.reset(test_mode)
            done = False
            for _ in range(default_config["max_episode_length"]):
                agent.random_policy()
                state_new, new_reward, done, info = test_env.step(agent.act_c, agent.act_s)
                if done:
                    break
            print("In Episode", num, ", The PSR is", info[0], ", the PDR is", info[1], ".")
