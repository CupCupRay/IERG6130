import numpy as np


class Env(object):
    """docstring for env"""

    def __init__(self, max_channel, total_packet, attack_mode=0):
        super(Env, self).__init__()
        self.__Max_channel = max_channel
        self.__Total_packet = total_packet
        self.__channels = np.zeros(self.__Max_channel)

        self.__current_state = 0  # Default current_state is 0
        self.__time = 0  # Reset time
        self.__attack_mode = attack_mode  # Set attacker's mode
        self.__attacked_channels = []
        self.__ACK_sent = [False]
        self.__last_round_send_packet = False

        self.__sent_packets = 0
        self.__received_ACK = 0
        self.__PSR = 0.0
        self.__PDR = 0.0

    def reset(self, attack_mode=0):
        self.__current_state = 0  # Default current_state is 0
        self.__time = 0  # Reset time
        # print("Current Time is ", self.__time)
        self.__attack_mode = attack_mode  # Set attacker's mode
        self.__attacked_channels = []
        self.__ACK_sent = [False]
        self.__last_round_send_packet = False

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
        else:
            self.__send_packet(new_state, False)
            self.__last_round_send_packet = False

        # done
        if self.__received_ACK == self.__Total_packet:
            end = True
        else:
            end = False

        if end:
            self.__PSR = np.round(self.__sent_packets / self.__time, 3)
            # print("Totally send", self.__sent_packets, "packets. And cost time is", self.__time, ".")
            self.__PDR = np.round(self.__received_ACK / self.__sent_packets, 3)
            # print("Total", self.__received_ACK, "packets are received. And send", self.__sent_packets, "packets.")
            return new_state, reward, end, [self.__PSR, self.__PDR]
        else:
            return new_state, reward, end, None

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
            # Randomly choose only ONE channel to attack
            self.__attacked_channels.clear()
            self.__attacked_channels.append(np.random.choice(self.__Max_channel))
            self.__attack()
            return
        elif mode == 2:
            # Randomly choose only HALF channels to attack
            self.__attacked_channels.clear()
            while len(self.__attacked_channels) < self.__Max_channel / 2:
                x = np.random.randint(0, self.__Max_channel)
                if x not in self.__attacked_channels:
                    self.__attacked_channels.append(x)
            self.__attack()
            return


if __name__ == '__main__':
    Max_channel = 100
    Total_packet = 1000
    Num_episode = 10
    Max_num_per_episode = 100000
    Attack_mode = 0  # Can change the mode from 0 - 2


    class Agent(object):
        """docstring for Agent"""

        def __init__(self):
            super(Agent, self).__init__()
            self.__action_move_channel = 0
            self.__action_send_packet = 1

        def update_policy(self):
            self.__action_move_channel = np.random.choice(100)
            self.__action_send_packet = np.random.choice(2)

        @property
        def act_c(self):
            return self.__action_move_channel

        @property
        def act_s(self):
            return self.__action_send_packet


    # Main Loop
    test_env = Env(Max_channel, Total_packet)
    agent = Agent()

    for test_mode in range(3):
        print("--------------------------------------------------------")
        print("For attack mode = ", test_mode)
        for num in range(Num_episode):
            test_env.reset(test_mode)
            done = False
            for _ in range(Max_num_per_episode):
                agent.update_policy()
                state_new, new_reward, done, info = test_env.step(agent.act_c, agent.act_s)
                if done:
                    print("In Episode,", num, "The PSR is", info[0], ", the PDR is", info[1], ".")
                    break
