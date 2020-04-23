import numpy as np
import random
from environment import Env

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

default_config = dict(
    # Environment hyper-parameters
    env_name = "Jamming Attack",
    max_iteration = 10000,
    max_episode_length = 10000,
    evaluate_interval = 100,
    max_channel = 100,
    total_packet = 1000,
    # Training hyper-parameters
    gamma = 0.99,
    eps = 0.3,
    seed = 0,
    decay_rate = 0.99,
    learning_rate = 1e-4,
    batch_size = 10,
    save_freq = 2000,
    log_freq = 10,
)


env = Env(default_config)
torch.manual_seed(default_config["seed"])



class Channel_Baseline(nn.Module):
    def __init__(self, num_channels=100):
        super(Channel_Baseline, self).__init__()
        self.affine1 = nn.Linear(100, 200)
        self.action_head = nn.Linear(200, num_channels) # action 1-100: different channels
        self.value_head = nn.Linear(200, 1)

        self.num_channels = num_channels
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=-1), state_values


    def select_action(self, x):
        x = Variable(torch.from_numpy(x).float().unsqueeze(0))
        probs, state_value = self.forward(x)
        m = Categorical(probs)
        action = m.sample()

        self.saved_log_probs.append((m.log_prob(action), state_value))
        return action





class Send_Packet_Baseline(nn.Module):
    def __init__(self, send_packets=2):
        super(Send_Packet_Baseline, self).__init__()
        self.affine1 = nn.Linear(100, 200)
        self.action_head = nn.Linear(200, send_packets) # action 1: send packet, action 2: do not send packet
        self.value_head = nn.Linear(200, 1)

        self.send_packets = send_packets
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=-1), state_values


    def select_action(self, x):
        x = Variable(torch.from_numpy(x).float().unsqueeze(0))
        probs, state_value = self.forward(x)
        m = Categorical(probs)
        action = m.sample()

        self.saved_log_probs.append((m.log_prob(action), state_value))
        return action





class Agent(object):
    """docstring for Agent"""

    def __init__(self, config):
        super(Agent, self).__init__()
        self.__current_channel = 0
        self.__action_move_channel = 0
        self.__action_send_packet = 1
        self.__Max_channel = config["max_channel"]
        self.__channel_policy = Channel_Baseline()
        self.__send_packet_policy = Send_Packet_Baseline()

    def update_current_channel(self, channel):
    	self.__current_channel = channel

    def random_policy(self):
        self.__action_move_channel = np.random.choice(self.__Max_channel)
        self.__action_send_packet = np.random.randint(2)

    def stay_policy(self):
        self.__action_move_channel = 0
        self.__action_send_packet = 1

    @property
    def cur_channel(self):
        return self.__current_channel

    @property
    def act_c(self):
        return self.__action_move_channel

    @property
    def act_s(self):
        return self.__action_send_packet

    @property
    def c_policy(self):
        return self.__channel_policy

    @property
    def s_policy(self):
        return self.__send_packet_policy



# Create the agent
agent = Agent(default_config)


# check & load pretrain model
# if os.path.isfile('pg_params.pkl'):
#     print('Load Baseline Network parametets ...')
#     Baseline.load_state_dict(torch.load('pg_params.pkl'))



# construct two optimal function
optimizer1 = optim.RMSprop(agent.c_policy.parameters(), default_config["learning_rate"], default_config["decay_rate"])
optimizer2 = optim.RMSprop(agent.s_policy.parameters(), default_config["learning_rate"], default_config["decay_rate"])





def finish_episode1():
    R = 0
    policy_loss = []
    value_loss = []
    rewards = []
    for r in agent.c_policy.rewards[::-1]:
        R = r + default_config["gamma"] * R
        rewards.insert(0, R)
    # turn rewards to pytorch tensor and standardize
    rewards = torch.Tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-6)

    for (log_prob, value), reward in zip(agent.c_policy.saved_log_probs, rewards):
        advantage = reward - value
        policy_loss.append(- log_prob * advantage)         # policy gradient
        value_loss.append(F.smooth_l1_loss(value[0][0], reward)) # value function approximation
        # print("TEST value[0][0]:", value[0][0])
        # print("TEST reward:", reward)
    optimizer1.zero_grad()
    # print("TEST policy_loss:", policy_loss)
    policy_loss = torch.stack(policy_loss).sum()
    value_loss = torch.stack(value_loss).sum()
    loss = policy_loss + value_loss

    loss.backward()
    optimizer1.step()

    # clean rewards and saved_actions
    del agent.c_policy.rewards[:]
    del agent.c_policy.saved_log_probs[:]






def finish_episode2():
    R = 0
    policy_loss = []
    value_loss = []
    rewards = []
    for r in agent.s_policy.rewards[::-1]:
        R = r + default_config["gamma"] * R
        rewards.insert(0, R)
    # turn rewards to pytorch tensor and standardize
    rewards = torch.Tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-6)

    for (log_prob, value), reward in zip(agent.s_policy.saved_log_probs, rewards):
        advantage = reward - value
        policy_loss.append(- log_prob * advantage)         # policy gradient
        value_loss.append(F.smooth_l1_loss(value[0][0], reward)) # value function approximation
    optimizer2.zero_grad()
    # print("TEST policy_loss:", policy_loss)
    policy_loss = torch.stack(policy_loss).sum()
    value_loss = torch.stack(value_loss).sum()
    loss = policy_loss + value_loss

    loss.backward()
    optimizer2.step()

    # clean rewards and saved_actions
    del agent.s_policy.rewards[:]
    del agent.s_policy.saved_log_probs[:]








# Main loop
if __name__ == '__main__':
    running_reward = None
    reward_sum = 0
    prev_x = None
    filename = './data/pgb_logs.txt'

    for i_episode in range(default_config["max_iteration"]):
        attack_mode = random.randint(0, 6)
        state_new = env.reset(attack_mode)
        agent.update_current_channel(state_new)
        done = False


        for t in range(default_config["max_episode_length"]):
            # Get current channel
            x = np.zeros(default_config["max_channel"])
            x[agent.cur_channel] = 1
            # Put into the NN
            action_c = agent.c_policy.select_action(x).cpu().detach().numpy()[0]
            action_s = agent.s_policy.select_action(x).cpu().detach().numpy()[0]
            state_new, reward, done, info = env.step(action_c, action_s)
            agent.update_current_channel(state_new)
            reward_sum += reward

            agent.c_policy.rewards.append(reward)
            agent.s_policy.rewards.append(reward)


            if done:
                # tracking log
                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                print('REINFORCE POLICY GRADIENT WITH BASELINE ep %03d done. reward: %f. reward running mean: %f' % (i_episode, reward_sum, running_reward))
                if i_episode % default_config["log_freq"] == 0:
                    with open(filename, 'a') as file_object:
                        file_object.write('REINFORCE POLICY GRADIENT WITH BASELINE ep %d done. reward: %f. reward running mean: %f\n' % (i_episode, reward_sum, running_reward))
                        file_object.close()

                reward_sum = 0
                break

        print('In Episode %d, The PSR is %f, the PDR is %f.' %(i_episode, info[0], info[1]))
        if i_episode % default_config["log_freq"] == 0:
            with open(filename, 'a') as file_object:
                file_object.write('In Episode %d, The PSR is %f, the PDR is %f.\n' %(i_episode, info[0], info[1]))
                file_object.close()

        # use baseline gradient update model weights
        if i_episode % default_config["batch_size"] == 0:
            finish_episode1()
            finish_episode2()

        # Save model in every 100 episode
        if i_episode % default_config["save_freq"] == 0:
            print('ep %d: model saving...' % (i_episode))
            torch.save(agent.c_policy.state_dict(), './data/pgb_switch_channel.pkl')
            torch.save(agent.s_policy.state_dict(), './data/pgb_send_packet.pkl')

    print("-------------------------------------------------------------------------------------------")
    print("Finish Training")
