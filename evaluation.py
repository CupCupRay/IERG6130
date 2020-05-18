import numpy as np
import random
from environment import Env
import pg_train as pg
import pgb_train as pgb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

default_config = dict(
    # Environment hyper-parameters
    env_name = "Jamming Attack",
    max_iteration = 1000,
    max_episode_length = 1000,
    evaluate_interval = 100,
    max_channel = 100,
    # Test Sensitivity to Hyper-parameters
    # max_channel = 200,
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
    # Evaluation
    test_mode = "pg",
)


env = Env(default_config)
# torch.manual_seed(default_config["seed"])


# Create the agent



# check & load pretrain model
if default_config["test_mode"] = "pg":
	agent = pg.Agent(default_config)
	if os.path.isfile('./data/pg_send_packet.pkl'):
    	print('Load Policy Network parametets ...')
    	agent.s_policy.load_state_dict(torch.load('./data/pg_send_packet.pkl'))
    if os.path.isfile('./data/pg_switch_channel.pkl'):
    	print('Load Policy Network parametets ...')
    	agent.c_policy.load_state_dict(torch.load('./data/pg_switch_channel.pkl'))

elif default_config["test_mode"] = "pgb":
	agent = pgb.Agent(default_config)
	if os.path.isfile('./data/pgb_send_packet.pkl'):
    	print('Load Baseline Network parametets ...')
    	agent.s_policy.load_state_dict(torch.load('./data/pgb_send_packet.pkl'))
    if os.path.isfile('./data/pgb_switch_channel.pkl'):
    	print('Load Baseline Network parametets ...')
    	agent.c_policy.load_state_dict(torch.load('./data/pgb_switch_channel.pkl'))

else:
	print("ERROR IN TEST MODE!")



# Main loop
if __name__ == '__main__':
    running_reward = None
    reward_sum = 0
    prev_x = None
    filename = './data/evaluation_logs.txt'

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
            # print(int(action_c), " ", int(action_s))
            state_new, reward, done, info = env.step(int(action_c), int(action_s))
            agent.update_current_channel(state_new)
            reward_sum += reward

            agent.c_policy.rewards.append(reward)
            agent.s_policy.rewards.append(reward)


            if done:
                # tracking log
                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                print('REINFORCE ep %03d done. reward: %f. reward running mean: %f' % (i_episode, reward_sum, running_reward))
                if i_episode % default_config["log_freq"] == 0:
                    with open(filename, 'a') as file_object:
                        file_object.write('REINFORCE ep %d done. reward: %f. reward running mean: %f\n' % (i_episode, reward_sum, running_reward))
                        file_object.close()

                reward_sum = 0
                break

        print('In Episode %d, The PSR is %f, the PDR is %f.' %(i_episode, info[0], info[1]))
        if i_episode % default_config["log_freq"] == 0:
            with open(filename, 'a') as file_object:
                file_object.write('In Episode %d, The PSR is %f, the PDR is %f.\n' %(i_episode, info[0], info[1]))
                file_object.close()

        # use policy gradient update model weights
        # if i_episode % default_config["batch_size"] == 0:
        #     finish_episode1()
        #     finish_episode2()

        # Save model in every 100 episode
        # if i_episode % default_config["save_freq"] == 0:
        #     print('ep %d: model saving...' % (i_episode))
        #     torch.save(agent.c_policy.state_dict(), './data/pg_switch_channel.pkl')
        #     torch.save(agent.s_policy.state_dict(), './data/pg_send_packet.pkl')

    print("-------------------------------------------------------------------------------------------")
    print("Finish Training")
