import numpy as np

class env(object):
	"""docstring for env"""
	def __init__(self, MAX_Channel, Total_package):
		super(env, self).__init__()
		self.__MAX_Channel = MAX_Channel
		self.__Total_package = Total_package
		self.__channels = np.zeros(MAX_Channel)

		self.__current_state = 0 # Default current_state is 0
		self.__time = 0 # Reset time
		self.__attack_mode = attack_mode # Set attacker's mode
		self.__attacked_channels = []
		self.__sent_packages = 0		
		self.__ACK_sent = [False]






	def reset(self, attack_mode):
		self.__current_state = 0 # Default current_state is 0
		self.__time = 0 # Reset time
		print("Curretn Time is ", self.__time)
		self.__attack_mode = attack_mode # Set attacker's mode
		self.__attacked_channels = []
		self.__sent_packages = 0	
		self.__ACK_sent = [False]







	def step(self, action):

		if (self.__current_state + action > self.__MAX_Channel) || (self.__current_state + action < 0):
			print("Error in action")
			return

		self.__time += 1
		print("Curretn Time is ", self.__time)

		# Opponent attack
		self.__opponent_attack(self.__attack_mode)

		# Agent changes the channel and sends package
		self.__current_state += action
		new_state = self.__current_state
		self.__send_package()

		# reward
		if(self.__ACK_sent[self.__time - 1]): # whether receive the ACK (sent from t - 1)
			self.__sent_packages++
			reward = 1
		else:
			reward = 0

		# done
		if(self.__sent_packages == self.__Total_package):
			done = True
		else:
			done = False

		return new_state, reward, done, information




	@property
	def act_dim(self):
		return self.__MAX_Channel
	

	@property
	def obs_dim(self):
		return self.__MAX_Channel




	def __attack(self):
		for i in range(len.(self.__MAX_Channel)):
			if i in self.__attacked_channels:
				self.__channels[i] = 1 
			else:
				self.__channels[i] = 0




	def __send_package(self):
		if self.__channels[self.current_state] == 0: # Successfully send package
			self.__ACK_sent.append(True)
		else: # The channal has been occupied
			self.__ACK_sent.append(False)




	def __opponent_attack(self, mode = 0):
		# Modify the self.__attacked_channels list
		if mode == 0:
			# Randomly choose only ONE channel to attack
			self.__attacked_channels.clear()
			self.__attacked_channels.append(np.random.choice(env.MAX_Channel))
			self.__attack()
			return
		else if mode == 1:
			return