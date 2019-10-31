import tensorflow as tf
import numpy as np
import gym
#import tensorflow_probability as tfp

class ppo():
	def __init__(self,name,s_dim,a_dim,memory,a_bound):
		self.s_dim = s_dim
		self.a_dim =a_dim
		self.memory = memory
		self.a_bound =a_bound
		self.name = name
		self.policy  = self.make_policy()
		self.critic = self.make_critic()
		
	def make_policy(self):
			

		state_inputs = tf.keras.Input(shape=(self.s_dim,), name='state')
		advantage = tf.keras.Input(shape=(1, ), name="Advantage")
		action= tf.keras.Input(shape=(self.a_dim,), name="action")
		x = tf.keras.layers.Dense(16, activation='relu')(state_inputs)
		#x = tf.keras.layers.Dense(64, activation='relu')(x)
		#x = tf.keras.layers.Dense(64, activation='relu')(x)
		x1 = tf.keras.layers.Dense(16, activation='relu')(x)

		mu_0 = tf.keras.layers.Dense(self.a_dim, activation='tanh')(x1)
		x2 = tf.keras.layers.Dense(16, activation='relu')(x)
		sigma_0 = tf.keras.layers.Dense(self.a_dim, activation='softplus')(x2)

		mu = tf.keras.layers.Lambda(lambda x: x * self.a_bound)(mu_0)
		sigma = tf.keras.layers.Lambda(lambda x: x + 0.0001)(sigma_0)
		musig=tf.keras.layers.concatenate([mu,sigma])
		def proximal_policy_optimization_loss(advantage, action):
			loss_clipping = 0.2
			entropy_loss = 0.0
			pi=3.1415926
			def loss(y_true, y_pred):
				mu=tf.keras.backend.expand_dims(y_pred[:,0],1)
				sigma = tf.keras.backend.expand_dims(y_pred[:,1],1)
				old_mu=tf.keras.backend.expand_dims(y_true[:,0],1)
				old_sigma = tf.keras.backend.expand_dims(y_true[:,1],1)
				sigma_sq=tf.keras.backend.square(sigma)
				old_sigma_sq=tf.keras.backend.square(old_sigma)
				pdf = 1. / tf.keras.backend.sqrt(2. *pi* sigma_sq) * tf.keras.backend.exp(-tf.keras.backend.square(action - mu) / (2. * sigma_sq))
				log_pdf = tf.keras.backend.log(pdf + tf.keras.backend.epsilon())
				old_pdf = 1. / tf.keras.backend.sqrt(2. *pi* old_sigma_sq) * tf.keras.backend.exp(-tf.keras.backend.square(action - old_mu) / (2. * old_sigma_sq))
				old_log_pdf = tf.keras.backend.log(old_pdf + tf.keras.backend.epsilon() )
				entropy =  tf.keras.backend.sum(0.5 * (tf.keras.backend.log(2. * pi * sigma_sq) + 1.))
				#acloss = -tf.keras.backend.sum(advantage*log_pdf + entropy_loss*entropy)

				#r =  pdf/ (old_pdf+ 1e-10)
				r = tf.keras.backend.exp(log_pdf- old_log_pdf)
				loss = -tf.keras.backend.mean(tf.keras.backend.minimum(r * advantage, tf.keras.backend.clip(r, min_value=1 - loss_clipping,max_value=1 + loss_clipping) * advantage)) + entropy_loss *entropy
				return loss
			return loss	
		policy= tf.keras.Model(inputs=(state_inputs, advantage,action), outputs=(musig), name='p_actor_model')
		policy.compile(loss=proximal_policy_optimization_loss(advantage=advantage,action=action), optimizer=tf.keras.optimizers.Adam(lr=0.0001))
		return policy

	def make_critic(self):
		state_inputs = tf.keras.Input(shape=(self.s_dim,), name='state')

		x = tf.keras.layers.Dense(16, activation='relu')(state_inputs)
		x = tf.keras.layers.Dense(16, activation='relu')(x)
		value_outputs = tf.keras.layers.Dense(1, activation=None)(x)
		critic= tf.keras.Model(inputs=state_inputs, outputs=value_outputs, name='p_critic_model')
		critic.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
		return critic
	def save_weights(self):
		actorpath=r"C:\\Users\\Dell\\Desktop\\this_year\\SCM-RL\\ppo_continous_"+self.name+"actor.h5"
		criticpath=r"C:\\Users\\Dell\\Desktop\\this_year\\SCM-RL\\ppo_continous_"+self.name+"critic.h5"
		self.policy.save_weights(actorpath)
		self.critic.save_weights(criticpath)
		print("saved")
	def load_weights(self):
		actorpath=r"C:\\Users\\Dell\\Desktop\\this_year\\SCM-RL\\ppo_continous_"+self.name+"actor.h5"
		criticpath=r"C:\\Users\\Dell\\Desktop\\this_year\\SCM-RL\\ppo_continous_"+self.name+"critic.h5"
		self.policy.load_weights(actorpath)
		self.critic.load_weights(criticpath)
		print("loaded")

	def gae_calc(self,val,val_,rew,done):
		mask=1 
		gae=0
		gamma=0.95
		lambd = 0.95
		returns=np.zeros_like(val)
		for i in reversed(range(0,len(val))):
			mask=1
			if done[i]:
				mask = 0 	
			delta=rew[i]+gamma*val_[i]*mask - val[i]
			gae=delta+gamma*lambd*mask*gae
			returns[i]=gae+val[i]
		return returns
		
	def adv_calc(self,val,val_,rew,done):
		gamma=0.99
		returns=np.zeros_like(val)
		for i in range(0,len(val)):
			returns[i] = rew[i] + (1- done[i])*val_[i]*gamma
		return returns
	def train(self,batch=512,epochs=10):
		obs =np.array( self.memory.batch_s)
		values = self.critic.predict(np.array(self.memory.batch_s))
		values_ = self.critic.predict(np.array(self.memory.batch_s_))
		returns = self.adv_calc(values,values_,self.memory.batch_r,self.memory.batch_done)	
		advantage=returns-values
		Action=np.array(self.memory.batch_a)
		Old_Prediction_musig =np.array(self.memory.musig) 
		self.policy.fit(x=(obs,advantage,Action),y=(Old_Prediction_musig),batch_size=batch,shuffle=True, epochs=epochs, verbose=False)
		self.critic.fit([obs],[returns], batch_size=batch, shuffle=True, epochs=epochs, verbose=False)
		self.memory.clear()




class Memory:
	def __init__(self):
		self.batch_s = []
		self.batch_a = []
		self.batch_r = []
		self.batch_s_ = []
		self.batch_done = []
		self.musig = []
	def store(self, s, a, s_, r, done,musig):
		self.batch_s.append(s)
		self.batch_a.append(a)
		self.batch_r.append(r)
		self.batch_s_.append(s_)
		self.batch_done.append(done)
		self.musig.append(musig)
	def clear(self):
		self.batch_s.clear()
		self.batch_a.clear()
		self.batch_r.clear()
		self.batch_s_.clear()
		self.batch_done.clear()
		self.musig.clear()
	def cnt_samples(self):
		return len(self.batch_s)




env1=gym.make('MountainCarContinuous-v0')

env1=env1.unwrapped

s_dim1 = env1.observation_space.shape[0]
print(s_dim1)
a_dim1 = env1.action_space.shape[0]
print(a_dim1)
a_bound1 = env1.action_space.high[0]
print(a_bound1)
DUMMY_ACTION1, DUMMY_VALUE1 = np.zeros((1,a_dim1)), np.zeros((1, 1))

memory_1=Memory()
agent_1 =  ppo(name = "ppo_agent_01",s_dim=s_dim1 ,a_dim= a_dim1,memory = memory_1,a_bound=a_bound1)



env2=gym.make('Pendulum-v0')
#env2=gym.make('MountainCarContinuous-v0')

env2=env2.unwrapped#removes step restriction

s_dim2 = env2.observation_space.shape[0]
print(s_dim2)
a_dim2 = env2.action_space.shape[0]
print(a_dim2)
a_bound2 = env2.action_space.high[0]
print(a_bound2)
DUMMY_ACTION2, DUMMY_VALUE2 = np.zeros((1,a_dim2)), np.zeros((1, 1))

memory_2=Memory()
agent_2 =  ppo(name = "ppo_agent_02",s_dim=s_dim2 ,a_dim= a_dim2,memory = memory_2,a_bound=a_bound2)






episodes = 20000000
steps = 3000
render=0
var=1# need decaying exploration noise to solve this problem 




for episode in range(1,episodes):
	s1=env1.reset()
	rews1 = 0	
	s2=env2.reset()
	rews2 = 0	
	if episode > 1000:
		render=1
	if var<0.1:
		var=0

	for step in range(steps):
		if render:
			env1.render()
			env2.render()

		out1 = agent_1.policy.predict((np.array([s1]),DUMMY_VALUE1,DUMMY_ACTION1))	
		mu_pred1,sigma_pred1 =out1[0][0],out1[0][1]
		action1= np.random.normal(mu_pred1, sigma_pred1,a_dim1)
		e1=np.random.normal(action1,var,size=(1,a_dim1))[0] # need exploration
		a1 =np.clip(e1,-a_bound1,a_bound1 )
		s_1, reward1, done1, info1 = env1.step(a1)
		
		out2 = agent_2.policy.predict((np.array([s2]),DUMMY_VALUE2,DUMMY_ACTION2))	
		mu_pred2,sigma_pred2 =out2[0][0],out2[0][1]
		action2= np.random.normal(mu_pred2, sigma_pred2,a_dim2)
		a2  = action2 # no exploration here
		s_2, reward2, done2, info2 = env2.step(a2)



		if done1:
			reward1=1000 + reward1
			print("goal reached!!")
			var*=0.995
		agent_1.memory.store(s1.ravel(),a1,s_1.ravel(),reward1,done1,out1[0])# s, a, s_1, r, done1,musig
		rews1+=reward1


		if step%500==0:
			done2 = True
		agent_2.memory.store(s2.ravel(),a2,s_2.ravel(),reward2,done2,out2[0])# s, a, s_2, r, done2,musig
		rews2+=reward2
		if done1:
			s_1=env1.reset()
		if done2:

			s_2=env2.reset()
			
		s1 = s_1
		s2  = s_2
	# updation
	print("env1[mcc] | "+str(episode)+" | "+str(rews1)+" | "+ str(var))
	print("env2[pend] | "+str(episode)+" | "+str(rews2)+" | "+ str(0))
	print("_"*100)
	agent_1.train()
	agent_2.train(batch=64)