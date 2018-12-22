import tensorflow as tf
import numpy as np
import gym



class AC:
	def __init__(self,alr,clr,n_features,n_actions):
		self.critic_learning_rate=clr
		self.actor_learning_rate=alr
		
		self.n_features=n_features
		self.n_actions=n_actions
		
		self.build_networks()#building networks before tf.session
		#replacing target with primary weights,but not sure how its assignment and not a function
		#self.replacing_target_parameter=[tf.assign(t,p) for t,p in zip(tf.get_collection('target_network_parameters'),tf.get_collection('primary_network_parameters'))]
		#self.target_params=[]
		#self.primary_params=[]

		self.sess=tf.Session()
		self.sess.run(tf.global_variables_initializer())
	def build_networks(self):
		# building a computation graph 
		
		
		
		
		#critic target ie Value is based on it
		######
		#here the value of state is a single number but if working i=with images the value of state might be bigger
		######
		seed=128

		# use this v_t for value of state and next stat
		###########################
		
		
		#critic primary net ie the network being trained for better state utility values
		
		self.sv=tf.placeholder(shape=[1,self.n_features],dtype=tf.float32,name='Input')

		self.w1= tf.Variable(tf.random_normal([self.n_features, 10], seed=seed))
		self.w2= tf.Variable(tf.random_normal([10, 5], seed=seed))
		self.w3= tf.Variable(tf.random_normal([5, 5], seed=seed))
		self.w4= tf.Variable(tf.random_normal([5, 5], seed=seed))
		self.w5= tf.Variable(tf.random_normal([5, 1], seed=seed))


		self.b1= tf.Variable(tf.random_normal([1,10], seed=seed))
		self.b2= tf.Variable(tf.random_normal([1,5], seed=seed))
		self.b3= tf.Variable(tf.random_normal([1,5], seed=seed))
		self.b4= tf.Variable(tf.random_normal([1,5], seed=seed))
		self.b5= tf.Variable(tf.random_normal([1,1], seed=seed))

		

		l1 = tf.add(tf.matmul(self.sv, self.w1),self.b1)
		l1 = tf.nn.relu(l1)

		l2 = tf.add(tf.matmul(l1, self.w2),self.b2)
		l2 = tf.nn.relu(l2)

		l3 = tf.add(tf.matmul(l2, self.w3),self.b3)
		l3 = tf.nn.relu(l3)

		l4 = tf.add(tf.matmul(l3, self.w4),self.b4)
		l4 = tf.nn.relu(l4)

		lout = tf.add(tf.matmul(l2, self.w5),self.b5)
		
		self.v_eval=lout
		# train this for better v eval values and then replace the params
		self.v_target=tf.placeholder(shape=[1,1],dtype=tf.float32,name='primary_output')#value target for the	nn to train
		self.cp_loss =tf.losses.mean_squared_error(self.v_target,self.v_eval)

		optimiser=tf.train.AdamOptimizer(learning_rate=self.critic_learning_rate ,name='rms')
		self.train_critic_primary=optimiser.minimize(self.cp_loss)	

		####################################################################

		#poilicy net
		self.s=tf.placeholder(shape=[1,self.n_features],dtype=tf.float32,name='Input_to_policy')

		self.w1p= tf.Variable(tf.random_normal([self.n_features, 5], seed=seed))
		self.w2p= tf.Variable(tf.random_normal([5, 5], seed=seed))
		self.w3p= tf.Variable(tf.random_normal([5, 5], seed=seed))
		self.w4p= tf.Variable(tf.random_normal([5, 5], seed=seed))
		self.w5p= tf.Variable(tf.random_normal([5,self.n_actions], seed=seed))


		self.b1p= tf.Variable(tf.random_normal([1,5], seed=seed))
		self.b2p= tf.Variable(tf.random_normal([1,5], seed=seed))
		self.b3p= tf.Variable(tf.random_normal([1,5], seed=seed))
		self.b4p= tf.Variable(tf.random_normal([1,5], seed=seed))
		self.b5p= tf.Variable(tf.random_normal([1,self.n_actions], seed=seed))

		


		l1p= tf.add(tf.matmul(self.s, self.w1p),self.b1p)
		l1p = tf.nn.relu(l1p)

		l2p = tf.add(tf.matmul(l1p, self.w2p),self.b2p)
		l2p = tf.nn.relu(l2p)

		l3p = tf.add(tf.matmul(l2p, self.w3p),self.b3p)
		l3p = tf.nn.relu(l3p)

		l4p = tf.add(tf.matmul(l3p, self.w4p),self.b4p)
		l4p = tf.nn.relu(l4p)

		loutp = tf.add(tf.matmul(l3p, self.w5p),self.b5p)
		loutp= tf.nn.relu(loutp)
		self.actions_probability=tf.nn.softmax(loutp)
		#self.actions_probability=tf.multinomial(logits=loutp,num_samples=1)

		#policy loss
		# feed state advantage and action(stochastic or deterministic)
		self.Advantage=tf.placeholder(shape=None,dtype=tf.float32,name='advantage')
		self.action_took=tf.placeholder(shape=None,dtype=tf.int32,name='stochastic_action_index')

		#self.policy_loss =tf.reduce_sum(tf.multiply(self.Advantage,tf.log(self.actions_probability[0,self.action_took])))
		self.loss= tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(self.action_took,self.n_actions),logits=self.actions_probability)
		self.policy_loss = tf.reduce_sum(self.Advantage*self.loss)
		#if using this minimise the +ve loss
		self.policy_train=tf.train.AdamOptimizer(self.actor_learning_rate).minimize(self.policy_loss)

	def take_action(self,s):
		action_p=self.sess.run(self.actions_probability,feed_dict={self.s: [s]})
		return action_p
	def train_actor(self,s,a,Ad):
		self.sess.run(self.policy_train,feed_dict={self.s:[s],self.Advantage:Ad,self.action_took:a})
		#policy_loss= self.sess.run(self.policy_loss,feed_dict={self.s:[s],self.Advantage:[Ad],self.action_took:a})
		#print("                          policy:",policy_loss)
		#print(self.w1.eval(self.sess))
	def train_primary_critic(self,s,v):
		self.sess.run(self.train_critic_primary,feed_dict={self.sv:[s],self.v_target:[v]})
		#cp_loss=self.sess.run(self.cp_loss,feed_dict={self.s:[s],self.v_target:v})
		#print("critic:",cp_loss)
	def get_value_of_state(self,s):
		v=self.sess.run(self.v_eval,feed_dict={self.sv:[s]})	
		return v.ravel()

	#def save(self,path):

	#def restore(self,path):
			

if __name__ == "__main__":
	env = gym.make('MountainCar-v0')
	#env=gym.unwrapped
	#self,alr,clr,n_features,n_actions
	ac_agent=AC(0.001,0.01,env.observation_space.shape[0],env.action_space.n)
	episodes = 10000000000000000000
	gamma=0.85
	total_steps=0
	obs=env.reset()
	
	steps=0
	obs,reward,terminate,_=env.step(1)
	episode_reward=0
	max_so_far=-1
	while True:
		env.render()
		action=ac_agent.take_action(obs)
		action=np.ravel(action)
		stochastic_action=np.random.choice(np.arange(action.shape[0]), p=action)
		obs_,reward,terminate,_=env.step(stochastic_action)
		#reward=abs(obs_[0]+0.5)
		reward=0
		if obs_[0]> max_so_far:
			reward=abs(10)
			max_so_far=obs_[0]
			#print("hit max")

		v_s= ac_agent.get_value_of_state(obs)
		v_s_n=ac_agent.get_value_of_state(obs_)
		
		v_target= reward + gamma*v_s_n
		#if terminate:
			
			
			#v_target= reward + 0*v_s_n 
			
		advantage= v_target-v_s
		advantage=np.ravel(advantage)
		#print(stochastic_action,steps,advantage)
		ac_agent.train_actor(obs,stochastic_action,advantage)
		ac_agent.train_primary_critic(obs,v_target)
		
		
		
		episode_reward+=reward
		
			
		obs=obs_
		total_steps+=1
		steps+=1
	
"""
if __name__ == "__main__":
	env = gym.make('CartPole-v0')
	#env=gym.unwrapped
	#self,alr,clr,n_features,n_actions
	
	ac_agent=AC(0.001,0.01,env.observation_space.shape[0],env.action_space.n)
	episodes = 10000000000000000000
	gamma=0.8
	
	
	for episode in range(episodes):
		
		terminate=False
	
		steps=0
		obs=env.reset()
		
		




		episode_reward=0
		while not terminate:
			env.render()
	
			#print(one_hot_action)

			action=ac_agent.take_action(obs)
			action=np.ravel(action)
			
			stochastic_action=np.random.choice(np.arange(action.shape[0]), p=action)
			#deterministic_action=ac_agent.custom_one_hot(np.argmax(action),env.action_space.n)
			



			obs_,reward,terminate,_=env.step(stochastic_action)
			



			if obs_[2] <0.01:
				reward=5
			elif terminate:
				reward=-5		
			else:
				reward=0
				
			
			v_s= ac_agent.get_value_of_state(obs)
			v_s_n=ac_agent.get_value_of_state(obs_)
			
			
			v_target= reward + gamma*(v_s_n)
			if terminate:
				
				
				v_target= reward + 0*v_s
				
			advantage= v_target-v_s#td error as advantage
			advantage=np.ravel(advantage)
			#print(advantage,deterministic_action)
			ac_agent.train_actor(obs,stochastic_action,advantage)
			ac_agent.train_primary_critic(obs,v_target)
			#print("veval",v_s,"v_target",v_target)
			#print("action",stochastic_action,"rew",advantage)	
			episode_reward+=reward
			obs=obs_
	
			
			steps+=1
		print (episode,episode_reward)
"""