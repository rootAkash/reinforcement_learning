import tensorflow as tf
import numpy as np
import gym



class DQN:
	def __init__(self,lr,gamma,n_features,n_actions,epsilon,parameter_changing_pointer,memory_size):
		self.learning_rate=lr
		self.gamma=gamma
		self.n_features=n_features
		self.n_actions=n_actions
		self.epsilon=epsilon
		self.batch_size=100
		self.experience_counter=0 # initializing to 0
		self.experience_limit=memory_size

		self.replace_target_pointer=parameter_changing_pointer
		self.learning_counter=0
		self.memory=np.zeros([self.experience_limit,self.n_features*2+2])
		self.build_networks()#building networks before tf.session
		#replacing target with primary weights,but not sure how its assignment and not a function
		#self.replacing_target_parameter=[tf.assign(t,p) for t,p in zip(tf.get_collection('target_network_parameters'),tf.get_collection('primary_network_parameters'))]
		#self.target_params=[]
		#self.primary_params=[]
		####################
		

		self.sess=tf.Session()
		self.sess.run(tf.global_variables_initializer())
	def build_networks(self):
		# building a computation graph 
		#primary
		
		self.s=tf.placeholder(shape=[None,self.n_features],dtype=tf.float32,name='Input')
		self.q_target=tf.placeholder(shape=[None,self.n_actions],dtype=tf.float32,name='primary_output')#target for the	nn to train
		#with tf.variable_scope('primary_network'):
		#c=['primary_network_parameters',tf.GraphKeys.GLOBAL_VARIABLES]
		#with tf.variable_scope('layer1')
		#target ie action is based on it
		seed=128
		self.w11= tf.Variable(tf.random_normal([self.n_features, 5], seed=seed))
		self.w21= tf.Variable(tf.random_normal([5, 5], seed=seed))
		self.w31= tf.Variable(tf.random_normal([5, 5], seed=seed))
		self.w41= tf.Variable(tf.random_normal([5, 5], seed=seed))
		self.w51= tf.Variable(tf.random_normal([5, self.n_actions], seed=seed))


		self.b11= tf.Variable(tf.random_normal([1,5], seed=seed))
		self.b21= tf.Variable(tf.random_normal([1,5], seed=seed))
		self.b31= tf.Variable(tf.random_normal([1,5], seed=seed))
		self.b41= tf.Variable(tf.random_normal([1,5], seed=seed))
		self.b51= tf.Variable(tf.random_normal([1,self.n_actions], seed=seed))

		self.target_params=[self.w11,self.w21,self.w31,self.w41,self.w51,self.b11,self.b21,self.b31,self.b41,self.b51]



		l11= tf.add(tf.matmul(self.s, self.w11),self.b11)
		l11 = tf.nn.relu(l11)

		l21 = tf.add(tf.matmul(l11, self.w21),self.b21)
		l21 = tf.nn.relu(l21)

		l31 = tf.add(tf.matmul(l21, self.w31),self.b31)
		l31 = tf.nn.relu(l31)

		l41 = tf.add(tf.matmul(l31, self.w41),self.b41)
		l41 = tf.nn.relu(l41)

		lout1 = tf.add(tf.matmul(l41, self.w51),self.b51)
		self.q_t=lout1

		#primary net ie the network being trained
		self.w1= tf.Variable(tf.random_normal([self.n_features, 5], seed=seed))
		self.w2= tf.Variable(tf.random_normal([5, 5], seed=seed))
		self.w3= tf.Variable(tf.random_normal([5, 5], seed=seed))
		self.w4= tf.Variable(tf.random_normal([5, 5], seed=seed))
		self.w5= tf.Variable(tf.random_normal([5, self.n_actions], seed=seed))


		self.b1= tf.Variable(tf.random_normal([1,5], seed=seed))
		self.b2= tf.Variable(tf.random_normal([1,5], seed=seed))
		self.b3= tf.Variable(tf.random_normal([1,5], seed=seed))
		self.b4= tf.Variable(tf.random_normal([1,5], seed=seed))
		self.b5= tf.Variable(tf.random_normal([1,self.n_actions], seed=seed))

		self.primary_params=[self.w1,self.w2,self.w3,self.w4,self.w5,self.b1,self.b2,self.b3,self.b4,self.b5]



		l1 = tf.add(tf.matmul(self.s, self.w1),self.b1)
		l1 = tf.nn.relu(l1)

		l2 = tf.add(tf.matmul(l1, self.w2),self.b2)
		l2 = tf.nn.relu(l2)

		l3 = tf.add(tf.matmul(l2, self.w3),self.b3)
		l3 = tf.nn.relu(l3)

		l4 = tf.add(tf.matmul(l3, self.w4),self.b4)
		l4 = tf.nn.relu(l4)

		lout = tf.add(tf.matmul(l4, self.w5),self.b5)
		#lout = tf.nn.sigmoid(lout)
		self.q_eval=lout
		#self.loss=tf.math.reduce_mean(tf.squared_difference(q_target,q_eval))
		self.loss =tf.losses.mean_squared_error(self.q_target,self.q_eval)

		optimiser=tf.train.RMSPropOptimizer(learning_rate=self.learning_rate ,name='rms')
		self.train_op=optimiser.minimize(self.loss)	

	def target_params_replace(self):
		#self.sess.run(t.assign(p) for t,p in enumerate(zip(self.target_params,self.primary_params)))	
		for i in range(len(self.target_params)):
			self.sess.run(self.target_params[i].assign(self.primary_params[i]))
	def store_experience(self,obs,a,r,obs_):
		index = self.experience_counter%self.experience_limit# reusability of memory
		self.memory[index,:]=np.hstack((obs,[a,r],obs_))
		self.experience_counter+=1	
	def fit(self):
		if self.experience_counter<self.experience_limit:
			#to prevent acessing garbage values	
			indices=np.random.choice(self.experience_counter,size=self.batch_size)
			# choosing random batchsize memory below experience_counter
			# if experice counter is less than batch then reduncancy will occur but is not a problem here
		else:
			indices=np.random.choice(self.experience_limit,size=self.batch_size)
		batch=self.memory[indices,:]
		state=batch[:,:self.n_features]
		actions=batch[:,self.n_features]#astype(int) ie kind of argmax
		rewards=batch[:,self.n_features+1]
		next_state=batch[:,self.n_features+2:]
		q_eval=self.sess.run(self.q_eval,feed_dict={self.s:state})
		qtaget=q_eval.copy()#qtaget=q_eval.copy()
		qt=self.sess.run(self.q_t,feed_dict={self.s:next_state})
		

		
		for i in range (len(batch)):
			a= int(actions[i])
			qtaget[i,a]=rewards[i]+self.gamma*max(qt[i])
		
		

		self.sess.run(self.train_op,feed_dict={self.s:state,self.q_target:qtaget})

		if self.epsilon<0.9:
			self.epsilon+=0.002
		if self.learning_counter%self.replace_target_pointer==0:
			self.target_params_replace()
			print("target params changed")
		self.learning_counter+=1
	def epsilon_greedy(self,obs):
		if np.random.uniform(0,1)<self.epsilon:
			# increses action by nn as episode moves forward
			action_to_be_taken=self.sess.run(self.q_eval,feed_dict={self.s:[obs]})	#increases dimension by 1
			#print(np.argmax(action_to_be_taken[0]))
			return np.argmax(action_to_be_taken[0])
		else:

			action_to_be_taken=np.random.choice(self.n_actions)
			#print(action_to_be_taken)
			return action_to_be_taken		
			#action_to_be_taken=self.sess.run(self.q_eval,feed_dict={self.s:[obs]})	#increases dimension by 1
			#print(np.argmax(action_to_be_taken[0]))
			#return np.argmax(action_to_be_taken[0])
	#def save(self,path):

	#def restore(self,path):
			

if __name__ == "__main__":
	#env = gym.make('MountainCar-v0')
	env = gym.make('CartPole-v0')
	#env=gym.unwrapped
	dqn=DQN(0.001,0.9,env.observation_space.shape[0],env.action_space.n,0.0,500,10000)
	episodes = 10000
	total_steps=0
	for episode in range(episodes):
		steps=0
		obs=env.reset()
		episode_reward=0
		while True:
			env.render()
			action=dqn.epsilon_greedy(obs)
			obs_,reward,terminate,_=env.step(action)
			#reward=abs(obs_[0]+0.5)
			reward= 10/(1+abs(obs_[2]))
			dqn.store_experience(obs,action,reward,obs_)
			if total_steps>200:
				dqn.fit()
			episode_reward+=reward
			if terminate:
				break
			obs=obs_
			total_steps+=1
			steps+=1
		print("episode {} with reward = {} at epsilon {} in steps {}".format(episode+1,episode_reward,dqn.epsilon,steps))
	while True:
		env.render()							