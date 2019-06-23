import tensorflow as tf
import numpy as np
import gym
#import tensorflow_probability as tfp



env=gym.make('MountainCarContinuous-v0')

env=env.unwrapped

s_dim = env.observation_space.shape[0]
print(s_dim)
a_dim = env.action_space.shape[0]
print(a_dim)
a_bound = env.action_space.high[0]
print(a_bound)
DUMMY_ACTION, DUMMY_VALUE = np.zeros((1,a_dim)), np.zeros((1, 1))



state_inputs = tf.keras.Input(shape=(s_dim,), name='state')
advantage = tf.keras.Input(shape=(1, ), name="Advantage")
action= tf.keras.Input(shape=(a_dim,), name="action")
x = tf.keras.layers.Dense(16, activation='relu')(state_inputs)
#x = tf.keras.layers.Dense(64, activation='relu')(x)
#x = tf.keras.layers.Dense(64, activation='relu')(x)
x1 = tf.keras.layers.Dense(16, activation='relu')(x)

mu_0 = tf.keras.layers.Dense(a_dim, activation='tanh')(x1)
x2 = tf.keras.layers.Dense(16, activation='relu')(x)
sigma_0 = tf.keras.layers.Dense(a_dim, activation='softplus')(x2)

mu = tf.keras.layers.Lambda(lambda x: x * a_bound)(mu_0)
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
policy.summary()





x = tf.keras.layers.Dense(16, activation='relu')(state_inputs)
x = tf.keras.layers.Dense(16, activation='relu')(x)
value_outputs = tf.keras.layers.Dense(1, activation=None)(x)
critic= tf.keras.Model(inputs=state_inputs, outputs=value_outputs, name='p_critic_model')
critic.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
critic.summary()



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

def save_weights():
	actorpath=r"C:\\Users\\Dell\\Desktop\\holidASY\\my_ppo\\ppo_tf.keras\\ppo_continous_actor.h5"
	criticpath=r"C:\\Users\\Dell\\Desktop\\holidASY\\my_ppo\\ppo_tf.keras\\ppo_continous_critic.h5"
	policy.save_weights(actorpath)
	critic.save_weights(criticpath)
	print("saved")
def load_weights():
	actorpath=r"C:\\Users\\Dell\\Desktop\\holidASY\\my_ppo\\ppo_tf.keras\\ppo_continous_actor.h5"
	criticpath=r"C:\\Users\\Dell\\Desktop\\holidASY\\my_ppo\\ppo_tf.keras\\ppo_continous_critic.h5"
	policy.load_weights(actorpath)
	critic.load_weights(criticpath)
	print("loaded")

def gae_calc(val,val_,rew,done):
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
	
def adv_calc(val,val_,rew,done):
	gamma=0.99
	returns=np.zeros_like(val)
	for i in range(0,len(val)):
		returns[i] = rew[i] + (1- done[i])*val_[i]*gamma
	return returns
	


episodes = 20000000
steps = 3000
memory=Memory()
render=0
var=1# need decaying exploration noise to solve this problem 
for episode in range(1,episodes):
	s=env.reset()
	rews = 0	
	if episode > 1000:
		render=1
	if var<0.1:
		var=0

	for step in range(steps):
		if render:
			env.render()

		out =policy.predict((np.array([s]),DUMMY_VALUE,DUMMY_ACTION))	
		mu_pred,sigma_pred =out[0][0],out[0][1]
		#print(mu_pred,sigma_pred)
		action= np.random.normal(mu_pred, sigma_pred,a_dim)
		e=np.random.normal(action,var,size=(1,a_dim))[0]
		a =np.clip(e,-1,1 )*a_bound
		#action=np.clip(action,-1,1)
		#print(action)
		s_, reward, done, info = env.step(a)
		if done:
			reward=1000 + reward
			print("goal reached!!")
			var*=0.995
		memory.store(s.ravel(),a,s_.ravel(),reward,done,out[0])# s, a, s_, r, done,musig
		rews+=reward
		if done:
			s_=env.reset()

		s=s_
		
	# updation
	obs =np.array( memory.batch_s)
	values = critic.predict(np.array(memory.batch_s))
	values_ = critic.predict(np.array(memory.batch_s_))
	returns = adv_calc(values,values_,memory.batch_r,memory.batch_done)	
	advantage=returns-values
	Action=np.array(memory.batch_a)
	Old_Prediction_musig =np.array(memory.musig) 
	#print(np.shape(Action))
	#print(Old_Prediction_musig)
	print(str(episode)+" | "+str(rews)+" | "+ str(var))
	policy.fit(x=(obs,advantage,Action),y=(Old_Prediction_musig),batch_size=512,shuffle=True, epochs=10, verbose=False)
	# big batch_size and more epochs lead to a policy thats initially good but further training makes it bad 
	critic.fit([obs],[returns], batch_size=512, shuffle=True, epochs=10, verbose=False)
	#print(actor_loss,critic_loss)
	#var*=0.99
	memory.clear()
