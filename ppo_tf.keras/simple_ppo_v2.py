# the current policy which we use to sample data , the sampled data automatically becomes the old policy data once we start training 
#the policy net for more than 1 epoch  
import tensorflow as tf
import numpy as np
import gym


#env=gym.make('Acrobot-v1')
env=gym.make('CartPole-v1')
#env=gym.make('MountainCar-v0')
env=env.unwrapped#removes step restriction

s_dim = env.observation_space.shape[0]
print(s_dim)
a_dim = env.action_space.n
print(a_dim)
#a_bound = env.action_space.high
#print(a_bound)
DUMMY_ACTION, DUMMY_VALUE = np.zeros((1,a_dim)), np.zeros((1, 1))



state_inputs = tf.keras.Input(shape=(s_dim,), name='state')
advantage = tf.keras.Input(shape=(1, ), name="Advantage")
old_prediction = tf.keras.Input(shape=(a_dim,), name="Old_Prediction")
x = tf.keras.layers.Dense(64, activation='relu')(state_inputs)
x = tf.keras.layers.Dense(64, activation='relu')(x)
action_outputs = tf.keras.layers.Dense(a_dim, activation='softmax')(x)


def proximal_policy_optimization_loss(advantage, old_prediction):
	loss_clipping = 0.2
	entropy_loss = 0.01
	#y_true = one hot actions , y_pred = prob output
	def loss(y_true, y_pred):
		prob = y_true * y_pred
		old_prob = y_true * old_prediction
		r = prob / (old_prob + 1e-10)
		loss = -tf.keras.backend.mean(tf.keras.backend.minimum(r * advantage, tf.keras.backend.clip(r, min_value=1 - loss_clipping,max_value=1 + loss_clipping) * advantage) + entropy_loss * (prob * tf.keras.backend.log(prob + 1e-10)))
		return loss
	return loss	
policy= tf.keras.Model(inputs=[state_inputs, advantage, old_prediction], outputs=[action_outputs], name='p_actor_model')
policy.compile(loss=proximal_policy_optimization_loss(advantage=advantage,old_prediction=old_prediction), optimizer=tf.keras.optimizers.Adam(lr=0.0001))# custom lAdam(lr=0.0001) to be defined
policy.summary()
x = tf.keras.layers.Dense(64, activation='relu')(state_inputs)
x = tf.keras.layers.Dense(64, activation='relu')(x)
value_outputs = tf.keras.layers.Dense(1, activation=None)(x)
critic= tf.keras.Model(inputs=state_inputs, outputs=value_outputs, name='p_critic_model')
critic.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
critic.summary()


def max(a,b):
	if a>b:
		return a
	else:
		return b
def abs(a):
	if a>=0:
		return a
	else:
		return -a


class Memory:
	def __init__(self):
		self.batch_s = []
		self.batch_a = []
		self.batch_r = []
		self.batch_s_ = []
		self.batch_done = []

	def store(self, s, a, s_, r, done):
		self.batch_s.append(s)
		self.batch_a.append(a)
		self.batch_r.append(r)
		self.batch_s_.append(s_)
		self.batch_done.append(done)

	def clear(self):
		self.batch_s.clear()
		self.batch_a.clear()
		self.batch_r.clear()
		self.batch_s_.clear()
		self.batch_done.clear()
	def cnt_samples(self):
		return len(self.batch_s)
def onehot(a,s):
	i = np.zeros(a_dim)
	i[a]=1
	return i
def save_weights():
	actorpath=r"C:\\Users\\Dell\\Desktop\\holidASY\\my_ppo\\ppo_tf.keras\\ppo_simple_actor.h5"
	criticpath=r"C:\\Users\\Dell\\Desktop\\holidASY\\my_ppo\\ppo_tf.keras\\ppo_simple_critic.h5"
	policy.save_weights(actorpath)
	critic.save_weights(criticpath)
	print("saved")
def load_weights():
	actorpath=r"C:\\Users\\Dell\\Desktop\\holidASY\\my_ppo\\ppo_tf.keras\\ppo_simple_actor.h5"
	criticpath=r"C:\\Users\\Dell\\Desktop\\holidASY\\my_ppo\\ppo_tf.keras\\ppo_simple_critic.h5"
	policy.load_weights(actorpath)
	critic.load_weights(criticpath)
	print("loaded")

def gae_calc(val,val_,rew,done):
	mask=1 
	gae=0
	gamma=0.99
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
	



episodes = 2000
steps = 200
memory=Memory()
render=0
s=env.reset()
for episode in range(1,episodes):
	
	if episode>220:
		render=1
	for step in range(steps):
		if render:
			env.render()
		pred_action = policy.predict([np.array([s]),DUMMY_VALUE,DUMMY_ACTION])# prob distribution
		action = np.random.choice(np.arange(pred_action.shape[1]), p=pred_action.ravel())# action chosen
		#random_action= np.random.choice(np.arange(pred_action.shape[1]))
		#if steps% 3 == 0 and episode<50:
		#	action = random_action
		action_one_hot=onehot(action,a_dim)# acton matrix
		s_, reward, done, info = env.step(action)
		memory.store(s.ravel(),action_one_hot.ravel(),s_.ravel(),reward,done)# s, a, s_, r, done
		if done:
			s_=env.reset()

		s=s_
	# updation
	obs =np.array( memory.batch_s)
	values = critic.predict(np.array(memory.batch_s))
	values_ = critic.predict(np.array(memory.batch_s_))
	returns = gae_calc(values,values_,memory.batch_r,memory.batch_done)	
	advantage=returns-values
	old_Prediction=memory.batch_a##########################this is wrong since its not action probablity under current policy
	old_Prediction=np.array(old_Prediction)
	action=np.array(memory.batch_a)########################
	print(episode)
	policy.fit(x=[obs,advantage, old_Prediction],y=action,batch_size=200,shuffle=True, epochs=15, verbose=False)
	critic.fit([obs],[returns], batch_size=200, shuffle=True, epochs=15, verbose=False)
	#print(actor_loss,critic_loss)
	memory.clear()
