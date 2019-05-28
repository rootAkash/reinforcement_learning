"""
here curiosity by random network distillation is used for exploration .
epsilon greedy is not used for exploration.
And the policy rollout is is episodic even if we are not at all  using extrinsic rewards
since once it wins it returns to default state and it will deicourage curiosity based learning to win.

"""

import tensorflow as tf
import numpy as np
import gym
from collections import deque 
import random # for random sampling from deque

env=gym.make('MountainCar-v0')


env=env.unwrapped

n_obs=2
n_act=3

qprimary = tf.keras.models.Sequential()
qprimary.add(tf.keras.layers.Dense(units=128,input_dim=n_obs, activation='sigmoid'))
qprimary.add(tf.keras.layers.Dense(units=128, activation="relu"))
qprimary.add(tf.keras.layers.Dense(units=n_act, activation=None))
#optimizer=tf.keras.optimizers.Adam(lr=0.001,beta_1=0.9,epsilon=None,decay=0.00,amsgrad=False)
qprimary.compile(loss="mse", optimizer="RMSprop", metrics=['accuracy'])
qprimary.summary()

qtarget = tf.keras.models.Sequential()
qtarget.add(tf.keras.layers.Dense(units=128,input_dim=n_obs, activation='sigmoid'))
qtarget.add(tf.keras.layers.Dense(units=128, activation="relu"))
qtarget.add(tf.keras.layers.Dense(units=n_act, activation=None))
#optimizer=tf.keras.optimizers.Adam(lr=0.001,beta_1=0.9,epsilon=None,decay=0.00,amsgrad=False)
qtarget.compile(loss="mse", optimizer="RMSprop", metrics=['accuracy'])
qtarget.summary()



encoder = tf.keras.models.Sequential()
encoder.add(tf.keras.layers.Dense(units=128,input_dim=2, activation='relu'))
encoder.add(tf.keras.layers.Dense(units=1000, activation="sigmoid"))
encoder.summary()
encoder.trainable=False

predictor= tf.keras.models.Sequential()
predictor.add(tf.keras.layers.Dense(units=128,input_dim=2, activation='relu'))
predictor.add(tf.keras.layers.Dense(units=1000, activation="sigmoid"))

predictor.compile(loss="mse", optimizer="RMSprop", metrics=['accuracy'])
predictor.summary()



Initial_curiosity_FILE=r"C:\\Users\\Dell\\Desktop\\holidASY\\dqn_rnd_curiosity_keras.h5"
tf.keras.models.save_model(model=predictor,filepath= Initial_curiosity_FILE ,overwrite=True,include_optimizer=True)
print("inital curiosity saved")





memory=deque(maxlen=2000)
epsilon=1.0
epsilon_min=0.01
epsilon_decay=0.99
def restore_curiosity():
	predictor=tf.keras.models.load_model( Initial_curiosity_FILE )
	predictor.compile(loss="mse", optimizer="RMSprop", metrics=['accuracy'])
	print("curiosity restored")

def act(s):
	global epsilon
	epsilon=epsilon*epsilon_decay
	epsilon=max(epsilon,epsilon_min)
	# No random action taken => pure curiosity
	#if np.random.random()<epsilon:

		#print("random")
		#return np.random.choice(np.arange(n_act))
	#print("action")	
	return np.argmax(qprimary.predict(s)[0])
def remember(s,a,r,ns,d):
	memory.append([s,a,r,ns,d])
	
def replay(size):
	gamma=0.8
	sample_size=size
	if len(memory) < sample_size:
		return
	samples=random.sample(memory,sample_size)
	#here we will update parameters of primary so it takes actions in direction of policy.
	# we will sample q values from target but wont update it here so a to keep q values(acts as rule book) stable
	for sample in samples:
		s,a,r,ns,d=sample
		q_target = qtarget.predict(s)
		if d == True:
			q_target[0][a]=r# making it episodic other wise it will relate gameover state to restart state and will do reward farming 
		else:
			q_target[0][a]=r+ gamma*max(qtarget.predict(ns)[0])	
		qprimary.fit(s,q_target,epochs=1,verbose=0)	
def qtraget_train():
	tau=0.01# for soft weight updates
	qprimary_weights= qprimary.get_weights()
	qtraget_weights=qtarget.get_weights()
	for i in range(len(qtraget_weights)):
		qtraget_weights[i]=(1-tau)*qprimary_weights[i]+tau*qtraget_weights[i]
	qtarget.set_weights(qtraget_weights)
def train_predictor(x,y):
	predictor.fit(np.array(x),np.array(y),epochs=1,verbose=0)	
def save():
	FILE=r"C:\\Users\\Dell\\Desktop\\holidASY\\dqn_rnd_keras.h5"
	tf.keras.models.save_model(model=qprimary,filepath= FILE ,overwrite=True,include_optimizer=True)
	print("saved")

def load():
	FILE=r"C:\\Users\\Dell\\Desktop\\holidASY\\dqn_rnd_keras.h5"
	qprimary=tf.keras.models.load_model( FILE )
	qprimary.compile(loss="mse", optimizer="RMSprop", metrics=['accuracy'])
	print("loaded")
episodes = 200
#load()
for i in range(1,episodes):
	s= env.reset()
	s=np.array([s])
	done=False
	steps=0
	x=[]
	y=[]
	while not done:
		env.render()
		action=act(s)
		ns,reward,done,info=env.step(action)
		
		ns=np.array([ns])
		predictor_target = encoder.predict(ns)

		intrinsic_reward=predictor.predict(ns) - predictor_target
		intrinsic_reward=np.sum(intrinsic_reward ,axis=1)**2
		intrinsic_reward*=10#some scaling, hardly makes any difference
		intrinsic_reward=intrinsic_reward.ravel()[0]
		x.append(ns.ravel())
		y.append(predictor_target.ravel())
		if done:
			reward=100
		remember(s,action,reward,ns,done)
		if steps%16 == 0:
			replay(16)
			qtraget_train()
			train_predictor(x,y)
			x=[]
			y=[]			
		s=ns
		steps+=1
	print("episode: "+str(i)+"| steps :"+str(steps) )
	# if taking more steps as episode increases then model is getting bored

	replay(512)
	qtraget_train()
	if i % 5 ==0:
		print("got bored!!")
		restore_curiosity()
	#train_predictor(x,y)
	save()
	