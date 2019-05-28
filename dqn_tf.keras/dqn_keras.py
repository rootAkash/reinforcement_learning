import tensorflow as tf
import numpy as np
import gym
#from collections import deque # not using tgis for fun
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

#memory=deque(maxlen=2000)
class my_que:
	def __init__(self,maxlen):
		self.max=maxlen
		self.counter=0
		self.queue=[]
	def push(self,item):
		if self.counter<self.max:
			self.queue.append(item)
			self.counter+=1
		else:
			self.queue.pop(0)
			self.queue.append(item)
			self.counter+=1			
	def show(self):
		return self.queue,self.counter


mem= my_que(maxlen=2000)
def sampling(list,size):
	length=len(list)
	
	if length<=size:
		ctr = 0
	else:	
		ctr = np.random.randint(0,length-size)

	return list[ctr:ctr+size]

epsilon=1.0
epsilon_min=0.01
epsilon_decay=0.995

def act(s):
	global epsilon
	epsilon=epsilon*epsilon_decay
	epsilon=max(epsilon,epsilon_min)
	if np.random.random()<epsilon:
		#print("random")
		return np.random.choice(np.arange(n_act))
	#print("action")	
	return np.argmax(qprimary.predict(s)[0])
def remember(s,a,r,ns,d):
	#memory.append([s,a,r,ns,d])
	mem.push([s,a,r,ns,d])
def replay(size):
	gamma=0.8
	sample_size=size
	memory,length=mem.show()
	if length<sample_size: # len(memory) < sample_size:
		return
	#samples=random.sample(memory,sample_size)
	samples=sampling(memory,sample_size)
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
	tau=0.1# for soft weight updates
	qprimary_weights= qprimary.get_weights()
	qtraget_weights=qtarget.get_weights()
	for i in range(len(qtraget_weights)):
		qtraget_weights[i]=(1-tau)*qprimary_weights[i]+tau*qtraget_weights[i]
	qtarget.set_weights(qtraget_weights)
def save():
	FILE=r"C:\\Users\\Dell\\Desktop\\holidASY\\dqn_keras.h5"
	tf.keras.models.save_model(model=qprimary,filepath= FILE ,overwrite=True,include_optimizer=True)
	print("saved")

def load():
	FILE=r"C:\\Users\\Dell\\Desktop\\holidASY\\dqn_keras.h5"
	qprimary=tf.keras.models.load_model( FILE )
	qprimary.compile(loss="mse", optimizer="RMSprop", metrics=['accuracy'])
	print("loaded")
episodes = 200
#load()
for i in range(episodes):
	s= env.reset()
	s=np.array([s])
	done=False
	steps=0
	while not done:
		env.render()
		action=act(s)
		ns,reward,done,info=env.step(action)
		reward=ns[1]
		#print(reward)
		ns=np.array([ns])
		if done:
			reward=100
		remember(s,action,reward,ns,done)
		if steps%2 == 0:
			replay(8)
			qtraget_train()
		s=ns
		steps+=1
	print("episode: "+str(i)+"| steps :"+str(steps) )
	replay(512)
	qtraget_train()
	save()
	