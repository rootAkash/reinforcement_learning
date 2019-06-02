import tensorflow as tf
import  numpy as np
import gym
from collections import deque
import random
env=gym.make('MountainCarContinuous-v0')

env=env.unwrapped#removes step restriction
#env.seed(1)

s_dim = env.observation_space.shape[0]
print(s_dim)
a_dim = env.action_space.shape[0]
print(a_dim)
a_bound = env.action_space.high
print(a_bound)



state_inputs = tf.keras.Input(shape=(s_dim,), name='state')
x = tf.keras.layers.Dense(64, activation='relu')(state_inputs)
x = tf.keras.layers.Dense(64, activation='relu')(x)
action_outputs = tf.keras.layers.Dense(a_dim, activation='tanh')(x)

primary_actor= tf.keras.Model(inputs=state_inputs, outputs=action_outputs, name='p_actor_model')
# create custom loss
# y_pred = dQ/da and y_true=U(s) that is actions that is U(s) =a
# so loss=U(s)*dQ/da ; policy params =@
# then dloss/d@=(dU(s)/d@)*dQ/da ; since dQ/da is a const here that is y_true
# so the eqn of dL/d@ = (da/d@)*(dQ/da) => d@/d@ ; which is ddpg actor loss (it changes the policy params to maximize the Q that is give better actions)
def actorloss(y_true,y_pred):
	q= tf.multiply(y_true,y_pred)
	loss = tf.reduce_mean(-q)# for maxiimizing Q we minimize -Q with gradient descent
	return loss

primary_actor.compile(loss=actorloss, optimizer="RMSprop", metrics=[actorloss])# custom loss needs to be defined
primary_actor.summary()
target_actor = tf.keras.Model(inputs= state_inputs, outputs= action_outputs, name='t_actor_model')
target_actor.trainable=False
target_actor.summary()



action_inputs =  tf.keras.Input(shape=(a_dim,), name='action')
x=tf.keras.layers.Dense(64, activation='relu')(state_inputs)
x=tf.keras.layers.concatenate([tf.keras.layers.Flatten()(x),action_inputs])
x=tf.keras.layers.Dense(64, activation='relu')(x)
Qout=tf.keras.layers.Dense(1, activation=None)(x)
primary_critic= tf.keras.Model(inputs=[state_inputs,action_inputs], outputs=Qout, name='p_critic_model')
primary_critic.compile(loss="mse", optimizer="RMSprop", metrics=['accuracy'])
primary_critic.summary()
target_critic= tf.keras.Model(inputs=[state_inputs,action_inputs], outputs=Qout, name='t_critic_model')
target_critic.trainable=False
target_critic.summary()
# now we create a function to give dQ(s,a)/da
get_qgrads= tf.keras.backend.function([primary_critic.input[0], primary_critic.input[1]], tf.keras.backend.gradients(primary_critic.output, [primary_critic.input[1]]))


memory=deque(maxlen=20000)
# WILL NOT USE OU NOISE
"""
class OUNoise:
	#docstring for OUNoise
	def __init__(self,action_dimension,mu=0, theta=0.15, sigma=0.2):
		self.action_dimension = action_dimension
		self.mu = mu
		self.theta = theta
		self.sigma = sigma
		self.state = np.ones(self.action_dimension) * self.mu
		self.reset()

	def reset(self):
		self.state = np.ones(self.action_dimension) * self.mu

	def noise(self):
		x = self.state
		dx = self.theta * (self.mu - x) + self.sigma * nr.randn(len(x))
		self.state = x + dx
	    return self.state
"""
def update_actor_target(tau=0.5):
	# for soft weight updates
	actor_primary_weights= primary_actor.get_weights()
	actor_target_weights=target_actor.get_weights()
	for i in range(len(actor_target_weights)):
		actor_target_weights[i]=(1-tau)*actor_primary_weights[i]+tau*actor_target_weights[i]
	target_actor.set_weights(actor_target_weights)
def update_critic_target(tau=0.5):
	# for soft weight updates
	critic_primary_weights= primary_critic.get_weights()
	critic_target_weights=target_critic.get_weights()
	for i in range(len(critic_target_weights)):
		critic_target_weights[i]=(1-tau)*critic_primary_weights[i]+tau*critic_target_weights[i]
	target_critic.set_weights(critic_target_weights)
def remember(s,a,r,ns,d):
	s=s.ravel()
	ns=ns.ravel()
	memory.append([s,a,r,ns,d])
def replay_train_critic_actor(size=128):
	gamma=0.99
	sample_size=size
	if len(memory) < sample_size:
		return
	samples=random.sample(memory,sample_size)
	#here we will update parameters of primary so it takes actions in direction of policy.
	# we will sample q values from target but wont update it here so a to keep q values(acts as rule book) stable
	qty=[]
	states_actions=[]
	actions=[]
	states=[]
	for sample in samples:
		s,a,r,ns,d=sample
		if d == True:
			y=np.array([r]).ravel()# making it episodic other wise it will relate gameover state to restart state and will do reward farming 
		else:
			a_=target_actor.predict(np.array([ns]))
			y=r+ gamma*target_critic.predict([np.array([ns]),a_])
		actions.append(a.ravel())
		qty.append(y.ravel())
		states.append(s)	
	primary_critic.fit([np.array(states),np.array(actions)],np.array(qty),epochs=1,verbose=0)		
	a = primary_actor.predict(np.array(states))
	actions=np.array(actions)
	dq_da=get_qgrads([states,a])
	primary_actor.fit(np.array(states),dq_da,epochs=1,verbose=0)

def save_actor_critic_weights():
	actorpath=r"C:\\Users\\Dell\\Desktop\\holidASY\\softAC,DDPG\\ddpg_car_actor.h5"
	criticpath=r"C:\\Users\\Dell\\Desktop\\holidASY\\softAC,DDPG\\ddpg_car_critic.h5"
	#tf.keras.models.save_model(primary_actor,actorpath,overwrite=True,include_optimizer=True)
	#tf.keras.models.save_model(primary_critic,criticpath,overwrite=True,include_optimizer=True)
	primary_actor.save_weights(actorpath)
	primary_critic.save_weights(criticpath)
	print("saved")
def load_actor_critic_weights():
	actorpath=r"C:\\Users\\Dell\\Desktop\\holidASY\\softAC,DDPG\\ddpg_car_actor.h5"
	criticpath=r"C:\\Users\\Dell\\Desktop\\holidASY\\softAC,DDPG\\ddpg_car_critic.h5"
	primary_actor.load_weights(actorpath)
	primary_critic.load_weights(criticpath)
	print("loaded")



episodes = 5000
steps = 2000 
update_critic_target(0)
update_actor_target(0)
ctr = 0
var=1
render =False
#load_actor_critic_weights()
s = env.reset()
for ep in range(episodes):
	s = env.reset()
	done=False
	stp=0
	while not done:
	
		if render:
			env.render()	
		action=primary_actor.predict(np.array([s.ravel()]))[0]# the NN output is (-1,1)
		e=np.random.normal(action,var,size=(1,a_dim))[0]
		a =np.clip(e,-1,1 )
		s_,r,done,_=env.step(a_bound*a.ravel())#scaling the action to actual action
		remember(s,a,r,s_,done)
		
		if ctr%100 == 0:
			var=var*0.9995
			replay_train_critic_actor(512)
			update_critic_target(0.5)
			update_actor_target(0.5)
		stp+=1
		ctr+=1
		s=s_
		if stp>2000:
			break
	print("episode: "+str(ep)+ " exploration variance: "+str(var) + " steps: "+str(stp))
	if stp<100:
		var*=0.95
	replay_train_critic_actor(512)
	update_critic_target(0.5)
	update_actor_target(0.5)
	if var<0.09:
		var=0
		render=True	
		save_actor_critic_weights()
