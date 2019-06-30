import tensorflow as tf
import numpy as np
import gym
import random
from collections import deque
import math
#env=gym.make('Pendulum-v0')
env=gym.make('MountainCarContinuous-v0')

env=env.unwrapped

s_dim = env.observation_space.shape[0]
print(s_dim)
a_dim = env.action_space.shape[0]
print(a_dim)
a_bound = env.action_space.high[0]
print(a_bound)
DUMMY_ACTION, DUMMY_VALUE = np.zeros((1,a_dim)), np.zeros((1, 1))


################## policy ################
E_inputs = tf.keras.Input(shape=(1,), name='E_samples')
dq_da = tf.keras.Input(shape=(1,), name='dq1/danew')

state_inputs = tf.keras.Input(shape=(s_dim,), name='state')
x = tf.keras.layers.Dense(100, activation='relu',kernel_initializer=tf.keras.initializers.glorot_uniform(seed=44))(state_inputs)
x1 = tf.keras.layers.Dense(50, activation='relu',kernel_initializer=tf.keras.initializers.glorot_uniform(seed=44))(x)
mu_0 = tf.keras.layers.Dense(a_dim, activation='tanh',kernel_initializer=tf.keras.initializers.glorot_uniform(seed=44))(x1)
x2 = tf.keras.layers.Dense(25, activation='relu',kernel_initializer=tf.keras.initializers.glorot_uniform(seed=44))(x)
sigma_0 = tf.keras.layers.Dense(a_dim, activation='softplus',kernel_initializer=tf.keras.initializers.glorot_uniform(seed=44))(x2)
mu = tf.keras.layers.Lambda(lambda x: x )(mu_0)
sigma = tf.keras.layers.Lambda(lambda x: tf.clip_by_value(x,1e-22 , 1e+02))(sigma_0)# clamp bet e -22 , e +2
musig=tf.keras.layers.concatenate([mu,sigma])
# so we need to differentiate Q1 wrt f and logpi(f) ; where f is reparameterised actions with E i.e. a = f(s,E)
# so  we find dQ1/df and recreate f(s,E) by taking in E and s used to find dQ1/df
# then we find -log(pi(f)) , which is the entopy
# then we do s.g.d. w.r.t policy params on the loss = -[  f*(dQ1/df) - log(pi(f))]
def SAC_loss(E_inputs,dq_da):
	temperature = 1
	pi=3.1415926
	mu_mask = tf.convert_to_tensor(np.array([[1],[0]]), dtype = tf.float32)
	sig_mask = tf.convert_to_tensor(np.array([[0],[1]]), dtype = tf.float32)
		
	def loss(y_true, y_pred):
		mu = tf.keras.backend.dot(y_pred,mu_mask)
		sigma = tf.keras.backend.dot(y_pred,sig_mask)
		sigma_sq=tf.keras.backend.square(sigma)
		new_act = mu + tf.multiply(sigma,E_inputs) 
		new_actions = tf.tanh(new_act)
		pdf = 1. / tf.keras.backend.sqrt(2. *pi* sigma_sq) * tf.keras.backend.exp(-tf.keras.backend.square(new_act- mu) / (2. * sigma_sq))
		entropy= tf.keras.backend.log(pdf )  - tf.keras.backend.log(1 - tf.keras.backend.square(new_actions) + 1e-07)
		#log_pdf = tf.keras.backend.log(pdf )  - tf.reduced_sum(tf.keras.backend.log(1 - tf.keras.backend.square(new_actions) + tf.keras.backend.epsilon()),axis=0)# need to experiment for multi_actions

		loss =tf.keras.backend.mean( temperature*entropy - tf.multiply(dq_da,new_actions) )
		return loss
	return loss	
policy= tf.keras.Model(inputs=(state_inputs,E_inputs,dq_da), outputs=(musig), name='p_actor_model')
policy.compile(loss=SAC_loss(E_inputs=E_inputs,dq_da=dq_da), optimizer=tf.keras.optimizers.Adam(lr=0.001))
policy.summary()

###################### q functions #######################

action_inputs =  tf.keras.Input(shape=(a_dim,), name='action')
x=tf.keras.layers.Dense(100, activation='relu',kernel_initializer=tf.keras.initializers.glorot_uniform(seed=44))(state_inputs)
x=tf.keras.layers.concatenate([tf.keras.layers.Flatten()(x),action_inputs])
x=tf.keras.layers.Dense(64, activation='relu',kernel_initializer=tf.keras.initializers.glorot_uniform(seed=44))(x)
Qout=tf.keras.layers.Dense(1, activation=None,kernel_initializer=tf.keras.initializers.glorot_uniform(seed=44))(x)
q_1= tf.keras.Model(inputs=[state_inputs,action_inputs], outputs=Qout, name='p_critic_model')
q_1.compile(loss="mse",optimizer=tf.keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
q_1.summary()
q_2= tf.keras.Model(inputs=[state_inputs,action_inputs], outputs=Qout, name='t_critic_model')
q_2.compile(loss="mse",optimizer=tf.keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
q_2.summary()
# now we create a function to give dQ(s,a)/da
get_q_1_grads= tf.keras.backend.function([q_1.input[0], q_1.input[1]], tf.keras.backend.gradients(q_1.output, [q_1.input[1]]))

########################### val func #######################


x = tf.keras.layers.Dense(100, activation='relu',kernel_initializer=tf.keras.initializers.glorot_uniform(seed=44))(state_inputs)
x = tf.keras.layers.Dense(64, activation='relu',kernel_initializer=tf.keras.initializers.glorot_uniform(seed=44))(x)
value_outputs = tf.keras.layers.Dense(1, activation=None,kernel_initializer=tf.keras.initializers.glorot_uniform(seed=44))(x)
value= tf.keras.Model(inputs=state_inputs, outputs=value_outputs, name='p_value_model')
value.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
value.summary()
t_value= tf.keras.Model(inputs=state_inputs, outputs=value_outputs, name='t_value_model')
#t_value.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
t_value.trainable =False
t_value.summary()

memory=deque(maxlen=20000)


def get_log_pdf(x, mean, std):
	sigma_sq = (std)**2
	pdf= 1./(2*3.1415926*sigma_sq)**0.5 * math.exp(-(float(x)-float(mean))**2/(2.*sigma_sq))
	log_pdf = np.log(pdf) 
	return log_pdf
def get_entropy(x, mean, std,Act):
	sigma_sq = (std)**2
	pdf= 1./(2*3.1415926*sigma_sq)**0.5 * math.exp(-(float(x)-float(mean))**2/(2.*sigma_sq))
	ent = np.log(pdf) - np.log((1- (Act)**2 )+ 1e-07)
	return -ent
def get_ntropy(x, mean, std,Act):
	# this is very slow as compared to calculating using pdf formula
	dist = tf.contrib.distributions.Normal(loc=np.array([mean]), scale=np.array([std]))
	lp = dist.log_prob(np.array([x]))
	#print(tf.keras.backend.eval(lp))
	#lp=tf.keras.backend.eval(lp).ravel()[0] - np.sum(np.log((1- (Act)**2 )+ 1e-07)).ravel()[0]
	lp=tf.keras.backend.eval(lp).ravel()[0] - np.log((1- (Act)**2 )+ 1e-07)
	return lp
def update_value_target(tau=0.01):
	# for soft weight updates
	value_weights= value.get_weights()
	value_target_weights=t_value.get_weights()
	for i in range(len(value_target_weights)):
		value_target_weights[i]=(1-tau)*value_target_weights[i]+tau*value_weights[i]
	t_value.set_weights(value_target_weights)
def remember(s,a,r,ns,d):
	s=s.ravel()
	ns=ns.ravel()
	memory.append([s,a,r,ns,d])
def replay_and_train(size=128):
	gamma=0.95
	sample_size=size
	if len(memory) < sample_size:
		return
	samples=random.sample(memory,sample_size)
	yvs=[]
	yqs=[]
	Es=[]
	states=[]
	new_Actions=[]
	actions=[]
	new_musigs=[]
	for sample in samples:
		s,a,r,ns,d=sample
		yq = r + gamma*(1-d)*t_value.predict(np.array([ns]))[0]
		E =np.random.normal(0,1)
		new_musig = policy.predict((np.array([s]),np.array([[0]]),np.array([[0]])))
		new_mu,new_sig=new_musig[0][0],new_musig[0][1]
		new_Action  =np.tanh(new_mu + new_sig*E) 
		entropy = get_entropy(new_mu + new_sig*E,new_mu,new_sig,new_Action)
		yv = min(q_1.predict([np.array([s]),np.array([new_Action])])[0],q_2.predict([np.array([s]),np.array([new_Action])])[0]) + 1.*entropy
		new_Actions.append(np.array([new_Action]))
		actions.append(np.array([a]))
		Es.append(np.array([E]))
		yvs.append(yv)
		yqs.append(yq)
		states.append(s)	
		new_musigs.append(new_musig)
	a = np.array(new_Actions)   
	Es =np.array(Es)
	states=np.array(states)
	q_1.fit([states,np.array(actions)],np.array(yqs),epochs=1,batch_size=512,verbose=0)		
	q_2.fit([states,np.array(actions)],np.array(yqs),epochs=1,batch_size=512,verbose=0)		
	value.fit(states,np.array(yvs),epochs=1,batch_size=512,verbose=0)	
	dq_da=get_q_1_grads([states,a])[0]
	policy.fit((states,Es,dq_da),np.array(new_musigs),epochs=1,batch_size=512,verbose=0)
	update_value_target(0.01)
	

#C:\Users\Dell\Desktop\holidASY\Soft AC
def save_actor_weights():
	actorpath=r"C:\\Users\\Dell\\Desktop\\holidASY\\Soft AC\\SACcar_actor.h5"
	policy.save_weights(actorpath)
	print("saved")
def load_actor_weights():
	actorpath=r"C:\\Users\\Dell\\Desktop\\holidASY\\Soft AC\\SACcar_actor.h5"
	policy.load_weights(actorpath)
	print("loaded")



episodes = 5000
steps = 3500 
update_value_target(1)
ctr = 0
render =False
train_iter = 1000  # 1000, batch of 8/16 works best for now
#load_actor_weights()
s = env.reset()
for ep in range(episodes):
	s = env.reset()
	done=False
	rews=0
	if ep>15:
		render=1
		save_actor_weights()
	for step in range(steps):
		if done:
			s = env.reset()
		if render:
			env.render()	
		musig = policy.predict((np.array([s]),np.array([[0]]),np.array([[0]])))
		mu,sig=musig[0][0],musig[0][1]
		E =np.random.normal(0,1)
		Action  =np.tanh(mu + sig*E)
		if ep < 10 :
			# for some additional exploration not necesserily needed
			if E > 0.5:
				Action = np.clip(a_bound*E,-a_bound,a_bound)		
		s_,r,done,_=env.step(np.array([Action*a_bound]))
		if done :
			r =r+10000	# to encourage reaching target more
			print("reached")
		remember(s,Action,r,s_,done)
		rews+=r	
		ctr+=1
		s=s_
	print("episode: "+str(ep)+ " rews: "+str(rews))		
	print("training")
	for i in  range(train_iter):
		replay_and_train(16)
		if i % (train_iter//10)==0:
			print('.',end='')
	print('|')		
		