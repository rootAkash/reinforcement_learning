import tensorflow as tf
import numpy as np
import gym


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
def discount_rewards_mine(r,gamma):
	r = np.array(r)
	discounted_r = np.zeros_like(r)*0.0
	running_add = 0
	for t in reversed(range(0, r.size)):
        
		running_add = running_add * gamma + r[t]
		discounted_r[t] = running_add
	#discounted_r -= np.mean(discounted_r) #normalizing the result
	#discounted_r /= np.std(discounted_r)	
	return discounted_r		


FILE=r"C:\\Users\\Dell\\Desktop\\holidASY\\cart_policy_s_w_w3.h5"


model = tf.keras.models.Sequential()


model.add(tf.keras.layers.Dense(units=8,input_dim=4, activation='sigmoid'))
#model.add(tf.keras.layers.Dense(units=8, activation='relu',kernel_initializer='RandomNormal'))
model.add(tf.keras.layers.Dense(units=4, activation='relu'))
# output layer
model.add(tf.keras.layers.Dense(units=2, activation='softmax'))

# compile the model using traditional Machine Learning losses and optimizers

optimizer=tf.keras.optimizers.Adam(lr=0.01,beta_1=0.9,epsilon=None,decay=0.0001,amsgrad=False)
#model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
model.summary()
#model_1=model
# gym initialization

env=gym.make('CartPole-v0')
observation = env.reset()

right=np.array([0,1])
left=np.array([1,0])
done =False
# Hyperparameters
gamma = 0.9

# initialization of variables used in the main loop
x_train, y_train, rewards = [],[],[]
reward_sum = 0
episode_nb = 0
try:
	print("loading")
	#model=tf.keras.models.load_model(FILE)
	#model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
	#model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
	print("loaded")
except:
	print("not loaded")	
episodes=1000000
for episode_nb in range(episodes):


	while not done:
		#observation=np.array([observation])
		#print(np.shape(observation))
		env.render()
		#print(observation[2])
		# forward the policy network and sample action according to the proba distribution
		proba = model.predict(np.array([observation]))
		#action = np.argmax(proba[0].ravel())
		action = np.random.choice(np.arange(proba.shape[1]), p=proba.ravel())
		print(proba[0].ravel())

		#print(np.argmax(proba[0]))
		y = np.array([action]) #  our labels
		#print(np.shape(np.array([[observation]])))
		# log the input and label to train later
		x_train.append(observation)

		y_train.append(action)
		observation, reward, donein, info = env.step(action)
		reward=0
		
		if abs(observation[2])<0.1:
			
			#reward=5*1/(1+abs(observation[0]))
			reward=3
		if abs(observation[2])>0.1:
			reward=0
			
		
		if abs(observation[0])>2 or abs(observation[2])>1.5:
			done = True
			reward=-3
			print("done")
			
		# do one step in our environment

		rewards.append(reward)
		reward_sum += reward


	print('At the end of episode', episode_nb, 'the total reward was :', reward_sum)

	# increment episode number

	x_train=np.array(x_train)

	y_train=np.array(y_train)

	drr=discount_rewards_mine(rewards, gamma)
	#model.fit(x=x_train, y=y_train, epochs=1,verbose=1, sample_weight=drr)
	
	yyy=[]
	for i in range(len(drr)):
		act=y_train[i].ravel()
		if act==0:
			y_=left
		else:
			y_=right
		yy=drr[i].ravel()*y_	
		#print(yy)

		yyy.append(yy)
	yyy=np.array(yyy)	
	model.fit(x=x_train, y=yyy, epochs=1,verbose=1)
	print(drr)
	print(yyy)
	
	#if episode_nb%10==0:
		#model_1=model	                                                     
	# Reinitialization
	
	x_train, y_train, rewards = [],[],[]
	observation = env.reset()
	reward_sum = 0
	done =False
	if episode_nb%100==0:	
		#tf.keras.models.save_model(model=model,filepath=FILE,overwrite=True,include_optimizer=False)
		print("saved")	