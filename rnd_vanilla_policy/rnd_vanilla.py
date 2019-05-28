"""
here a simple vanilla  policy gradients is paired up with random network distillation
rewards are only intrinsic and policy  is directly optimized using discounted rewards ;
As expected it takes nearly eternity to win  :-0

"""
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
def discount_rewards_mine(r,gamma=0.9):
	r = np.array(r)
	discounted_r = np.zeros_like(r)*0.0
	running_add = 0
	for t in reversed(range(0, r.size)):
        
		running_add = running_add * gamma + r[t]
		discounted_r[t] = running_add
	#discounted_r -= np.mean(discounted_r) #normalizing the result
	#discounted_r /= np.std(discounted_r)	
	return discounted_r		


FILE=r"C:\\Users\\Dell\\Desktop\\holidASY\\car_curiosity.h5"


model = tf.keras.models.Sequential()


model.add(tf.keras.layers.Dense(units=8,input_dim=2, activation='sigmoid'))
model.add(tf.keras.layers.Dense(units=4, activation='relu'))
# output layer
model.add(tf.keras.layers.Dense(units=3, activation='softmax'))

# compile the model using traditional Machine Learning losses and optimizers

optimizer=tf.keras.optimizers.Adam(lr=0.001,beta_1=0.9,epsilon=None,decay=0.0001,amsgrad=False)
#model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
model.summary()
#model_1=model
# gym initialization
target = tf.keras.models.Sequential()


target.add(tf.keras.layers.Dense(units=8,input_dim=2, activation='sigmoid'))
#target.add(tf.keras.layers.Dense(units=4, activation='relu'))
# output layer
target.add(tf.keras.layers.Dense(units=3, activation='relu'))

# compile the model using traditional Machine Learning losses and optimizers

#model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
#target.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
target.summary()
target.trainable=False

trainable= tf.keras.models.Sequential()


trainable.add(tf.keras.layers.Dense(units=8,input_dim=2, activation='sigmoid'))
#trainable.add(tf.keras.layers.Dense(units=4, activation='relu'))
# output layer
trainable.add(tf.keras.layers.Dense(units=3, activation='relu'))

# compile the trainable using traditional Machine Learning losses and optimizers

optimizer_1=tf.keras.optimizers.Adam(lr=0.01,beta_1=0.9,epsilon=None,decay=0.0001,amsgrad=False)
#trainable.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
trainable.compile(loss="mse", optimizer=optimizer_1, metrics=['accuracy'])
trainable.summary()

env=gym.make('MountainCar-v0')
observation = env.reset()
print(observation)
print(env.observation_space)

done =False

# Hyperparameters
gamma = 0.9

# initialization of variables used in the main loop
x_train, y_train = [],[]

try:
	print("loading")
	#model=tf.keras.models.load_model(FILE)
	#model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
	print("loaded")
except:
	print("not loaded")	
episodes=1000000
steps =100
for episode_nb in range(episodes):


	for step in range(steps):
		#observation=np.array([observation])
		#print(np.shape(observation))
		env.render()
		#print(observation[2])
		# forward the policy network and sample action according to the proba distribution
		proba = model.predict(np.array([observation]))

		#action = np.argmax(proba[0].ravel())
		action = np.random.choice(np.arange(proba.shape[1]), p=proba.ravel())
		#print(proba[0].ravel())
		#print("act1:"+str(action))
		#print(np.argmax(proba[0]))
		y = np.array([action]) #  our labels
		#print(np.shape(np.array([[observation]])))
		# log the input and label to train later
		x_train.append(observation)

		y_train.append(action)
		observation, reward, donein, info = env.step(action)
		
			
		# do one step in our environment

	# increment episode number

	x_train=np.array(x_train)

	y_train=np.array(y_train)

	#model.fit(x=x_train, y=y_train, epochs=1,verbose=1, sample_weight=drr)
	tar=target.predict(x_train)
	#print(tar)
	#print(np.shape(tar))
	rewards= (trainable.predict(x_train) - tar)
	print("###############################")
	#print(np.shape(rewards))

	rewards=np.sum(rewards,axis=1)
	#print("sum")
	#print(rewards)
	#print(np.shape(rewards))
	rewards=rewards**2
	#print("square")
	#print(rewards)
	#print(np.shape(rewards))
	drr=rewards 
	drr=drr - np.mean(drr,axis=0)
	drr= drr/np.std(drr,axis=0)
	#drr=drr*10
	drr=discount_rewards_mine(drr,0.7)
	print('At the end of episode', episode_nb, 'the total reward was :', np.sum(drr,axis=0))
	print("###############################")
	yyy=[]
	for i in range(len(drr)):
		act=y_train[i].ravel()
		#print(act)
		if act==0:
			y_=np.array([1,0,0])
		if act==1:
			y_=np.array([0,1,0])	
		if act==2:
			y_=np.array([0,0,1])
		yy=drr[i].ravel()*y_	
		#print(yy)
		#print("......")
		yyy.append(yy)
	yyy=np.array(yyy)
	print("###############################")	
	print(yyy)
	model.fit(x=x_train, y=yyy, epochs=1,batch_size=32,verbose=1)
	trainable.fit(x=x_train, y=tar, epochs=1,batch_size=32,verbose=1)
	print("###############################")
	
	#if episode_nb%10==0:
		#model_1=model	                                                     
	# Reinitialization
	
	x_train, y_train = [],[]
	#observation = env.reset()
	
	done =False
		
	#tf.keras.models.save_model(model=model,filepath=FILE,overwrite=True,include_optimizer=False)
	print("saved")	
