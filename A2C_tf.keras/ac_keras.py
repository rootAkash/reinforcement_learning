import tensorflow as tf
import numpy as np
import gym




ActorFILE=r"C:\\Users\\Dell\\Desktop\\holidASY\\ackeras_Actor.h5"
CriticFILE=r"C:\\Users\\Dell\\Desktop\\holidASY\\ackeras_Critic.h5"

policy = tf.keras.models.Sequential()


policy.add(tf.keras.layers.Dense(units=20,input_dim=4, activation='sigmoid'))
#policy.add(tf.keras.layers.Dense(units=8, activation='relu',kernel_initializer='RandomNormal'))
#policy.add(tf.keras.layers.Dense(units=4, activation='relu',kernel_initializer='RandomNormal'))
# output layer
policy.add(tf.keras.layers.Dense(units=2, activation='softmax'))

# compile the model using traditional Machine Learning losses and optimizers

optimizer=tf.keras.optimizers.Adam(lr=0.001,beta_1=0.9,epsilon=None,decay=0.00,amsgrad=False)
#model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
policy.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
policy.summary()
#model_1=model
# gym initialization



value = tf.keras.models.Sequential()


value.add(tf.keras.layers.Dense(units=20,input_dim=4, activation='sigmoid'))
#value.add(tf.keras.layers.Dense(units=4, activation='sigmoid',kernel_initializer='RandomNormal'))
#value.add(tf.keras.layers.Dense(units=4, activation='relu',kernel_initializer='RandomNormal'))
# output layer
value.add(tf.keras.layers.Dense(units=1, activation=None))

# compile the model using traditional Machine Learning losses and optimizers

optimizer=tf.keras.optimizers.Adam(lr=0.005,beta_1=0.9,epsilon=None,decay=0.00,amsgrad=False)
#model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
value.compile(loss="mse", optimizer=optimizer, metrics=['accuracy'])
value.summary()
#value_t=value
#value_t.compile(loss="mse", optimizer=optimizer, metrics=['accuracy'])
#value_t.summary()
# gym initialization
env=gym.make('CartPole-v0')
observation = env.reset()

right=np.array([0,1])
left=np.array([1,0])
done =False
# Hyperparameters
gamma = 0.99
reward_sum=0
# initialization of variables used in the main loop


try:
	print("loading")
	#policy=tf.keras.models.load_model( ActorFILE )
	
	#value=tf.keras.models.load_model( CriticFILE )
	#value.compile(loss="mse", optimizer=optimizer, metrics=['accuracy'])
	#policy.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])

	print("loaded")
except:
	print("not loaded")	
episodes=1000000
for episode_nb in range(1,episodes):


	while not done:
		
		env.render()
		
		proba = policy.predict(np.array([observation]))
		#action = np.argmax(proba[0].ravel())
		action = np.random.choice(np.arange(proba.shape[1]), p=proba.ravel())
		#print(proba[0].ravel()) 
		#print(np.argmax(proba[0]))
		y = np.array([action]) #  our labels
		

		observation_, reward, done__, info = env.step(action)
		reward=0
		
		if abs(observation_[2])<0.1:
			
			reward=5*1/(1+abs(observation_[0]))
			
			#reward=30
		if abs(observation_[2])>0.1:
			reward=-1
			
		
		if abs(observation_[0])>2 or abs(observation_[2])>1:
			done = True
			reward=-30
			print("done")
		#adv(st,at) = {q(st,at)}-v(st) ==> {r(t+1) + v(st+1)} - v(st) 		II q(st,at)=r(t+1) + max(q(st+1,all)

		targetQ=reward+	gamma*(value.predict(np.array([observation_])).ravel()) 
		adv = targetQ - value.predict(np.array([observation]))
		print("targetQ:"+str(targetQ))
		print("val_state:"+str(targetQ - adv ))  
		print("adv:"+str(adv)) 
		print("episode:"+str(episode_nb))
		if action==0:
			y_=left
		else:
			y_=right
		yy=adv*y_
		print("action*adv:"+str(yy))
		# do one step in our environment
		value.fit(x=np.array([observation]), y=targetQ, epochs=1,verbose=1)
		#value_t.fit(x=np.array([observation]), y=targetQ, epochs=1,verbose=1)
		policy.fit(x=np.array([observation]), y=yy, epochs=1,verbose=1)



		observation=observation_


	print('At the end of episode', episode_nb, 'the total reward was :', reward_sum)
	#value=value_t


	# Reinitialization
	
	
	observation = env.reset()
	reward_sum = 0
	done =False
	if episode_nb%10==0:	
		#tf.keras.models.save_model(model=policy,filepath=ActorFILE,overwrite=True,include_optimizer=True)
		#tf.keras.models.save_model(model=value,filepath=CriticFILE,overwrite=True,include_optimizer=True)
		print("saved")	
