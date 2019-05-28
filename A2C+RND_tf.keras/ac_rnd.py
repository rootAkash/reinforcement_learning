"""
Advantage Actor critic + rnd kind of works but not as good as dqn+rnd
mostly because we are directly optimizing for better actions in the action space in  the action space and this code doent use PPO.
for such simple discrete action space  environments (off policy ) optimizes faster
But the agent can be clearly seen reinforcing actions that lead to unseen states , but takes lots of time to win

"""

import tensorflow as tf
import numpy as np
import gym




prediFILE=r"C:\\Users\\Dell\\Desktop\\holidASY\\ac_predictor.h5"
policyFILE=r"C:\\Users\\Dell\\Desktop\\holidASY\\ac_rnd.h5"

policy = tf.keras.models.Sequential()
policy.add(tf.keras.layers.Dense(units=4,input_dim=2, activation='relu'))
policy.add(tf.keras.layers.Dense(units=8,input_dim=2, activation='relu'))
# output layer
policy.add(tf.keras.layers.Dense(units=3, activation='softmax'))
optimizer=tf.keras.optimizers.Adam(lr=0.001,beta_1=0.9,epsilon=None,decay=0.00,amsgrad=False)
policy.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
policy.summary()

value = tf.keras.models.Sequential()
value.add(tf.keras.layers.Dense(units=128,input_dim=2, activation='sigmoid'))
value.add(tf.keras.layers.Dense(units=128, activation="relu"))
value.add(tf.keras.layers.Dense(units=1, activation=None))
optimizer=tf.keras.optimizers.Adam(lr=0.01,beta_1=0.9,epsilon=None,decay=0.00,amsgrad=False)
value.compile(loss="mse", optimizer="RMSprop", metrics=['accuracy'])
value.summary()

target = tf.keras.models.Sequential()
target.add(tf.keras.layers.Dense(units=128,input_dim=2, activation='relu'))
target.add(tf.keras.layers.Dense(units=1000, activation="sigmoid"))
target.summary()
target.trainable=False

trainable= tf.keras.models.Sequential()
trainable.add(tf.keras.layers.Dense(units=128,input_dim=2, activation='relu'))
#trainable.add(tf.keras.layers.Dense(units=6, activation='relu'))
trainable.add(tf.keras.layers.Dense(units=1000, activation="sigmoid"))

optimizer_1=tf.keras.optimizers.Adam(lr=0.01,beta_1=0.9,epsilon=None,decay=0.0,amsgrad=False)
trainable.compile(loss="mse", optimizer="RMSprop", metrics=['accuracy'])
trainable.summary()



# to restore curiosity
tf.keras.models.save_model(model=trainable,filepath=prediFILE,overwrite=True,include_optimizer=True)


env=gym.make('MountainCar-v0')
env=env.unwrapped#removes step restriction
observation = env.reset()
render =False
done =False
# Hyperparameters
gamma = 0.99


try:
	print("loading")
	#policy=tf.keras.models.load_model( policyFILE )
	
	#value=tf.keras.models.load_model( CriticFILE )
	#value.compile(loss="mse", optimizer=optimizer, metrics=['accuracy'])
	#policy.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])

	print("loaded")
except:
	print("not loaded")	
episodes=1000000
p=1
for episode_nb in range(1,episodes):
	step=0

	observation = env.reset()#since game over brings back to default so cant relate end state(win) with
	# default start otherwise it wont be curious to learn to win (this env doent have exploration levels)
	reward_sum = 0
	done =False
	max_far=0
	render=True
	while not done:
		if render:
			env.render()
		
		proba = policy.predict(np.array([observation]))
		action = np.random.choice(np.arange(proba.shape[1]), p=proba.ravel())

		act=action
		observation_, reward, done, info = env.step(action)
		
		predictor_target = target.predict(np.array([observation_]))

		intrinsic_reward=trainable.predict(np.array([observation_])) - predictor_target
		intrinsic_reward=np.sum(intrinsic_reward ,axis=1)**2
		intrinsic_reward*=10#some scaling, hardly makes any difference


		#adv(st,at) = {q(st,at)}-v(st) ==> {r(t+1) + v(st+1)} - v(st) 		II q(st,at)=r(t+1) + max(q(st+1,all)
		#in this env non episodic evaluation will disourage game over since it returns to same place
		if done == True:
			targetQ=100 + 0*intrinsic_reward # no realtion to reset after game dones
		else:	
			targetQ=intrinsic_reward +	gamma*(value.predict(np.array([observation_])).ravel()) 
		adv = targetQ - value.predict(np.array([observation]))
		#print(adv)
		#print("targetQ:"+str(targetQ))
		#print("val_state:"+str(targetQ - adv ))  
		#print("adv:"+str(adv)) 
		
		#print("episode:"+str(episode_nb))
		if act==0:
			y_=np.array([1,0,0])
		if act==1:
			y_=np.array([0,1,0])	
		if act==2:
			y_=np.array([0,0,1])
		yy=adv*y_
		#if adv.ravel()>0:
			#print("+ "+str(act))

		#if adv.ravel()<0:
			#print("- "+str(act))	
		#print("action*adv:"+str(yy))
		# do one step in our environment
		value.fit(x=np.array([observation]), y=targetQ, epochs=1,verbose=0)
		
		policy.fit(x=np.array([observation]), y=yy, epochs=1,verbose=0)
		if step % 5==0:
			trainable.fit(x=np.array([observation_]), y=predictor_target, epochs=1,verbose=0)


		observation=observation_
		step+=1
		if step == 30000:
			print("useless!!!")
			#break

	print('At the end of episode', episode_nb, 'the total steps was :', step)
	#value=value_t


	# Reinitialization
	if episode_nb> 50:
		render=True
	
	if episode_nb%4==0:
		# restore creativity since after lots of same game it wil loose creativity 
		trainable=tf.keras.models.load_model( prediFILE )
		trainable.compile(loss="mse", optimizer=optimizer_1, metrics=['accuracy'])
	
		tf.keras.models.save_model(model=policy,filepath= policyFILE ,overwrite=True,include_optimizer=True)
		#tf.keras.models.save_model(model=value,filepath=CriticFILE,overwrite=True,include_optimizer=True)
		print("saved")	
