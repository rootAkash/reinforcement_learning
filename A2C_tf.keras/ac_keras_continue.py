import numpy as np
import gym
import tensorflow as tf



env=gym.make('Pendulum-v0')
#env=env.unwrapped

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
sigma_0 = tf.keras.layers.Dense(a_dim, activation='softplus')(x2)# can use softplus activation / sigmoid activation

mu = tf.keras.layers.Lambda(lambda x: x * a_bound)(mu_0)
sigma = tf.keras.layers.Lambda(lambda x: x + 0.0001)(sigma_0)
# concat layer
musig=tf.keras.layers.concatenate([mu,sigma])
def ac_loss(advantage, action):
	entropy_loss = 0.01
	pi=3.1415926

	def loss(y_true, y_pred):

		mu=tf.keras.backend.expand_dims(y_pred[:,0],1)
		sigma = tf.keras.backend.expand_dims(y_pred[:,1],1)
		sigma_sq=tf.keras.backend.square(sigma)
		pdf = 1. / tf.keras.backend.sqrt(2. *pi* sigma_sq) * tf.keras.backend.exp(-tf.keras.backend.square(action - mu) / (2. * sigma_sq))
		log_pdf = tf.keras.backend.log(pdf + tf.keras.backend.epsilon())
		entropy = tf.keras.backend.sum(0.5 * (tf.keras.backend.log(2. * pi * sigma_sq) + 1.))
		loss = -tf.keras.backend.sum(advantage*log_pdf + entropy_loss*entropy)
		
		return loss
	return loss	
policy= tf.keras.Model(inputs=[state_inputs, advantage,action], outputs=[musig], name='p_actor_model')

policy.compile(loss=ac_loss(advantage=advantage,action=action), optimizer=tf.keras.optimizers.Adam(lr=0.0001))# big lr will make policy unstable
policy.summary()





x = tf.keras.layers.Dense(16, activation='relu')(state_inputs)
x = tf.keras.layers.Dense(16, activation='relu')(x)
value_outputs = tf.keras.layers.Dense(1, activation=None)(x)
critic= tf.keras.Model(inputs=state_inputs, outputs=value_outputs, name='p_critic_model')
critic.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
critic.summary()


#C:\Users\Dell\Desktop\holidASY\A2C_tf.keras


def save_weights():
	actorpath=r"C:\\Users\\Dell\\Desktop\\holidASY\\A2C_tf.keras\\a2c_cont_actor.h5"
	criticpath=r"C:\\Users\\Dell\\Desktop\\holidASY\\A2C_tf.keras\\a2c_cont_critic.h5"
	policy.save_weights(actorpath)
	critic.save_weights(criticpath)
	print("saved")
def load_weights():
	actorpath=r"C:\\Users\\Dell\\Desktop\\holidASY\\A2C_tf.keras\\a2c_cont_actor.h5"
	criticpath=r"C:\\Users\\Dell\\Desktop\\holidASY\\A2C_tf.keras\\a2c_cont_critic.h5"
	policy.load_weights(actorpath)
	critic.load_weights(criticpath)
	print("loaded")

	


episodes = 20000000
render=0
s=env.reset()
dummy_Prediction= policy.predict((np.array([s]),DUMMY_VALUE,DUMMY_ACTION))
#load_weights()
for episode in range(1,episodes):
	done = 0
	s=env.reset()
	rews = 0	
	if episode>2000:
		render=1
	while not done:
		if render:
			env.render()
		out =policy.predict((np.array([s]),DUMMY_VALUE,DUMMY_ACTION))	
		mu_pred,sigma_pred =out[0][0],out[0][1] # prob distribution
		action= np.random.normal(mu_pred, sigma_pred,a_dim)
		s_, reward, done, info = env.step(action)
		values = critic.predict(np.array([s]))
		values_ = critic.predict(np.array([s_]))
		returns =reward/10 + 0.9*values_*(1-done)
		adv =  returns- values
		Action = np.array([action])
		critic.fit(np.array([s]),returns, epochs=1, verbose=False)
		policy.fit(x=(np.array([s]),adv,Action),y=(dummy_Prediction),epochs=1, verbose=False)
		rews+=reward
		s=s_
	print(str(episode)+" | "+str(rews))	
	if episode % 2000 == 0:
		save_weights()
	
