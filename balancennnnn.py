import tensorflow as tf
import gym
import numpy as np
import pandas as pd
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
gamma=0.99
#write a function to save and reload using numpy
def discount_rewards(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r		


env=gym.make('CartPole-v0')
seed = 128
rng = np.random.RandomState(seed)
#policy net
observationsx=tf.placeholder(shape=[None,4],dtype=tf.float32,name='observations')
rewa=tf.placeholder(shape=[None],dtype=tf.float32,name='Rewards')
"""
y1=tf.layers.dense(observationsx,10,activation=tf.nn.relu,use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer())
y2=tf.layers.dense(y1,10,activation=tf.nn.relu,use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer())
y3=tf.layers.dense(y2,30,activation=tf.nn.relu,use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer())
y4=tf.layers.dense(y3,30,activation=tf.nn.relu,use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer())
y5=tf.layers.dense(y4,10,activation=tf.nn.relu,use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer())
yout=tf.layers.dense(y5,2,use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer())
sample_op=tf.multinomial(logits=yout,num_samples=1)
"""
##nn for policy
w1= tf.Variable(tf.random_normal([4, 5], seed=seed))
w2= tf.Variable(tf.random_normal([5, 5], seed=seed))
w3= tf.Variable(tf.random_normal([5, 5], seed=seed))
w4= tf.Variable(tf.random_normal([5, 5], seed=seed))
w5= tf.Variable(tf.random_normal([5, 2], seed=seed))


b1= tf.Variable(tf.random_normal([5], seed=seed))
b2= tf.Variable(tf.random_normal([5], seed=seed))
b3= tf.Variable(tf.random_normal([5], seed=seed))
b4= tf.Variable(tf.random_normal([5], seed=seed))
b5= tf.Variable(tf.random_normal([2], seed=seed))



 

l1 = tf.add(tf.matmul(observationsx, w1),b1)
l1 = tf.nn.relu(l1)

l2 = tf.add(tf.matmul(l1, w2),b2)
l2 = tf.nn.relu(l2)

l3 = tf.add(tf.matmul(l2, w3),b3)
l3 = tf.nn.relu(l3)

l4 = tf.add(tf.matmul(l3, w4),b4)
l4 = tf.nn.relu(l4)

lout = tf.add(tf.matmul(l4, w5),b5)
lout = tf.nn.relu(lout)
sample_op=tf.multinomial(logits=lout,num_samples=1)

##nn for action
observationsx1=tf.placeholder(shape=[None,4],dtype=tf.float32,name='observations1')
w11= tf.Variable(tf.random_normal([4, 5], seed=seed))
w21= tf.Variable(tf.random_normal([5, 5], seed=seed))
w31= tf.Variable(tf.random_normal([5, 5], seed=seed))
w41= tf.Variable(tf.random_normal([5, 5], seed=seed))
w51= tf.Variable(tf.random_normal([5, 2], seed=seed))


b11= tf.Variable(tf.random_normal([5], seed=seed))
b21= tf.Variable(tf.random_normal([5], seed=seed))
b31= tf.Variable(tf.random_normal([5], seed=seed))
b41= tf.Variable(tf.random_normal([5], seed=seed))
b51= tf.Variable(tf.random_normal([2], seed=seed))



    

l11= tf.add(tf.matmul(observationsx1, w11),b11)
l11 = tf.nn.relu(l11)

l21 = tf.add(tf.matmul(l11, w21),b21)
l21 = tf.nn.relu(l21)

l31 = tf.add(tf.matmul(l21, w31),b31)
l31 = tf.nn.relu(l31)

l41 = tf.add(tf.matmul(l31, w41),b41)
l41 = tf.nn.relu(l41)

lout1 = tf.add(tf.matmul(l41, w51),b51)
lout1 = tf.nn.relu(lout1)
sample_op1=tf.multinomial(logits=lout1,num_samples=1)


########################


#loss func
acti=tf.placeholder(shape=[None],dtype=tf.int32,name='Actions')
cross_entropy= tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(acti,2),logits=lout)
loss=tf.reduce_sum(rewa*cross_entropy)

optimiser=tf.train.RMSPropOptimizer(learning_rate=0.01,name='rms')
train_op=optimiser.minimize(loss)
assign1=w11.assign(w1)
assign2=w21.assign(w2)
assign3=w31.assign(w3)
assign4=w41.assign(w4)
assign5=w51.assign(w5)

assign6=b11.assign(b1)
assign7=b21.assign(b2)
assign8=b31.assign(b3)
assign9=b41.assign(b4)
assign10=b51.assign(b5)





#saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=None)
with tf.Session() as sess:
	tf.initialize_all_variables().run()
	episodes=1000000000000
	'''try:
		last_chk_path = tf.train.latest_checkpoint(checkpoint_dir='/home/rig/Desktop/savemodel')
		saver.restore(sess, save_path=last_chk_path)
		print('LOADED')
	except:
		print('Not Loaded')'''
	
	env.reset()
	for ep in range(episodes):

		#saver.save(sess,save_path='/home/rig/Desktop/savemodel',global_step=ep)
		
		
		
		observations=[]
		actions=[]
		rewards=[]
		
		env.reset()
		steps=0
		observation, reward, donein, info = env.step(env.action_space.sample())
		done=False
			
		c=0
		while not done:
			steps=steps+1
			
			env.render()
			
			
			
			action=sess.run(sample_op1,feed_dict={observationsx1:[observation]})
				
			
			
			


			observations.append(observation)
			actions.append(action)
			observation, reward, donein, info = env.step(np.squeeze(action))
			
			#print(observation[2])
			if(abs(observation[2])<0.1):
				
				#reward=5*1/(1+abs(observation[0]))
				reward=3
				c=c+1
			if(abs(observation[2])>0.1):
				reward=-1

			rewards.append(reward)
			if(abs(observation[0])>2 or abs(observation[2])>1.5):
				done = True
		print("balance score:" + str(c))			
		observations=np.asarray(observations)
		actions=np.squeeze(np.asarray(actions))
		actions=actions.flatten()
		rewards=np.array(rewards)
		                             
		disc_rew=discount_rewards(rewards)

		disc_rew=np.asarray(disc_rew) 
		
		sess.run(train_op,feed_dict={observationsx:observations,acti:actions,rewa:(disc_rew)})
		#print(disc_rew)
		if ep%10 == 0:
			sess.run(assign1)
			sess.run(assign2)
			sess.run(assign3)
			sess.run(assign4)
			sess.run(assign5)
			sess.run(assign6)
			sess.run(assign7)
			sess.run(assign8)
			sess.run(assign9)
			sess.run(assign10)
			print("updating weights")	
		print("training episode  "+str(ep+1))	
        
