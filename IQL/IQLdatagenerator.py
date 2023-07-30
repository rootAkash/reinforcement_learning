#collect data on an env using a random policy and then make a json


import json
import gym
import numpy as np
from tqdm import tqdm
env = gym.make('CartPole-v1')#,render_mode = "human")
data = []
def policy(obs):
  random_action_policy = np.array([1 for i in range(env.action_space.n)])/env.action_space.n
  return np.random.choice(np.arange(random_action_policy.shape[0]), p=random_action_policy.ravel())
  #return np.argmax(random_action_policy, axis=0)
def store_transition(observation ,action,reward,observation_,done):
  data.append({'state':observation.tolist(),'action':action.tolist(),'n_state':observation_.tolist(),'n_action':None,'reward':reward,'done':done})#s,a,ns,r,done

  

n_games=100
max_steps =5000
for i in tqdm(range(n_games)):
  score = 0 
  done = False
  observation = env.reset()[0]
  step = 0
  while not done:
    
    action =policy(observation)
    observation_,reward,done,truncated,info = env.step(action)
    if step>max_steps:
      done = True
    store_transition(observation ,action,reward,observation_,done)
    observation = observation_
    step+=1

env.close()
with open('Qdata.json', 'w') as fout:
    json.dump(data, fout)