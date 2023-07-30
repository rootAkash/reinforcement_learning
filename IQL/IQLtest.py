"""instead of training a policy function with AWR here we directly query the q functions to give us near optimal actions"""
import gym
import numpy as np
from tqdm import tqdm
from IQL import IQLAgent
import torch
env = gym.make('CartPole-v1',render_mode = "human")

def randompolicy(obs):
  random_action_policy = np.array([1 for i in range(env.action_space.n)])/env.action_space.n
  return np.random.choice(np.arange(random_action_policy.shape[0]), p=random_action_policy.ravel())
  #return np.argmax(random_action_policy, axis=0)
def greedyQpolicy(agent,observation,actionspace):
  states = []
  actions = []
  for i in range(actionspace):
    states.append(observation)
    actions.append(i)
  qsa1 = agent.Q_primary1.predict(torch.tensor(states),torch.tensor(actions))
  qsa2 = agent.Q_primary2.predict(torch.tensor(states),torch.tensor(actions))
  qsafinal = torch.minimum(qsa1,qsa2)
  action_choosen = torch.argmax(qsafinal.ravel()).cpu().numpy()
  return action_choosen

agent = IQLAgent(gamma=0.9,lr=0.005,input_dims=env.observation_space.shape[0],
                   n_actions=env.action_space.n,tau=0.01)
agent.load_models("./models/")  
n_games=10
max_steps =5000
for i in range(n_games):
  score = 0 
  done = False
  observation = env.reset()[0]
  step = 0
  while not done:
    
    action =greedyQpolicy(agent,observation,env.action_space.n)
    
      
      
    observation_,reward,done,truncated,info = env.step(action)
    if step>max_steps:
      done = True
    observation = observation_
    step+=1
    score+=reward
  print("score achieved : ",score , " for episode: ",i)