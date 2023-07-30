import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gym
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm
#load dataset

BATCH_SIZE = 126

class CustomQDataset(Dataset):
    def __init__(self, q_file):
      f = open(q_file)
      self.qfile  = json.load(f)
    def __len__(self):
        return len(self.qfile)
    def __getitem__(self, idx):
        element= self.qfile[idx]
        return torch.tensor(element['state']), torch.tensor(element['action'])\
          ,torch.tensor(element['n_state']),torch.tensor(element['reward']),torch.tensor(element['done'])#s,a,ns,r,d
dataset = CustomQDataset('Qdata.json')
train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

def valueloss(gt,pred, expectile=0.8):
    diff = gt - pred #Q min target - v(s)
    weight = torch.where(diff > 0, expectile, (1 - expectile))
    loss = weight * (diff**2)
    return loss.mean()
#dqn net
class DeepQNetwork(nn.Module):
  """Q(s,a) -> """
  def __init__(self,lr,input_dims,fc1_dims,fc2_dims,n_actions):
    super(DeepQNetwork,self).__init__()
    self.input_dims = input_dims
    self.fc1_dims = fc1_dims
    self.fc2_dims = fc2_dims
    self.n_actions = n_actions
    #layers
    self.fc1 = nn.Linear(self.input_dims+self.n_actions,self.fc1_dims)
    self.fc2  = nn.Linear(self.fc1_dims,self.fc2_dims)
    self.fc3  = nn.Linear(self.fc2_dims,1)
    #optimizer
    self.optimizer = optim.AdamW(self.parameters(),lr=lr,amsgrad=True)
    self.loss =  nn.MSELoss()#nn.SmoothL1Loss()#nn.MSELoss()
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.to(self.device)
  def forward(self,state,action):
    action =F.one_hot(action, num_classes=self.n_actions)
    sa = torch.concat((state,action),dim=1)
    x = F.relu(self.fc1(sa))
    x = F.relu(self.fc2(x))
    Qactions = self.fc3(x)
    return Qactions
  def predict(self, state,action):
    with torch.no_grad():
      state = state.to(self.device)
      action  = action.to(self.device)
      output=self.forward(state,action)
    return output
#dqn net
class DeepVNetwork(nn.Module):
  """V(s) ->v """
  def __init__(self,lr,input_dims,fc1_dims,fc2_dims,n_actions):
    super(DeepVNetwork,self).__init__()
    self.input_dims = input_dims
    self.fc1_dims = fc1_dims
    self.fc2_dims = fc2_dims
    self.n_actions = n_actions
    #layers
    self.fc1 = nn.Linear(self.input_dims,self.fc1_dims)
    self.fc2  = nn.Linear(self.fc1_dims,self.fc2_dims)
    self.fc3  = nn.Linear(self.fc2_dims,1)
    #optimizer
    self.optimizer = optim.AdamW(self.parameters(),lr=lr,amsgrad=True)
    #self.loss = valueloss()
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.to(self.device)
  def forward(self,state):
    x = F.relu(self.fc1(state))
    x = F.relu(self.fc2(x))
    Qactions = self.fc3(x)
    return Qactions
  def predict(self, state):
    with torch.no_grad():
      state = state.to(self.device)
      output=self.forward(state)
    return output
  


#agent
class IQLAgent():
  def __init__(self,gamma,lr,input_dims,n_actions,tau=0.0001):
    self.gamma=gamma
    self.lr=lr
    self.n_actions = n_actions
    self.action_space = [i for i in range(n_actions)]
    self.input_dims = input_dims
    self.TAU = tau


    #q1 nets
    self.Q_primary1 =  DeepQNetwork(lr=self.lr,input_dims=self.input_dims,
                                   fc1_dims=256,fc2_dims=256,n_actions=self.n_actions)
    self.Q_target1 =  DeepQNetwork(lr=self.lr,input_dims=self.input_dims,
                                   fc1_dims=256,fc2_dims=256,n_actions=self.n_actions)
    self.Q_target1.load_state_dict(self.Q_primary1.state_dict())
    #q2 nets
    self.Q_primary2 =  DeepQNetwork(lr=self.lr,input_dims=self.input_dims,
                                   fc1_dims=256,fc2_dims=256,n_actions=self.n_actions)
    self.Q_target2 =  DeepQNetwork(lr=self.lr,input_dims=self.input_dims,
                                   fc1_dims=256,fc2_dims=256,n_actions=self.n_actions)
    self.Q_target2.load_state_dict(self.Q_primary2.state_dict())
    #value net
    self.Value =  DeepVNetwork(lr=self.lr,input_dims=self.input_dims,
                                   fc1_dims=256,fc2_dims=256,n_actions=self.n_actions)

  def learn(self,batch_s,batch_a,batch_ns,batch_r,batch_done):
    
    state_batch = batch_s.to(self.Q_primary1.device)
    new_state_batch = batch_ns.to(self.Q_primary1.device)
    reward_batch = batch_r.to(self.Q_primary1.device)
    terminal_batch = batch_done.to(self.Q_primary1.device)
    action_batch = batch_a.to(self.Q_primary1.device)
    
    q_sa1 = self.Q_primary1(state_batch,action_batch) #q1(s,a)
    q_sa2 = self.Q_primary2(state_batch,action_batch) #q2(s,a)
    v_s = self.Value(state_batch)#v(s)
    with torch.no_grad():
      q_sa1target = self.Q_target1(state_batch,action_batch)#(q1tar(s,a))
      q_sa2target = self.Q_target2(state_batch,action_batch)#(q2tar(s,a))
      Q_satarget = torch.minimum(q_sa1target,q_sa2target).detach()#value target for v(s)
      v_s_ = self.Value(new_state_batch)#v(s_)
      v_s_[terminal_batch] = 0.0 #done = true | computing target for q_sa1 , q_sa2
      
    q_groundtruth =   reward_batch.unsqueeze(1) + self.gamma*v_s_ #q(s,a) = r + (1-done)*gamma*max(Q(s_,a_)) | max(Q(s_,a_))=> v(s_)
     # Compute Q losses
    qloss1 = self.Q_primary1.loss(q_sa1, q_groundtruth)
    qloss2 = self.Q_primary2.loss(q_sa2, q_groundtruth)
     # Optimize the Value function
    v_loss = valueloss(Q_satarget,v_s)
    self.Value.optimizer.zero_grad()
    v_loss.backward()
    self.Value.optimizer.step()
    # optimize the Q function
    self.Q_primary1.optimizer.zero_grad()
    self.Q_primary2.optimizer.zero_grad()
    qloss1.backward()
    # In-place gradient clipping q1
    torch.nn.utils.clip_grad_value_(self.Q_primary1.parameters(), 100)
    self.Q_primary1.optimizer.step()
    qloss2.backward()
    # In-place gradient clipping q2
    torch.nn.utils.clip_grad_value_(self.Q_primary2.parameters(), 100)
    self.Q_primary2.optimizer.step()

    

    # Soft update of the target network's weights
    # θ′ ← τ θ + (1 −τ )θ′
    #soft update target q1
    target_net_state_dict1 = self.Q_target1.state_dict()
    policy_net_state_dict1 = self.Q_primary1.state_dict()
    for key in policy_net_state_dict1:
        target_net_state_dict1[key] = policy_net_state_dict1[key]*self.TAU + target_net_state_dict1[key]*(1-self.TAU)
    self.Q_target1.load_state_dict(target_net_state_dict1)
    #soft update target q2
    target_net_state_dict2 = self.Q_target2.state_dict()
    policy_net_state_dict2 = self.Q_primary2.state_dict()
    for key in policy_net_state_dict2:
        target_net_state_dict2[key] = policy_net_state_dict2[key]*self.TAU + target_net_state_dict2[key]*(1-self.TAU)
    self.Q_target2.load_state_dict(target_net_state_dict2)

    return qloss1.detach().cpu().numpy(),qloss2.detach().cpu().numpy(),v_loss.detach().cpu().numpy()
  def save_models(self,path):
    print("..saving at "+path)
    torch.save(self.Q_primary1.state_dict(),path+"Q_primary1.pth")
    torch.save(self.Q_target1.state_dict(),path+"Q_target1.pth")
    torch.save(self.Q_primary2.state_dict(),path+"Q_primary2.pth")
    torch.save(self.Q_target2.state_dict(),path+"Q_target2.pth")
    torch.save(self.Value.state_dict(),path+"Value.pth")
  def load_models(self,path):
    print("..loading from  "+path)
    self.Q_primary1.load_state_dict(torch.load(path+"Q_primary1.pth"))
    self.Q_target1.load_state_dict(torch.load(path+"Q_target1.pth"))
    self.Q_primary2.load_state_dict(torch.load(path+"Q_primary2.pth"))
    self.Q_target2.load_state_dict(torch.load(path+"Q_target2.pth"))
    self.Value.load_state_dict(torch.load(path+"Value.pth"))



if __name__ == '__main__':
  import gym 
  env = gym.make('CartPole-v1')#, render_mode="human")#gym.make('LunarLander-v2')
  #env = gym.make('LunarLander-v2', render_mode="human")

  agent = IQLAgent(gamma=0.9,lr=0.005,input_dims=env.observation_space.shape[0],
                   n_actions=env.action_space.n,tau=0.01)#gamma,lr,input_dims,n_actions,tau=0.0001
  epochs = 200
  q1_losses=[]
  q2_losses=[]
  v_losses=[]
  for e in range(epochs):
    batch_qloss1,batch_qloss2,batch_v_loss =[],[],[]
    for i,(state,action,nextstate,reward,done) in enumerate(tqdm(train_dataloader)) :
      #print(state,action,nextstate,reward,done)
      qloss1,qloss2,v_loss = agent.learn(batch_s=state,batch_a=action,batch_ns=nextstate,batch_r=reward,batch_done=done)
      batch_qloss1.append(qloss1)
      batch_qloss2.append(qloss2)
      batch_v_loss.append(v_loss)
      
    q1_losses.append(np.average(batch_qloss1))
    q2_losses.append(np.average(batch_qloss2))
    v_losses.append(np.average(batch_v_loss))
    
      
  agent.save_models("./models/")
  env.close()
  plt.plot(q1_losses, 'g', q2_losses, 'r',v_losses ,'y')
  plt.show()


