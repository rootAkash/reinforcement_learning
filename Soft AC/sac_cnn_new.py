import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random
from tqdm import tqdm
import cv2
class Policy(nn.Module):

    def __init__(self,observation_size,action_size,hidden_units):
        super().__init__()
        self.observation_size=observation_size #4,50,50
        self.hidden_units=hidden_units
        self.action_size = action_size
        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.h1 = nn.Conv2d(in_channels=self.observation_size[0], out_channels=64, kernel_size=3, padding=1,stride =2)
        self.h2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1,stride =2)
        #self.h3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1,stride =1)
        #self.h4 = nn.Conv2d(in_channels=64, out_channels=2*64, kernel_size=3, padding=1,stride =2)
        #self.h5 = nn.Conv2d(in_channels=2*64, out_channels=2*64, kernel_size=3, padding=1,stride =1)
        self.conv_out_dim = self.conv_op(torch.zeros((1,self.observation_size[0],self.observation_size[1],self.observation_size[2])))
        self.flat_dim = self.conv_out_dim[1]*self.conv_out_dim[2]*self.conv_out_dim[3] 
        #print(self.conv_out_dim,self.flat_dim)
        self.h6 = nn.Linear(self.flat_dim,self.hidden_units)
        self.h7 = nn.Linear(self.hidden_units,self.hidden_units)
        self.mu = nn.Linear(self.hidden_units,self.action_size)
        self.sigma = nn.Linear(self.hidden_units,self.action_size)
        
    def conv_op(self, x):
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        #x = F.relu(self.h3(x))
        #x = F.relu(self.h4(x))
        #x = F.relu(self.h5(x))
        return x.shape
    
    def forward(self, x):
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        #x = F.relu(self.h3(x))
        #x = F.relu(self.h4(x))
        #x = F.relu(self.h5(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.h6(x))
        x = F.relu(self.h7(x))
        mus = torch.tanh(self.mu(x))
        sigs= F.softplus(self.sigma(x))
        sigs= torch.clamp(sigs, min=0.001, max=100)#1e-22 , 1e+02
        return mus , sigs
    def predict(self, x):
        with torch.no_grad():
          output=self.forward(x)
        return output  
class Q_net(nn.Module):

    def __init__(self,observation_size,action_size,hidden_units):
        super().__init__()
        self.observation_size=observation_size
        self.hidden_units=hidden_units
        self.action_size = action_size
        self.a1 = nn.Linear(self.action_size, self.hidden_units) 
        self.h1 = nn.Conv2d(in_channels=self.observation_size[0], out_channels=64, kernel_size=3, padding=1,stride =2)
        self.h2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1,stride =2)
        #self.h3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1,stride =1)
        #self.h4 = nn.Conv2d(in_channels=64, out_channels=64*2, kernel_size=3, padding=1,stride =2)
        #self.h5 = nn.Conv2d(in_channels=64*2, out_channels=64*2, kernel_size=3, padding=1,stride =1)
        self.conv_out_dim = self.conv_op(torch.zeros((1,self.observation_size[0],self.observation_size[1],self.observation_size[2])))
        self.flat_dim = self.conv_out_dim[1]*self.conv_out_dim[2]*self.conv_out_dim[3]
        #print(self.conv_out_dim,self.flat_dim)
        self.h6 = nn.Linear(self.hidden_units+self.flat_dim,self.hidden_units)
        self.h7 = nn.Linear(self.hidden_units,self.hidden_units)
        #self.h8 = nn.Linear(self.hidden_units,self.hidden_units)
        self.q = nn.Linear(self.hidden_units,1)
    def conv_op(self, x):
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        #x = F.relu(self.h3(x))
        #x = F.relu(self.h4(x))
        #x = F.relu(self.h5(x))
        return x.shape    
    def forward(self, x,a):
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        #x = F.relu(self.h3(x))
        #x = F.relu(self.h4(x))
        #x = F.relu(self.h5(x))
        x = x.view(x.size(0), -1)
        a = F.relu(self.a1(a))
        x = torch.cat([x,a], dim=1)
        x = F.relu(self.h6(x))
        x = F.relu(self.h7(x))
        #x = F.relu(self.h8(x))
        qout = self.q(x)
        return qout
    def predict(self, x,a):
        with torch.no_grad():
          output=self.forward(x,a)
        return output
def preprocess(s_hist):
  s_h = np.array(s_hist.copy())/255.0 

  return (s_h -0.5)*2
def remember(s,a,r,ns,d):
  #s=s.ravel()
  #ns=ns.ravel()
  memory.append([s,a,np.array([r]),ns,np.array([d])])
def sample_games(buffer,batch_size):
  # Sample game from buffer either uniformly or according to some priority
  #print("samplig from .",len(buffer))
  return list(np.random.choice(len(buffer),batch_size))
def soft_update(target, source, tau):
  for target_param, param in zip(target.parameters(), source.parameters()):
    target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
  return target
@torch.jit.script
def get_log_pdf_multi(x,mean,std):
  k= x.shape[1] #action dim
  pi = 3.1415926
  cov = std**2
  det = torch.prod(cov,dim=1,keepdim=True)
  #norm_const = 1.0/ ( np.power(2*pi,k/2) * torch.pow(det,0.5) )
  norm_const = 1.0/ ( 2*pi**k/2 * torch.pow(det,0.5) )
  prod  = (1/cov)*torch.square(x - mean)
  prod2 =torch.sum(prod,dim=1,keepdim=True) 
  pdf = norm_const * torch.exp( -0.5 *prod2)
  final_log_pdf = torch.log(pdf+1e-07)
  return final_log_pdf
@torch.jit.script
def get_entropy_multi(x, mean, std,Act):
  #log pdf (squashed guassian) = log pdfguassian(mupolicy,sigma_policy) - sum of  log(1-A**2) ; where each A is component of tanh squahed action vector 
  log_pdf_final = get_log_pdf_multi(x,mean,std) - torch.sum(torch.log(1- torch.square(Act) +1e-07),dim=1,keepdim=True)
  return -log_pdf_final


def replay_and_train(target_entropy,log_alpha,policy,q_1,q_2,t_q_1,t_q_2,popt,qopt,log_alpha_opt,size=128):
  #target entropy is a constant so no need to make it into torch tensor
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  gamma=0.95
  mse = nn.MSELoss()
  alpha = np.exp(log_alpha.item())#get new alpha value
  sample_size=size
  if len(memory) < sample_size:
    return
  samples=random.sample(memory,sample_size)
  s,a,r,ns,d=zip(*samples)

  #s,a,r,ns,d = list(s),list(a),list(r),list(ns),list(d)
  #print(s,a,r,ns,d)
  s= torch.tensor(s).float().to(device)
  a= torch.tensor(a).float().to(device)
  r= torch.tensor(r).float().to(device)
  ns= torch.tensor(ns).float().to(device)
  d= torch.tensor(d).float().to(device)

  new_n_mu,new_n_sig = policy.predict(ns)
  n_E =np.random.multivariate_normal(np.zeros_like(new_n_mu[0].cpu().numpy()),np.diag(np.ones_like(new_n_mu[0].cpu().numpy())),size) 
  n_E = torch.tensor(n_E).float().to(device)
  n_Act_guassian = new_n_mu + n_E*new_n_sig # guassian action using reparametrisation
  new_next_Action  =torch.tanh(n_Act_guassian)  #final squashed guassian action
  n_entropy = get_entropy_multi(n_Act_guassian,new_n_mu,new_n_sig,new_next_Action)
  yq = r + gamma*(1-d)*torch.minimum(t_q_1.predict(ns,new_next_Action),t_q_2.predict(ns,new_next_Action)) + alpha*n_entropy # value target should be bootstrapped to current policy
  #training nets 
  #train q nets
  qloss = mse(q_1(s,a),yq) + mse(q_2(s,a),yq)
  qopt.zero_grad()                                                                                                          #    
  qloss.backward()                                                                                                         #
  qopt.step() 
  #train policy
  new_mu,new_sig = policy(s)
  E =np.random.multivariate_normal(np.zeros_like(new_mu[0].detach().cpu().numpy()),np.diag(np.ones_like(new_mu[0].detach().cpu().numpy())),size) 
  E = torch.tensor(E).float().to(device)
  Act_guassian = new_mu + E*new_sig # guassian action using reparametrisation
  new_Action  =torch.tanh(Act_guassian)  #final squashed guassian action
  entropy = get_entropy_multi(Act_guassian,new_mu,new_sig,new_Action)

  policy_objective = torch.minimum(q_1(s,new_Action),q_2(s,new_Action))#state from buffer action from recent policy and not from buffer to train to maximise q
  final_policy_objective = policy_objective + alpha*entropy # maximise this
  final_policy_loss = - torch.mean(final_policy_objective)  #therefore minimise this
  popt.zero_grad()                                                                                                          #    
  final_policy_loss.backward()                                                                                                         #
  popt.step() 
  #train alpha
  log_alpha_loss = torch.mean(log_alpha*(entropy.detach() - target_entropy))
  log_alpha_opt.zero_grad()                                                                                                          #    
  log_alpha_loss.backward()                                                                                                         #
  log_alpha_opt.step() 
  #train t_value
  soft_update(target=t_q_1, source=q_1, tau=0.005)
  soft_update(target=t_q_2, source=q_2, tau=0.005)
  return qloss.item(),final_policy_loss.item(),log_alpha_loss.item(),torch.mean(entropy).item()
class state_buffer():
    def __init__(self,state_shape):
        self.buff=np.zeros(state_shape)
        self.state_shape = state_shape
        self.size =state_shape[0]
    def append(self,state):
        self.buff[0:self.size-1] = self.buff[1:self.size]  
        self.buff[self.size-1] = state
    def rec_buff(self):
        ret = np.copy(self.buff)
        return ret
    def flat_buff(self):
        ret =np.copy(self.buff.ravel())
        return ret
    def reset(self):
        self.buff=np.zeros(self.state_shape)

  


import gym

memory=deque(maxlen=5000000)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#env,networks ad optimisers
env=gym.make('Pendulum-v0')
#env=gym.make('MountainCarContinuous-v0')
#env=gym.make('LunarLanderContinuous-v2')
env=env.unwrapped
print("device:",device)
s_dim = (3,64,64) #env.observation_space.shape[0]
print(s_dim)
a_dim = env.action_space.shape[0]
print(a_dim)
a_bound = env.action_space.high[0]
print(a_bound)
v_hist = state_buffer(state_shape=s_dim)
#########################################################################
target_entropy = -a_dim# -dim(A)
log_alpha = torch.tensor([0.0], requires_grad=True).to(device)
policy = Policy(s_dim,a_dim,256).to(device)
q_1 = Q_net(s_dim,a_dim,256).to(device)
q_2 = Q_net(s_dim,a_dim,256).to(device)
t_q_1 = Q_net(s_dim,a_dim,256).to(device)
t_q_2 = Q_net(s_dim,a_dim,256).to(device)
soft_update(target=t_q_1, source=q_1, tau=1)
soft_update(target=t_q_2, source=q_2, tau=1)
qopt = optim.Adam(list(q_1.parameters()) + list(q_2.parameters()),lr=0.001)
popt = optim.Adam(policy.parameters(),lr=0.001)      
log_alpha_opt =  optim.Adam(params=[log_alpha],lr=0.001)
#########################################################################
max_steps=5000
train_iter = 1000  # 1000, batch of 8/16 works best for now
test_eps = 335#after this no training
warmup = 20 # before this only buffer is filled without training
episodes = test_eps+200
for e in range(episodes):
  v_hist.reset()
  done = False
  num_s = env.reset()
  v_state = cv2.resize(cv2.cvtColor(env.render(mode="rgb_array"), cv2.COLOR_BGR2GRAY), (s_dim[2],s_dim[1]))
  v_hist.append(v_state) 
  s =  preprocess(v_hist.rec_buff())
  rew = 0 
  stp=0
  while not done:
    #get action policy
    new_mu,new_sig = policy.predict(torch.tensor([s]).float().to(device))
    new_mu,new_sig=new_mu.cpu().numpy(),new_sig.cpu().numpy()
    E =np.random.multivariate_normal(np.zeros_like(new_mu[0]),np.diag(np.ones_like(new_mu[0])))#np.random.normal(mu=0,std_dev=1) diag cov of 1 is same as std dev of 1
    Act_guassian = new_mu[0] + new_sig[0]*E # guassian action using reparametrisation
    act  =np.tanh(Act_guassian)
    ################################
    if e>test_eps-10:
      env.render()
      #act = new_mu[0] # testing
    num_s_,r,done,_=env.step(act*a_bound)
    v_state_ =  cv2.resize(cv2.cvtColor(env.render(mode="rgb_array"), cv2.COLOR_BGR2GRAY), (s_dim[2],s_dim[1]))
    v_hist.append(v_state_) 
    s_ =  preprocess(v_hist.rec_buff())
    if stp>max_steps:
      done = True
    if e<=test_eps:    
      remember(s,act,r,s_,done)
    s=s_
    v_state=v_state_
    num_s=num_s_
    rew+=r
    stp+=1
  print(e,rew)
  if e>warmup and e <= test_eps:
    print("training")
    p=0
    q=0
    a=0
    e=0
    for i in  tqdm(range(train_iter)):
      ql,pl,al,ee=replay_and_train(target_entropy,log_alpha,policy,q_1,q_2,t_q_1,t_q_2,popt,qopt,log_alpha_opt,size=16)
      p+=pl
      q+=ql
      a+=al
      e+=ee
    print("p loss",p/train_iter,"|q loss ",q/train_iter,"|alpha loss ",a/train_iter,"|avg entropy ",np.round(e/train_iter,3))  
  