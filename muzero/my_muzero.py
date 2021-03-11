import numpy as np
def stcat(x,support=5):
  x = np.sign(x) * ((abs(x) + 1)**0.5 - 1) + 0.001 * x
  x = np.clip(x, -support, support)
  floor = np.floor(x)
  prob = x - floor
  logits = np.zeros( 2 * support + 1)
  first_index = int(floor + support)
  second_index = int(floor + support+1)
  logits[first_index] = 1-prob
  if prob>0:
    logits[second_index] = prob
  return logits
def catts(x,support=5):
  support = np.arange(-support, support+1, 1)
  x = np.sum(support*x)
  x = np.sign(x) * ((((1 + 4 * 0.001 * (abs(x) + 1 + 0.001))**0.5 - 1) / (2 * 0.001))** 2- 1)
  return x  

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm



class MuZeroNet(nn.Module):
    def __init__(self, input_size, action_space_n, reward_support_size, value_support_size):
        super().__init__()
        self.hx_size = 32
        self._representation = nn.Sequential(nn.Linear(input_size, self.hx_size),
                                             nn.Tanh())
        self._dynamics_state = nn.Sequential(nn.Linear(self.hx_size + action_space_n, 64),
                                             nn.Tanh(),
                                             nn.Linear(64, self.hx_size),
                                             nn.Tanh())
        self._dynamics_reward = nn.Sequential(nn.Linear(self.hx_size + action_space_n, 64),
                                              nn.LeakyReLU(),
                                              nn.Linear(64, 2*reward_support_size+1))
        self._prediction_actor = nn.Sequential(nn.Linear(self.hx_size, 64),
                                               nn.LeakyReLU(),
                                               nn.Linear(64, action_space_n))
        self._prediction_value = nn.Sequential(nn.Linear(self.hx_size, 64),
                                               nn.LeakyReLU(),
                                               nn.Linear(64, 2*value_support_size+1))
        self.action_space_n = action_space_n

        self._prediction_value[-1].weight.data.fill_(0)
        self._prediction_value[-1].bias.data.fill_(0)
        self._dynamics_reward[-1].weight.data.fill_(0)
        self._dynamics_reward[-1].bias.data.fill_(0)

    def p(self, state):
        actor_logit = torch.softmax(self._prediction_actor(state),dim=1)
        value = torch.softmax(self._prediction_value(state),dim=1)
        return actor_logit, value

    def h(self, obs_history):
        return self._representation(obs_history)

    def g(self, state, action):
        x = torch.cat((state, action), dim=1)
        next_state = self._dynamics_state(x)
        reward = torch.softmax(self._dynamics_reward(x),dim=1)
        return next_state, reward     

    def initial_state(self, x):
        hout = self.h(x)
        prob,v= self.p(hout)
        return hout,prob,v
    def next_state(self,hin,a):
        hout,r = self.g(hin,a)
        prob,v= self.p(hout)
        return hout,r,prob,v
    def inference_initial_state(self, x):
        with torch.no_grad():
          hout = self.h(x)
          prob,v=self.p(hout)

          return hout,prob,v
    def inference_next_state(self,hin,a):
        with torch.no_grad():
          hout,r = self.g(hin,a)
          prob,v=self.p(hout)
          return hout,r,prob,v     


import torch
import math
import numpy as np

import random
def dynamics(net,state,action):
    #print(state,action) 
    next_state,reward,prob,value = net.inference_next_state(state,torch.tensor([action]).float())
    reward = catts(reward.numpy().ravel())
    value = catts(value.numpy().ravel())
    prob = prob.tolist()[0]
    #print("dynamics",prob)
    return next_state,reward,prob,value


class MinMaxStats:
    """A class that holds the min-max values of the tree."""

    def __init__(self):
        self.MAXIMUM_FLOAT_VALUE = float('inf')       
        self.maximum =  -self.MAXIMUM_FLOAT_VALUE
        self.minimum =  self.MAXIMUM_FLOAT_VALUE

    def update(self, value: float):
        if value is None:
            raise ValueError

        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        # If the value is unknow, by default we set it to the minimum possible value
        if value is None:
            return 0.0

        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values.
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


class Node:
    """A class that represent nodes inside the MCTS tree"""

    def __init__(self, prior: float):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return None
        return self.value_sum / self.visit_count


def softmax_sample(visit_counts, actions, t):
    counts_exp = np.exp(visit_counts) * (1 / t)
    probs = counts_exp / np.sum(counts_exp, axis=0)
    action_idx = np.random.choice(len(actions), p=probs)
    return actions[action_idx]


"""MCTS module: where MuZero thinks inside the tree."""


def add_exploration_noise( node):
    """
    At the start of each search, we add dirichlet noise to the prior of the root
    to encourage the search to explore new actions.
    """
    actions = list(node.children.keys())
    noise = np.random.dirichlet([0.25] * len(actions)) # config.root_dirichlet_alpha
    frac = 0.25#config.root_exploration_fraction
    for a, n in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac



def ucb_score(parent, child,min_max_stats):
    """
    The score for a node is based on its value, plus an exploration bonus based on
    the prior.

    """
    pb_c_base = 19652
    pb_c_init = 1.25
    pb_c = math.log((parent.visit_count + pb_c_base + 1) / pb_c_base) + pb_c_init
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

    prior_score = pb_c * child.prior
    value_score = min_max_stats.normalize(child.value())
    return  value_score + prior_score 

def select_child(node, min_max_stats):
    """
    Select the child with the highest UCB score.
    """
    # When the parent visit count is zero, all ucb scores are zeros, therefore we return a random child
    if node.visit_count == 0:
        return random.sample(node.children.items(), 1)[0]

    _, action, child = max(
        (ucb_score(node, child, min_max_stats), action,
         child) for action, child in node.children.items())
    return action, child




def expand_node(node, to_play, actions_space,hidden_state,reward,policy):
    """
    We expand a node using the value, reward and policy prediction obtained from
    the neural networks.
    """
    node.to_play = to_play
    node.hidden_state = hidden_state
    node.reward = reward
    policy = {a:policy[a] for a in actions_space}
    policy_sum = sum(policy.values())
    for action, p in policy.items():
        node.children[action] = Node(p / policy_sum) # not needed since mine are already softmax but its fine 


def backpropagate(search_path, value,to_play,discount, min_max_stats):
    """
    At the end of a simulation, we propagate the evaluation all the way up the
    tree to the root.
    """
    for node in search_path[::-1]: #[::-1] means reversed
        node.value_sum += value 
        node.visit_count += 1
        min_max_stats.update(node.value())

        value = node.reward + discount * value


def select_action(node, mode ='softmax'):
    """
    After running simulations inside in MCTS, we select an action based on the root's children visit counts.
    During training we use a softmax sample for exploration.
    During evaluation we select the most visited child.
    """
    visit_counts = [child.visit_count for child in node.children.values()]
    actions = [action for action in node.children.keys()]
    action = None
    if mode == 'softmax':
        t = 1.0
        action = softmax_sample(visit_counts, actions, t)
    elif mode == 'max':
        action, _ = max(node.children.items(), key=lambda item: item[1].visit_count)
    counts_exp = np.exp(visit_counts)
    probs = counts_exp / np.sum(counts_exp, axis=0)    
    #return action ,probs,node.value()
    return action ,np.array(visit_counts)/sum(visit_counts),node.value()

def run_mcts(net, state,prob,root_value,num_simulations,discount = 0.9):
    """
    Core Monte Carlo Tree Search algorithm.
    To decide on an action, we run N simulations, always starting at the root of
    the search tree and traversing the tree according to the UCB formula until we
    reach a leaf node.
    """
    prob, root_value = prob.tolist()[0] ,catts(root_value.numpy().ravel())
    to_play = True
    action_space=[ i for i in range(len(prob))]#history.action_space()
    #print("action space",action_space)
    root = Node(0)
    expand_node(root, to_play,action_space,state,0.0,prob)#node, to_play, actions_space ,hidden_state,reward,policy
    add_exploration_noise( root)


    min_max_stats = MinMaxStats()

    for _ in range(num_simulations): 
        node = root
        search_path = [node]

        while node.expanded():
            action, node = select_child( node, min_max_stats)
            search_path.append(node)

        # Inside the search tree we use the dynamics function to obtain the next
        # hidden state given an action and the previous hidden state.
        parent = search_path[-2]
        
        #network_output = network.recurrent_inference(parent.hidden_state, action)
        next_state,r,action_probs, value = dynamics(net,parent.hidden_state,onehot(action,len(action_space))) 
        expand_node(node, to_play, action_space,next_state,r,action_probs)#node, to_play, actions_space ,hidden_state,reward,policy

        backpropagate(search_path, value, to_play, discount, min_max_stats)#search_path, value,,discount, min_max_stats
    return root    
import gym
class ScalingObservationWrapper(gym.ObservationWrapper):
    """
    Wrapper that apply a min-max scaling of observations.
    """

    def __init__(self, env, low=None, high=None):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Box)

        low = np.array(self.observation_space.low if low is None else low)
        high = np.array(self.observation_space.high if high is None else high)

        self.mean = (high + low) / 2
        self.max = high - self.mean

    def observation(self, observation):
        return (observation - self.mean) / self.max


import random
import numpy as np
import torch
def onehot(a,n=2):
  return np.eye(n)[a]
def play_game(env,net,n_sim,discount,render):
    trajectory=[]
    state = env.reset() 
    done = False
    while not done:
        if render:
          env.render()
        h ,prob,value= net.inference_initial_state(torch.tensor([state]).float()) 
        root  = run_mcts(net,h,prob,value,num_simulations=n_sim,discount=discount)
        action,action_prob,mcts_val = select_action(root)
        value=mcts_val 
        next_state, reward, done, info = env.step(action)
        data = (state,onehot(action),action_prob,mcts_val,reward)
        trajectory.append(data)
        state = next_state
    print("played for ",len(trajectory)," steps")   
    return trajectory    

def sample_games(buffer,batch_size):
    # Sample game from buffer either uniformly or according to some priority
    #print("samplig from .",len(buffer))
    return random.choices(buffer, k=batch_size)

def sample_position(trajectory):
    # Sample position from game either uniformly or according to some priority.
    return np.random.randint(0, len(trajectory))


def sample_batch(action_space_size,buffer,discount,batch_size,num_unroll_steps, td_steps):
    obs_batch, action_batch, reward_batch, value_batch, policy_batch = [], [], [], [], []
    games = sample_games(buffer,batch_size)
    for g in games:
      game_pos = sample_position(g)#state index
      state,action,action_prob,root_val,reward = zip(*g)
      state,action,action_prob,root_val,reward =list(state),list(action),list(action_prob),list(root_val),list(reward)

      _actions = action[game_pos:game_pos + num_unroll_steps]
      # random action selection to complete num_unroll_steps
      _actions += [onehot(np.random.randint(0, action_space_size))for _ in range(num_unroll_steps - len(_actions))]

      obs_batch.append(state[game_pos])
      action_batch.append(_actions)
      value, reward, policy = make_target(child_visits=action_prob ,root_values=root_val,rewards=reward,state_index=game_pos,discount=discount, num_unroll_steps=num_unroll_steps, td_steps=td_steps)
      reward_batch.append(reward)
      value_batch.append(value)
      policy_batch.append(policy)

    obs_batch = torch.tensor(obs_batch).float()
    action_batch = torch.tensor(action_batch).long()
    reward_batch = torch.tensor(reward_batch).float()
    value_batch = torch.tensor(value_batch).float()
    policy_batch = torch.tensor(policy_batch).float()
    return obs_batch, action_batch, reward_batch, value_batch, policy_batch


def make_target(child_visits,root_values,rewards,state_index,discount=0.99, num_unroll_steps=5, td_steps=10):
        # The value target is the discounted root value of the search tree N steps into the future, plus
        # the discounted sum of all rewards until then.
        target_values, target_rewards, target_policies = [], [], []
        for current_index in range(state_index, state_index + num_unroll_steps + 1):
            bootstrap_index = current_index + td_steps
            if bootstrap_index < len(root_values):
                value = root_values[bootstrap_index] * discount ** td_steps
            else:
                value = 0

            for i, reward in enumerate(rewards[current_index:bootstrap_index]):
                value += reward * discount ** i

            if current_index < len(root_values):
                target_values.append(stcat(value))
                target_rewards.append(stcat(rewards[current_index]))
                target_policies.append(child_visits[current_index])

            else:
                # States past the end of games are treated as absorbing states.
                target_values.append(stcat(0))
                target_rewards.append(stcat(0))
                # Note: Target policy is  set to 0 so that no policy loss is calculated for them
                #target_policies.append([0 for _ in range(len(child_visits[0]))])
                target_policies.append(child_visits[0]*0.0)

        return target_values, target_rewards, target_policies


def scalar_reward_loss( prediction, target):
        return -(torch.log(prediction) * target).sum(1)

def scalar_value_loss( prediction, target):
        return -(torch.log(prediction) * target).sum(1)
def update_weights(model, action_space_size, optimizer, replay_buffer,discount,batch_size,num_unroll_steps, td_steps ):
    batch = sample_batch(action_space_size,replay_buffer,discount,batch_size,num_unroll_steps, td_steps)
    obs_batch, action_batch, target_reward, target_value, target_policy = batch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    obs_batch = obs_batch.to(device)
    action_batch = action_batch.to(device)#.unsqueeze(-1) # its not onehot yet 
    target_reward = target_reward.to(device)
    target_value = target_value.to(device)
    target_policy = target_policy.to(device)

    # transform targets to categorical representation # its already done
    # Reference:  Appendix F
    #transformed_target_reward = config.scalar_transform(target_reward)
    target_reward_phi =target_reward #config.reward_phi(transformed_target_reward)
    #transformed_target_value = config.scalar_transform(target_value)
    target_value_phi = target_value#config.value_phi(transformed_target_value)

    hidden_state, policy_prob,value  = model.initial_state(obs_batch) # initial model_call ###################################### make changes
    #h,init_pred_p,init_pred_v = net.initial_state(in_s)

    value_loss = scalar_value_loss(value, target_value_phi[:, 0])
    policy_loss = -(torch.log(policy_prob) * target_policy[:, 0]).sum(1)
    reward_loss = torch.zeros(batch_size, device=device)

    gradient_scale = 1 / num_unroll_steps
    for step_i in range(num_unroll_steps):
        hidden_state, reward,policy_prob,value  = model.next_state(hidden_state, action_batch[:, step_i]) ######################### make changes
        #h,pred_reward,pred_policy,pred_value= net.next_state(h,act)
        policy_loss += -(torch.log(policy_prob) * target_policy[:, step_i + 1]).sum(1)
        value_loss += scalar_value_loss(value, target_value_phi[:, step_i + 1])
        reward_loss += scalar_reward_loss(reward, target_reward_phi[:, step_i])
        hidden_state.register_hook(lambda grad: grad * 0.5)

    # optimize
    value_loss_coeff = 1
    loss = (policy_loss + value_loss_coeff * value_loss + reward_loss) # find value loss coefficiet = 1?
    weights = 1
    weighted_loss = (weights * loss).mean()#1?
    weighted_loss.register_hook(lambda grad: grad * gradient_scale)
    loss = loss.mean()

    optimizer.zero_grad()
    weighted_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5)#5?
    optimizer.step()

def adjust_lr(optimizer, step_count):

    lr_init=0.05
    lr_decay_rate=0.01
    lr_decay_steps=10000
    lr = lr_init * lr_decay_rate ** (step_count / lr_decay_steps)
    lr = max(lr, 0.001)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


    
def net_train(net,  action_space_size, replay_buffer,discount,batch_size,num_unroll_steps, td_steps):
    model =MuZeroNet(input_size=4, action_space_n=2, reward_support_size=5, value_support_size=5) #net
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9,weight_decay=1e-4)
    training_steps=2000#20000
    # wait for replay buffer to be non-empty
    while len(replay_buffer) == 0:
        pass

    for step_count in tqdm(range(training_steps)):
        lr = adjust_lr( optimizer, step_count)
        update_weights(model, action_space_size, optimizer, replay_buffer,discount,batch_size,num_unroll_steps, td_steps)

    return model

import gym
import numpy as np
buffer =[]


episodes_per_train=20
training_steps=50
epochs=50
n_sim= 50
discount = 0.99
batch_size = 128
envs = ['CartPole-v1']
device = 'cuda' if torch.cuda.is_available() else 'cpu'
env=gym.make(envs[0])
env = ScalingObservationWrapper(env, low=[-2.4, -2.0, -0.42, -3.5], high=[2.4, 2.0, 0.42, 3.5])
render =True
#env=env.unwrapped

s_dim =env.observation_space.shape[0]
print("s_dim: ",s_dim)
a_dim =env.action_space.n
print("a_dim: ",a_dim)
a_bound =1 #env.action_space.high[0]
print("a_bound: ",a_bound)



net = MuZeroNet(input_size=4, action_space_n=2, reward_support_size=5, value_support_size=5)

for t in range(training_steps):
  for _ in range(episodes_per_train):
    buffer.append(play_game(env,net,n_sim,discount,render))
  print("training")  
  net = net_train(net,  action_space_size=a_dim, replay_buffer=buffer,discount=discount,batch_size=batch_size,num_unroll_steps=5, td_steps=5)
  

