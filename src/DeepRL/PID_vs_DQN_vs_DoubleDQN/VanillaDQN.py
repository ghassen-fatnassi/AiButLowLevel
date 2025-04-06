import gym
import torch
import torch.nn as nn
import torch.functional as F
import random as rd

rd.seed(18)
torch.manual_seed(777)

class Q(nn.Module):
   def __init__(self,state_size,action_size):
      super.__init__(self)
      self.input=state_size
      self.output=action_size
      self.NN=nn.Sequential([
         nn.Linear(self.input,16),
         nn.ReLU(),
         nn.Linear(16,32),
         nn.ReLU(),
         nn.Linear(32,self.output)
      ])
   def forward(self,state):
      return self.NN(state)
   

env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset(seed=42)
iterations=None
eps=None
q_function=Q()
for _ in range(iterations):
   if (rd.random()>eps):
      #sample random action
      action=None
      pass
   else:
      #action = argmax(q_function(observation))
      
   
   action=None
   observation, reward, terminated, truncated, info = env.step(action)
   if terminated or truncated:
      observation, info = env.reset()
env.close()