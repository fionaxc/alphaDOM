import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import List, Tuple, Dict

class PPOActor(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs)


class PPOCritic(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs)



class PPOAgent:
   def __init__(self, obs_dim: int, action_dim: int, lr: float = 3e-4, gamma: float = 0.99, epsilon: float = 0.2):
        self.actor = PPOActor(obs_dim, action_dim)
        self.critic = PPOCritic(obs_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
    
   def get_action(self, obs: np.ndarray, action_mask: np.ndarray) -> int:
        obs = torch.FloatTensor(obs)
        probs = self.actor(obs)
        probs = probs * torch.FloatTensor(action_mask)
        probs = probs / probs.sum()
        m = Categorical(probs)
        action = m.sample()
        return action.item()
    
   def update(self, obs: np.ndarray, actions: np.ndarray, rewards: np.ndarray, next_obs: np.ndarray, dones: np.ndarray, action_masks: np.ndarray):
        obs = torch.FloatTensor(obs)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_obs = torch.FloatTensor(next_obs)
        dones = torch.FloatTensor(dones)
        action_masks = torch.FloatTensor(action_masks)

        # Compute advantages
        values = self.critic(obs).squeeze()
        next_values = self.critic(next_obs).squeeze()
        advantages = rewards + self.gamma * next_values * (1 - dones) - values
        
        # Compute actor loss
        old_probs = self.actor(obs).gather(1, actions.unsqueeze(1)).squeeze()
        new_probs = self.actor(obs).gather(1, actions.unsqueeze(1)).squeeze()
        ratio = new_probs / old_probs
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        # Compute critic loss
        critic_loss = nn.MSELoss()(values, rewards + self.gamma * next_values * (1 - dones))

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        


