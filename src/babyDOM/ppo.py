import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import List, Tuple, Dict
from game_engine.action import Action

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
    def __init__(self, obs_dim: int, action_dim: int, lr: float = 3e-4, gamma: float = 0.99, epsilon: float = 0.2, value_coef: float = 0.5, entropy_coef: float = 0.01):
        self.actor = PPOActor(obs_dim, action_dim)
        self.critic = PPOCritic(obs_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

    def get_action(self, obs: np.ndarray, valid_actions: List[Action]) -> Tuple[Action, float]:
        obs = torch.FloatTensor(obs)
        logits = self.actor(obs)

        # Create an action mask based on the valid actions
        action_mask = torch.zeros(logits.size(), dtype = torch.bool)
        valid_indices = [vectorizer.action_to_index(action) for action in valid_actions]
        action_mask[valid_indices] = True

        # Apply mask and create categorical distribution
        maksed_logits = logits.masked_fill(~action_mask, -np.inf)
        probs = torch.softmax(maksed_logits, dim = -1)
        m = Categorical(probs)

        action_index = m.sample()
        selected_action = valid_actions[valid_indices.index(action_index.item())]
        return selected_action, m.log_prob(action_index).item()
   
    def get_value(self, obs: np.ndarray) -> float:
        obs = torch.FloatTensor(obs)
        return self.critic(obs).item()
   
    def update(self, observations: List[np.ndarray], actions: List[int], old_log_probs: List[float], 
               rewards: List[float], values: List[float], dones: List[bool], next_value: float, 
               epochs: int, vectorizer: DominionVectorizer):
        
        observations = torch.FloatTensor(observations)
        actions = torch.LongTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        rewards = torch.FloatTensor(rewards)
        values = torch.FloatTensor(values)
        dones = torch.FloatTensor(dones)

        # Compute returns and advantages
        returns = self.compute_returns(rewards, dones, next_value)
        advantages = returns - values

        for _ in range(epochs):
            # Get new log probs and values
            new_logits = self.actor(observations)
            new_values = self.critic(observations).squeeze()
            
            # Create action distribution
            new_probs = torch.softmax(new_logits, dim=-1)
            dist = Categorical(new_probs)
            
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy()

            # Compute ratio and surrogate loss
            ratio = torch.exp(new_log_probs - old_log_probs)
            surrogate1 = ratio * advantages
            surrogate2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            actor_loss = -torch.min(surrogate1, surrogate2).mean()

            # Compute value loss
            value_loss = nn.MSELoss()(new_values, returns)

            # Compute total loss
            loss = actor_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()

            # Update actor and critic
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
    
    def compute_returns(self, rewards: torch.Tensor, dones: torch.Tensor, next_value: torch.Tensor) -> torch.Tensor:
        returns = torch.zeros_like(rewards)
        running_return = next_value
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.gamma * running_return * (1 - dones[t])
            returns[t] = running_return
        return returns


def ppo_train(
        game_engine: Game,
        vectorizer: DominionVectorizer,
        num_episodes: int = 1000,
        batch_size: int = 32,
        update_epochs: int = 4
) -> Tuple[PPOAgent, PPOAgent]:
    
    obs_dim = vectorizer.observation_dim
    action_dim = vectorizer.action_space_size
    player1 = PPOAgent(obs_dim, action_dim)
    player2 = PPOAgent(obs_dim, action_dim)

    for episode in range(num_episodes):
        game_engine = Game()  # Reset the game state
        done = False
        episode_rewards = [0, 0]
        
        observations = [[], []]
        actions = [[], []]
        log_probs = [[], []]
        rewards = [[], []]
        values = [[], []]
        dones = [[], []]

        while not done:
            current_player = game_engine.current_player_turn
            agent = player1 if current_player == 0 else player2
            
            game_state = game_engine.get_observation_state()
            obs = vectorizer.vectorize(game_state)
            valid_actions = game_engine.get_valid_actions()
            
            action, log_prob = agent.get_action(obs, valid_actions, vectorizer)
            value = agent.get_value(obs)
            
            action.apply()
            
            done = game_engine.game_over
            reward = game_engine.players[current_player].victory_points() if done else 0
            
            observations[current_player].append(obs)
            actions[current_player].append(vectorizer.action_to_index(action))
            log_probs[current_player].append(log_prob)
            rewards[current_player].append(reward)
            values[current_player].append(value)
            dones[current_player].append(done)
            
            episode_rewards[current_player] += reward

            if len(observations[current_player]) >= batch_size:
                agent.update(
                    observations[current_player],
                    actions[current_player],
                    log_probs[current_player],
                    rewards[current_player],
                    values[current_player],
                    dones[current_player],
                    next_value=agent.get_value(obs),
                    epochs=update_epochs,
                    vectorizer=vectorizer
                )
                observations[current_player] = []
                actions[current_player] = []
                log_probs[current_player] = []
                rewards[current_player] = []
                values[current_player] = []
                dones[current_player] = []
        
        # End of episode update
        for current_player in [0, 1]:
            agent = player1 if current_player == 0 else player2
            if len(observations[current_player]) > 0:
                agent.update(
                    observations[current_player],
                    actions[current_player],
                    log_probs[current_player],
                    rewards[current_player],
                    values[current_player],
                    dones[current_player],
                    next_value=0,
                    epochs=update_epochs,
                    vectorizer=vectorizer
                )

        print(f"Episode {episode + 1}, Rewards: Player 1 = {episode_rewards[0]}, Player 2 = {episode_rewards[1]}")

    return player1, player2