import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import List, Tuple, Dict
from game_engine.game import Game
from vectorization.vectorizer import DominionVectorizer
from game_engine.action import Action
import pandas as pd
import os
import csv

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
    def __init__(self, obs_dim: int, 
                 action_dim: int, 
                 hidden_size: int, 
                 lr: float = 3e-4, 
                 gamma: float = 0.99, 
                 epsilon: float = 0.2, 
                 value_coef: float = 0.5, 
                 entropy_coef: float = 0.01):
        self.actor = PPOActor(obs_dim, action_dim, hidden_size)
        self.critic = PPOCritic(obs_dim, hidden_size)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

    def get_action(self, obs: np.ndarray, game: Game, vectorizer: DominionVectorizer) -> Tuple[Action, float]:
        obs = torch.FloatTensor(obs)
        logits = self.actor(obs)

        # Get the action mask directly from the vectorizer
        action_mask = vectorizer.get_action_mask(game)
        action_mask = torch.BoolTensor(action_mask)

        # Apply mask and create categorical distribution
        masked_logits = logits.masked_fill(~action_mask, -float('inf'))
        probs = torch.softmax(masked_logits, dim=-1)
        m = Categorical(probs)

        action_index = m.sample()
        # Use devectorize_action instead of index_to_action
        selected_action = vectorizer.devectorize_action(action_index.item(), game.current_player())
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


def record_game_history(game_engine, vectorizer, episode, turn_counter, current_player, action, reward, cumulative_reward):
    observation_state = game_engine.get_observation_state()
    valid_actions = game_engine.get_valid_actions()
    
    return {
        'Episode': episode + 1,
        'Turn': turn_counter,
        'Player': current_player + 1,
        'Action': str(action),
        'Reward': reward,
        'Cumulative_Reward': cumulative_reward,
        'Game_State': observation_state,
        'Valid_Actions': [str(a) for a in valid_actions]
    }

def ppo_train(
        game_engine: Game,
        vectorizer: DominionVectorizer,
        run_id: str,
        output_dir: str,
        num_episodes: int,
        batch_size: int,
        update_epochs: int,
        hidden_size: int
) -> Tuple[PPOAgent, PPOAgent]:
    
    obs_dim = vectorizer.vectorize_observation(game_engine).shape[0]
    action_dim = vectorizer.action_space_size
    player1 = PPOAgent(obs_dim, action_dim, hidden_size)
    player2 = PPOAgent(obs_dim, action_dim, hidden_size)
    
    # Create a directory for storing output files
    os.makedirs(output_dir, exist_ok=True)

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

        # Initialize game history
        game_history = []

        # Initialize variables for tracking game state
        turn_counter = 0
        cumulative_rewards = [0, 0]

        while not done:
            current_player = game_engine.current_player_turn
            agent = player1 if current_player == 0 else player2
            
            obs = vectorizer.vectorize_observation(game_engine)
            action_mask = vectorizer.get_action_mask(game_engine)
            
            action, log_prob = agent.get_action(obs, game_engine, vectorizer)
            value = agent.get_value(obs)

            action.apply()

            done = game_engine.game_over
            reward = game_engine.players[current_player].victory_points() if done else 0

            # Log the game state after the action and reward calculation
            game_history.append({
                'turn': turn_counter,
                'player': game_engine.current_player().name,
                'action': str(action),
                'state': game_engine.get_game_state_string(),
                'reward': reward
            })

            observations[current_player].append(obs)
            actions[current_player].append(vectorizer.vectorize_action(action))
            log_probs[current_player].append(log_prob)
            rewards[current_player].append(reward)
            values[current_player].append(value)
            dones[current_player].append(done)
            
            episode_rewards[current_player] += reward
            cumulative_rewards[current_player] += reward

            # Update the reward for the last action if the game is over
            if done:
                game_history[-1]['reward'] = reward

            turn_counter += 1

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
        
        # Save game history to CSV
        with open(os.path.join(output_dir, f"game_history_{episode+1}.csv"), "w", newline='') as f:
            fieldnames = ['turn', 'player', 'action', 'state', 'reward']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(game_history)

        # Reset game history for the next episode
        game_history = []

    return player1, player2