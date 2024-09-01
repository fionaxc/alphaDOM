import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import List, Tuple
from game_engine.game import Game
from vectorization.vectorizer import DominionVectorizer
from game_engine.action import Action
from .utils import convert_action_probs_to_readable
import os
import csv
import copy
import logging

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
                 lr: float = 1e-4, 
                 gamma: float = 0.99, 
                 epsilon: float = 0.2, 
                 value_coef: float = 0.5, 
                 entropy_coef: float = 0.01, 
                 gae_lambda: float = 0.95,
                 output_dir: str = "src/output"):
        """
        Initialize the PPOAgent that uses a combined loss function for actor and critic.

        Args:
            obs_dim (int): Dimension of the observation space.
            action_dim (int): Dimension of the action space.
            hidden_size (int): Size of the hidden layers in the neural network.
            lr (float): Learning rate for the optimizer.
            gamma (float): Discount factor for future rewards.
            epsilon (float): Clipping parameter for the PPO update.
            value_coef (float): Coefficient for the value loss in the combined loss function.
            entropy_coef (float): Coefficient for the entropy loss in the combined loss function.
            gae_lambda (float): Lambda parameter for the Generalized Advantage Estimation (GAE).
        """
        self.actor = PPOActor(obs_dim, action_dim, hidden_size)
        self.critic = PPOCritic(obs_dim, hidden_size)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.gae_lambda = gae_lambda

        # Paths to the gradient log files
        self.actor_gradient_log_path = os.path.join(output_dir, 'actor_gradient_log.txt')
        self.critic_gradient_log_path = os.path.join(output_dir, 'critic_gradient_log.txt')
        
        # Create the gradient log files if they don't exist
        for path in [self.actor_gradient_log_path, self.critic_gradient_log_path]:
            if not os.path.exists(path):
                with open(path, 'w') as f:
                    f.write('')  # Create an empty file

    def get_action(self, obs: np.ndarray, game: Game, vectorizer: DominionVectorizer) -> Tuple[Action, float, torch.Tensor]:
        """
        Get the action from the agent's policy. Returns the selected action and the log probability of the action (used for updating the policy).
        """
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
        selected_action = vectorizer.devectorize_action(action_index.item(), game.current_player())
        return selected_action, m.log_prob(action_index).item(), probs

    def get_value(self, obs: np.ndarray) -> float:
        """
        Get the value from the critic.
        """
        obs = torch.FloatTensor(obs)
        return self.critic(obs).item()
    
    def calculate_reward(self, game_engine: Game, current_player: int, done: bool) -> int:
        """
        Calculate the reward for the current player.
        
        Args:
            game_engine (Game): The game engine instance.
            current_player (int): The index of the current player.
            done (bool): Whether the game is over.
        
        Returns:
            float: The calculated reward.
        """
        # Winning reward
        if done and game_engine.winner() is not None and game_engine.winner().name == game_engine.players[current_player].name: 
            return 1
        # Tie reward
        elif done and game_engine.winner() is None: 
            return 0.2
        # Losing or continuing reward
        else:
            return 0

    def update(self, observations: List[np.ndarray], actions: List[int], old_log_probs: List[float], 
               rewards: List[float], values: List[float], dones: List[bool], next_value: float, 
               epochs: int, vectorizer: DominionVectorizer):
        observations = torch.FloatTensor(np.array(observations))
        actions = torch.LongTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        rewards = torch.FloatTensor(rewards)
        values = torch.FloatTensor(values)
        dones = torch.FloatTensor(dones)

        returns, advantages = self.compute_gae(rewards, values, next_value, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for epoch in range(epochs):
            print("observations: ", observations)
            new_logits = self.actor(observations)
            print("new_logits: ", new_logits)
            
            new_values = self.critic(observations).squeeze(-1)
            
            new_probs = torch.softmax(new_logits, dim=-1)
            print("new_probs: ", new_probs)
            dist = Categorical(new_probs)
            print("dist: ", dist)
            
            
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy()

            ratio = torch.exp(new_log_probs - old_log_probs)
            surrogate1 = ratio * advantages
            surrogate2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            actor_loss = -torch.min(surrogate1, surrogate2).mean()

            value_loss = nn.MSELoss()(new_values, returns)

            loss = actor_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            loss.backward()

            # Log gradients for actor and critic
            self.log_gradients(self.actor, "Actor", epoch, self.actor_gradient_log_path)
            self.log_gradients(self.critic, "Critic", epoch, self.critic_gradient_log_path)

            self.actor_optimizer.step()
            self.critic_optimizer.step()
    
    def compute_gae(self, rewards: torch.Tensor, values: torch.Tensor, next_value: torch.Tensor, 
                    dones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        # Compute a smoothed advantage estimate for each time step
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = next_value
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_value = values[t + 1]
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
        
        # Compute the returns as the sum of the advantages and the values (to be used as target for the critic)
        returns = advantages + values
        return returns, advantages

    def log_gradients(self, model: nn.Module, model_name: str, update_step: int, log_path: str):
        """
        Log the gradients of the model parameters to a file.

        Args:
            model (nn.Module): The model whose gradients are to be logged.
            model_name (str): The name of the model (e.g., 'Actor' or 'Critic').
            log_path (str): The path to the log file.
        """
        with open(log_path, 'a') as f:
            f.write(f"\n{'='*20} {model_name} Gradients at Update Step {update_step} {'='*20}\n")
            for name, param in model.named_parameters():
                if param.grad is not None:
                    f.write(f"{name}: grad mean={param.grad.mean():.6f}, grad std={param.grad.std():.6f}, grad min={param.grad.min():.6f}, grad max={param.grad.max():.6f}\n")
            f.write(f"\n{'='*60}\n")

def ppo_train(
        game_engine: Game,
        vectorizer: DominionVectorizer,
        run_id: str,
        output_dir: str,
        num_episodes: int,
        batch_size: int,
        update_epochs: int,
        hidden_size: int,
) -> Tuple[PPOAgent, PPOAgent]:
    
    obs_dim = vectorizer.vectorize_observation(game_engine).shape[0]
    action_dim = vectorizer.action_space_size
    player1 = PPOAgent(obs_dim, action_dim, hidden_size, output_dir=output_dir)
    player2 = PPOAgent(obs_dim, action_dim, hidden_size, output_dir=output_dir)
    
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
        cumulative_rewards = [0, 0]

        while not done:
            current_player = game_engine.current_player_turn
            agent = player1 if current_player == 0 else player2
            
            obs = vectorizer.vectorize_observation(game_engine)
            
            action, log_prob, probs = agent.get_action(obs, game_engine, vectorizer)
            value = agent.get_value(obs)

            action.apply()

            done = game_engine.game_over
            reward = agent.calculate_reward(game_engine, current_player, done)

            game_engine_observation_state_copy = copy.deepcopy(game_engine.get_observation_state())

            # Log the game state after the action and reward calculation
            game_history.append({
                'episode': episode + 1,
                'game_over': game_engine_observation_state_copy['game_state']['game_over'],
                'reward': reward,
                'cumulative_reward': cumulative_rewards[current_player],
                'current_turns': game_engine_observation_state_copy['game_state']['turn_number'],
                'after_doing_action': str(action),
                'action_probs': convert_action_probs_to_readable(probs, vectorizer, game_engine),
                'current_player_name': game_engine_observation_state_copy['game_state']['current_player_name'],
                'current_phase': game_engine_observation_state_copy['game_state']['current_phase'],
                'current_player_state': game_engine_observation_state_copy['current_player_state'],
                'opponent_state': game_engine_observation_state_copy['opponent_state'],
                'supply_piles': game_engine_observation_state_copy['game_state']['supply_piles'],
            })
            
            observations[current_player].append(obs)
            actions[current_player].append(vectorizer.vectorize_action(action))
            log_probs[current_player].append(log_prob)
            rewards[current_player].append(reward)
            values[current_player].append(value)
            dones[current_player].append(done)
            
            episode_rewards[current_player] += reward
            cumulative_rewards[current_player] += reward
            
            # Update the agent   
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
        
        # Update the reward for the last action if the game is over
        game_history[-1]['reward'] = reward
        # Update data for both players
        for p in [0, 1]:
            final_obs = vectorizer.vectorize_observation(game_engine)
            final_value = player1.get_value(final_obs) if p == 0 else player2.get_value(final_obs)
            final_reward = player1.calculate_reward(game_engine, p, done) if p == 0 else player2.calculate_reward(game_engine, p, done)
            
            observations[p].append(final_obs)
            values[p].append(final_value)
            rewards[p].append(final_reward)
            dones[p].append(True)
            
            # We don't append to actions and log_probs as no action was taken
            
            episode_rewards[p] += final_reward
            cumulative_rewards[p] += final_reward

        print(f"Episode {episode + 1}, Rewards: Player 1 = {episode_rewards[0]}, Player 2 = {episode_rewards[1]}")

        # Save game history to CSV
        with open(os.path.join(output_dir, f"game_history_{episode+1}.csv"), "w", newline='') as f:
            fieldnames = list(game_history[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(game_history)

        # Reset game history for the next episode
        game_history = []

    return player1, player2