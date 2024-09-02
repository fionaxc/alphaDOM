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
from .utils import stable_softmax
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
                 lr: float = 3e-4, 
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
        self.output_dir = output_dir

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

    def update(self, player_name: str, observations: List[np.ndarray], actions: List[int], old_log_probs: List[float], 
               rewards: List[float], values: List[float], dones: List[bool], next_value: float, 
               epochs: int, vectorizer: DominionVectorizer, episode: int = 0):
        observations = torch.FloatTensor(np.array(observations))
        actions = torch.LongTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        rewards = torch.FloatTensor(rewards)
        values = torch.FloatTensor(values)
        dones = torch.FloatTensor(dones)

        returns, advantages = self.compute_gae(rewards, values, next_value, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for epoch in range(epochs):
            new_logits = self.actor(observations)
            # print("new_logits: ", new_logits)
            
            new_values = self.critic(observations).squeeze(-1)
            new_probs = stable_softmax(new_logits)
            dist = Categorical(new_probs)
            
            # print("observations: ", observations)
            # print("new_logits: ", new_logits)
            # print("new_values: ", new_values)
            # print("new_probs: ", new_probs)
            # print("dist: ", dist)
            # print("actions: ", actions)

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
            self.actor_optimizer.step()
            self.critic_optimizer.step()

            # Log critical information every 10 episodes
            if (episode + 1) % 10 == 0 and epoch == epochs - 1:
                self.log_critical_info(player_name, episode + 1, actor_loss, value_loss, surrogate1, surrogate2, ratio, entropy, loss, advantages, returns)
    
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
    
    def log_critical_info(self, player_name: str, episode: int, actor_loss: torch.Tensor, value_loss: torch.Tensor, 
                          surrogate1: torch.Tensor, surrogate2: torch.Tensor, ratio: torch.Tensor, 
                          entropy: torch.Tensor, total_loss: torch.Tensor, advantages: torch.Tensor, returns: torch.Tensor):
        """
        Log critical information about the training process every 10 episodes
        """
        log_path = os.path.join(self.output_dir, 'training_log.txt')
        # Create the log file if it doesn't exist
        if not os.path.exists(log_path):
            with open(log_path, 'w') as f:
                f.write('')  # Create an empty file

        with open(log_path, 'a') as f:
            f.write(f"\n{'='*20} Player {player_name} - Episode {episode} {'='*20}\n")
            f.write(f"Actor Loss: {actor_loss.item():.6f}\n")
            f.write(f"Value Loss: {value_loss.item():.6f}\n")
            f.write(f"Surrogate Objective 1: {surrogate1.mean().item():.6f}\n")
            f.write(f"Surrogate Objective 2: {surrogate2.mean().item():.6f}\n")
            f.write(f"Ratio of Probabilities: {ratio.mean().item():.6f}\n")
            f.write(f"Entropy: {entropy.mean().item():.6f}\n")
            f.write(f"Total Loss: {total_loss.item():.6f}\n")
            num_elements = 7  # Change this variable to adjust the number of elements printed
            f.write(f"Advantages: {advantages[:num_elements].tolist()} ... {advantages[-num_elements:].tolist()}\n")
            f.write(f"Returns: {returns.mean().item():.6f} (mean), {returns.std().item():.6f} (std)\n")
            f.write("Actor Network:\n")
            for name, param in self.actor.named_parameters():
                f.write(f"{name}: mean={param.mean().item():.6f}, std={param.std().item():.6f}\n")
            f.write("Critic Network:\n")
            for name, param in self.critic.named_parameters():
                f.write(f"{name}: mean={param.mean().item():.6f}, std={param.std().item():.6f}\n")
            f.write("Actor Gradients:\n")
            for name, param in self.actor.named_parameters():
                if param.grad is not None:
                    f.write(f"{name}: grad mean={param.grad.mean().item():.6f}, grad std={param.grad.std().item():.6f}\n")
            f.write("Critic Gradients:\n")
            for name, param in self.critic.named_parameters():
                if param.grad is not None:
                    f.write(f"{name}: grad mean={param.grad.mean().item():.6f}, grad std={param.grad.std().item():.6f}\n")
            f.write("\n")

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

    # Initialize buffers
    buffer = {
        'observations': [[], []],
        'actions': [[], []],
        'log_probs': [[], []],
        'rewards': [[], []],
        'values': [[], []],
        'dones': [[], []]
    }

    for episode in range(num_episodes):
        game_engine = Game()  # Reset the game state
        done = False

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
            
            buffer['observations'][current_player].append(obs)
            buffer['actions'][current_player].append(vectorizer.vectorize_action(action))
            buffer['log_probs'][current_player].append(log_prob)
            buffer['rewards'][current_player].append(reward)
            buffer['values'][current_player].append(value)
            buffer['dones'][current_player].append(done)

            #  # Debugging: Print buffer lengths
            # print(f"Buffer lengths for player {current_player}: observations={len(buffer['observations'][current_player])}, actions={len(buffer['actions'][current_player])}, log_probs={len(buffer['log_probs'][current_player])}, rewards={len(buffer['rewards'][current_player])}, values={len(buffer['values'][current_player])}, dones={len(buffer['dones'][current_player])}")
            # # Game history length
            # print(f"Game history length: {len(game_history)}")

            cumulative_rewards[current_player] += reward

        # # End of episode update
        # for current_player in [0, 1]:
        #     agent = player1 if current_player == 0 else player2
        #     if len(observations[current_player]) > 0:
        #         agent.update(
        #             observations[current_player],
        #             actions[current_player],
        #             log_probs[current_player],
        #             rewards[current_player],
        #             values[current_player],
        #             dones[current_player],
        #             next_value=0,
        #             epochs=update_epochs,
        #             vectorizer=vectorizer
        #         )
        
        # Since the game is over, update the reward for the game_history and the buffer reward of the winner's last action that was appended
        # NOTE: I'm not sure if this is a problme because if the other player was dumb and just bought the last card, then assigning 
        # the reward to the other playe's last action doesn't feel right
        winner = game_engine.winner()
        player_1_reward = 0
        player_2_reward = 0
        if winner is not None:
            winner_index = 0 if winner.name == game_engine.players[0].name else 1
            # Update the last action of the winner in the game history and buffer
            for i in range(len(game_history) - 1, -1, -1):
                if game_history[i]['current_player_name'] == winner.name:
                    game_history[i]['reward'] = 1
                    buffer['rewards'][winner_index][-1] = 1
                    player_1_reward, player_2_reward = (1, 0) if winner_index == 0 else (0, 1)
                    break
        else:
            player_1_reward, player_2_reward = (0.2, 0.2)
            # Update the last action of both players in the game history and buffer
            for i in [0, 1]:
                buffer['rewards'][i][-1] = 0.2
                for j in range(len(game_history) - 1, -1, -1):
                    if game_history[j]['current_player_name'] == game_engine.players[i].name:
                        game_history[j]['reward'] = 0.2
                        break

        # Update both players
        for current_player in [0, 1]:
            agent = player1 if current_player == 0 else player2
            agent.update(
                game_engine.players[current_player].name,
                buffer['observations'][current_player],
                buffer['actions'][current_player],
                buffer['log_probs'][current_player],
                buffer['rewards'][current_player],
                buffer['values'][current_player],
                buffer['dones'][current_player],
                next_value=0, # Terminal state
                epochs=update_epochs,
                vectorizer=vectorizer,
                episode=episode
            )
            buffer['observations'][current_player] = []
            buffer['actions'][current_player] = []
            buffer['log_probs'][current_player] = []
            buffer['rewards'][current_player] = []
            buffer['values'][current_player] = []
            buffer['dones'][current_player] = []

        # # Update data for both players
        # for p in [0, 1]:
        #     final_obs = vectorizer.vectorize_observation(game_engine)
        #     final_value = player1.get_value(final_obs) if p == 0 else player2.get_value(final_obs)
        #     final_reward = player1.calculate_reward(game_engine, p, done) if p == 0 else player2.calculate_reward(game_engine, p, done)
        #     
        #     buffer['observations'][p].append(final_obs)
        #     buffer['values'][p].append(final_value)
        #     buffer['rewards'][p].append(final_reward)
        #     buffer['dones'][p].append(True)
        #     
        #     # Ensure the final reward is included correctly
        #     if len(buffer['rewards'][p]) < len(buffer['observations'][p]):
        #         buffer['rewards'][p].append(final_reward)
        #         episode_rewards[p] += final_reward
        #     else:
        #         buffer['rewards'][p][-1] = final_reward  # Correct the last reward if already appended
            
        print(f"Episode {episode + 1}, Rewards: Player 1 = {player_1_reward}, Player 2 = {player_2_reward}")

        # Save game history to CSV
        with open(os.path.join(output_dir, f"game_history_{episode+1}.csv"), "w", newline='') as f:
            fieldnames = list(game_history[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(game_history)

        # Reset game history for the next episode
        game_history = []

    return player1, player2