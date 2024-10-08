import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import List, Tuple
from game_engine.game import Game
from vectorization.vectorizer import DominionVectorizer
from game_engine.action import Action, ActionType
import os
from .utils import stable_softmax
import torch.nn.functional as F

class PPOActor(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, obs):
        # Return raw logits without applying softmax
        return self.actor(obs)

class PPOCritic(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
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
                 device: str,
                 lr: float = 1e-4,
                 gamma: float = 0.9999,
                 epsilon: float = 0.15,
                 value_coef: float = 0.43,
                 entropy_coef: float = 0.05,
                 gae_lambda: float = 0.75,
                 output_dir: str = "src/output",
                 checkpoint_path: str = None):
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
            output_dir (str): Directory to save training logs and checkpoints.
            checkpoint_path (str): Path to a checkpoint file to load model weights from.
        """
        self.actor = PPOActor(obs_dim, action_dim, hidden_size)
        self.critic = PPOCritic(obs_dim, hidden_size)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.gae_lambda = gae_lambda
        self.output_dir = output_dir
        self.device = device
        # Load checkpoint if provided
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
    
    def to(self, device):
        """
        Move the model to the specified device.
        """
        self.actor = self.actor.to(device)
        self.critic = self.critic.to(device)
        self.device = device
        return self
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model weights from a checkpoint file.

        Args:
            checkpoint_path (str): Path to the checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        # self.lr = checkpoint['lr']
        # self.gamma = checkpoint['gamma']
        # self.epsilon = checkpoint['epsilon']
        # self.value_coef = checkpoint['value_coef']
        # self.entropy_coef = checkpoint['entropy_coef']
        # self.gae_lambda = checkpoint['gae_lambda']
        print(f"Loaded checkpoint from {checkpoint_path}")
        # print(f"{checkpoint}")

    def get_action(self, obs: np.ndarray, game: Game, vectorizer: DominionVectorizer) -> Tuple[Action, float, torch.Tensor, torch.Tensor]:
        """
        Get the action from the agent's policy. Returns the selected action, the log probability of the action (used for updating the policy), and the action mask.
        """
        obs = torch.FloatTensor(obs).to(self.device)
        logits = self.actor(obs)

        # Get the action mask directly from the vectorizer
        action_mask = vectorizer.get_action_mask(game)
        action_mask = torch.BoolTensor(action_mask).to(self.device)

        # Apply mask and create categorical distribution
        masked_logits = logits.masked_fill(~action_mask, -float('inf'))
        probs = torch.softmax(masked_logits, dim=-1)
        m = Categorical(probs)

        action_index = m.sample()
        selected_action = vectorizer.devectorize_action(action_index.item(), game.current_player())
        return selected_action, m.log_prob(action_index).item(), probs.detach(), action_mask

    def get_value(self, obs: np.ndarray) -> float:
        """
        Get the value from the critic.
        """
        obs = torch.FloatTensor(obs).to(self.device)
        return self.critic(obs).item()
    
    def calculate_reward(self, game_engine: Game, current_player: int, done: bool, action: Action) -> int:
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
        # Losing reward
        elif done and game_engine.winner() is not None and game_engine.winner().name != game_engine.players[current_player].name:
            return -1
        # Continuing reward
        else:
            return 0

    def update(self, player_name: str, observations: List[np.ndarray], actions: List[int], old_log_probs: List[float], 
               rewards: List[float], values: List[float], dones: List[bool], action_masks: List[np.ndarray], next_value: float,
               epochs: int, vectorizer: DominionVectorizer, batch_size: int,game: int = 0):
        observations = torch.FloatTensor(np.array(observations)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        values = torch.FloatTensor(values).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        next_value = torch.FloatTensor([next_value]).to(self.device)
        action_masks = torch.stack(action_masks).to(self.device)

        #### GAE return implementation ######
        returns, advantages = self.compute_gae(rewards, values.detach(), next_value.detach(), dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ### RTG return implementation ####
        # rtgs = self.compute_rtg(rewards)
        # returns = rtgs
        # advantages = rtgs - values
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for epoch in range(epochs):
            new_logits = self.actor(observations)
            new_masked_logits = new_logits.masked_fill(~action_masks, -float('inf'))
            new_masked_logits.retain_grad()
            new_values = self.critic(observations).squeeze(-1)
            # regular softmax
            new_probs = torch.softmax(new_masked_logits, dim=-1)
            #new_probs = stable_softmax(new_masked_logits)
            dist = Categorical(new_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy()    

            ratio = torch.exp(new_log_probs - old_log_probs.detach())
            surrogate1 = ratio * advantages
            surrogate2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            # actor_loss = -torch.min(surrogate1, surrogate2).mean() - self.entropy_coef * entropy.mean()
            actor_loss = -torch.min(surrogate1, surrogate2).mean()

            value_loss = nn.MSELoss()(new_values, returns.detach())

            loss = actor_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()

            # Separate Loss Implementation ##
            # # Actor Loss Step
            # self.actor_optimizer.zero_grad()
            # actor_loss.backward(retain_graph=True)
            # self.actor_optimizer.step()
            # # Critic Loss Step
            # self.critic_optimizer.zero_grad()
            # value_loss.backward()
            # self.critic_optimizer.step()

            ## Combined Loss Implementation ##
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

            # Log critical information every last gradient step
            if epoch == epochs - 1 and game % (batch_size * 50) == 0:
                # Print out to make sure masking is correct
                # print("==================invalid action masking=============")
                # print("Original logits: ", new_logits)
                # print("Action masks: ", action_masks)
                # print("Masked logits: ", new_masked_logits)
                # print("Probabilities: ", new_probs)
                # print("Log probabilities: ", new_log_probs)
                # print("Entropy: ", entropy)
                # print("Ratio: ", ratio)
                # print("Surrogate1: ", surrogate1)
                # print("Surrogate2: ", surrogate2)
                # print("Actor loss: ", actor_loss)
                # print("Value loss: ", value_loss)
                # print("Advantages: ", advantages)
                # print("Returns: ", returns)
                # print("Gradients: ", new_masked_logits.grad)

                self.log_critical_info(player_name, game, actor_loss, value_loss, surrogate1, surrogate2, ratio, entropy, torch.tensor([0.0]), advantages, returns)

    def compute_rtg(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute the return-to-go for the rewards using the discount factor gamma
        """
        rtgs = torch.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(len(rewards))):
            running_add = running_add * self.gamma + rewards[t]
            rtgs[t] = running_add
        return rtgs
    
    def compute_gae(self, rewards: torch.Tensor, values: torch.Tensor, next_value: torch.Tensor, 
                    dones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the Generalized Advantage Estimation (GAE) for the rewards.
        """
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
    
    def log_critical_info(self, player_name: str, game: int, actor_loss: torch.Tensor, value_loss: torch.Tensor, 
                          surrogate1: torch.Tensor, surrogate2: torch.Tensor, ratio: torch.Tensor, 
                          entropy: torch.Tensor, total_loss: torch.Tensor, advantages: torch.Tensor, returns: torch.Tensor):
        """
        Log critical information about the training process every 10 games
        """
        log_path = os.path.join(self.output_dir, '_training_log.txt')
        # Create the log file if it doesn't exist
        if not os.path.exists(log_path):
            with open(log_path, 'w') as f:
                f.write('')  # Create an empty file

        with open(log_path, 'a') as f:
            f.write(f"\n{'='*20} Player {player_name} - Game {game} {'='*20}\n")
            f.write(f"Actor Loss: {actor_loss.item():.6f}\n")
            f.write(f"Value Loss: {value_loss.item():.6f}\n")
            f.write(f"Surrogate Objective 1: {surrogate1}\n")
            f.write(f"Surrogate Objective 2: {surrogate2}\n")
            f.write(f"Ratio of Probabilities: {ratio.mean().item():.6f}\n")
            f.write(f"Entropy: {entropy.mean().item():.6f}\n")
            f.write(f"Total Loss: {total_loss.item():.6f}\n")
            num_elements = 7  # Change this variable to adjust the number of elements printed
            truncated_advantages = [round(a.item(), 6) for a in advantages[:num_elements]] + ["..."] + [round(a.item(), 6) for a in advantages[-num_elements:]]
            f.write(f"Advantages: {truncated_advantages}\n")
            f.write(f"Returns: {returns.mean().item():.6f} (mean), {returns.std().item():.6f} (std)\n")
            f.write("Actor Network:\n")
            for name, param in self.actor.named_parameters():
                f.write(f"{name}: mean={param.mean().item():.6f}")
                if param.numel() > 1:  # Check if tensor has more than one element
                    f.write(f", std={param.std().item():.6f}\n")
                else:
                    f.write(", std=N/A\n")
            f.write("Critic Network:\n")
            for name, param in self.critic.named_parameters():
                f.write(f"{name}: mean={param.mean().item():.6f}")
                if param.numel() > 1:  # Check if tensor has more than one element
                    f.write(f", std={param.std().item():.6f}\n")
                else:
                    f.write(", std=N/A\n")
            f.write("Actor Gradients:\n")
            for name, param in self.actor.named_parameters():
                if param.grad is not None:
                    f.write(f"{name}: grad mean={param.grad.mean().item():.6f}")
                    if param.grad.numel() > 1:  # Check if tensor has more than one element
                        f.write(f", grad std={param.grad.std().item():.6f}\n")
                    else:
                        f.write(", grad std=N/A\n")
            f.write("Critic Gradients:\n")
            for name, param in self.critic.named_parameters():
                if param.grad is not None:
                    f.write(f"{name}: grad mean={param.grad.mean().item():.6f}")
                    if param.grad.numel() > 1:  # Check if tensor has more than one element
                        f.write(f", grad std={param.grad.std().item():.6f}\n")
                    else:
                        f.write(", grad std=N/A\n")
