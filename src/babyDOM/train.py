import torch
from typing import Tuple
from game_engine.game import Game
from vectorization.vectorizer import DominionVectorizer
from .ppo import PPOAgent
from .utils import convert_action_probs_to_readable
import os
import csv
import copy

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

    # Log initial critical information
    player1.log_critical_info("Player 1", 0, torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]))
    player2.log_critical_info("Player 2", 0, torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]))

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
        cumulative_rewards_after_action = [0, 0]

        while not done:
            current_player = game_engine.current_player_turn
            agent = player1 if current_player == 0 else player2
            
            obs = vectorizer.vectorize_observation(game_engine)
            
            action, log_prob, probs = agent.get_action(obs, game_engine, vectorizer)
            value = agent.get_value(obs)

            game_engine_observation_state_copy = copy.deepcopy(game_engine.get_observation_state())

            # Log the game state before we take an action
            game_history.append({
                'episode': episode + 1,
                'game_over': game_engine_observation_state_copy['game_state']['game_over'],
                'reward': 0,
                'cumulative_rewards_after_action': 0,
                'current_turns': game_engine_observation_state_copy['game_state']['turn_number'],
                'from_state_chose_action': str(action),
                'action_probs': convert_action_probs_to_readable(probs, vectorizer, game_engine),
                'current_player_name': game_engine_observation_state_copy['game_state']['current_player_name'],
                'current_phase': game_engine_observation_state_copy['game_state']['current_phase'],
                'current_player_state': game_engine_observation_state_copy['current_player_state'],
                'opponent_state': game_engine_observation_state_copy['opponent_state'],
                'supply_piles': game_engine_observation_state_copy['game_state']['supply_piles'],
            })

            action.apply()
    
            done = game_engine.game_over
            reward = agent.calculate_reward(game_engine, current_player, done, action)
            cumulative_rewards_after_action[current_player] += reward

            # Now that we've taken an action, update the rewards and cumulative rewards
            game_history[-1]['reward'] = reward
            game_history[-1]['cumulative_rewards_after_action'] = cumulative_rewards_after_action[current_player]
            
            buffer['observations'][current_player].append(obs)
            buffer['actions'][current_player].append(vectorizer.vectorize_action(action))
            buffer['log_probs'][current_player].append(log_prob)
            buffer['rewards'][current_player].append(reward)
            buffer['values'][current_player].append(value)
            buffer['dones'][current_player].append(done)
        
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
                    game_history[i]['cumulative_rewards_after_action'] += 1 - game_history[i]['reward']
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
                        game_history[j]['cumulative_rewards_after_action'] += 0.2 - game_history[j]['reward']
                        game_history[j]['reward'] = 0.2
                        break

        # Update policies for both players
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