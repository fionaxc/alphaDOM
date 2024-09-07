import torch
from typing import Tuple
from game_engine.game import Game
from vectorization.vectorizer import DominionVectorizer
from .ppo import PPOAgent
from .utils import convert_action_probs_to_readable
import os
import csv
import copy
from tqdm import tqdm
import logging

def play_game(game_engine: Game, vectorizer: DominionVectorizer, agent: PPOAgent, game_id: int):
    # Initialize new buffer to store experiences of one rollout
    buffer = {
        'observations': [[], []],  # For each player
        'actions': [[], []],
        'log_probs': [[], []],
        'rewards': [[], []],
        'values': [[], []],
        'dones': [[], []]
    }

    game_engine = Game()  # Reset the game state
    done = False

    # Initialize game history
    game_history = []

    # Initialize variables for tracking game state
    cumulative_rewards_after_action = [0, 0]

    while not done:
        current_player = game_engine.current_player_turn

        obs = vectorizer.vectorize_observation(game_engine)

        action, log_prob, probs = agent.get_action(obs, game_engine, vectorizer)
        value = agent.get_value(obs)

        game_engine_observation_state_copy = copy.deepcopy(game_engine.get_observation_state())

        # Log the game state before we take an action
        game_history.append({
            'game': game_id + 1,
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
    if winner is not None:
        winner_index = 0 if winner.name == game_engine.players[0].name else 1
        for i in range(len(game_history) - 1, -1, -1):
            if game_history[i]['current_player_name'] == winner.name:
                game_history[i]['cumulative_rewards_after_action'] += 1 - game_history[i]['reward']
                game_history[i]['reward'] = 1
                buffer['rewards'][winner_index][-1] = 1
                break
    else:
        for i in [0, 1]:
            buffer['rewards'][i][-1] = 0.2
            for j in range(len(game_history) - 1, -1, -1):
                if game_history[j]['current_player_name'] == game_engine.players[i].name:
                    game_history[j]['cumulative_rewards_after_action'] += 0.2 - game_history[j]['reward']
                    game_history[j]['reward'] = 0.2
                    break

    return buffer, game_history

def ppo_train(
        game_engine: Game,
        vectorizer: DominionVectorizer,
        run_id: str,
        output_dir: str,
        num_games: int,
        batch_size: int,
        update_epochs: int,
        hidden_size: int,
) -> PPOAgent:
    obs_dim = vectorizer.vectorize_observation(game_engine).shape[0]
    action_dim = vectorizer.action_space_size
    agent = PPOAgent(obs_dim, action_dim, hidden_size, output_dir=output_dir)
    
    # Create a directory for storing output files
    os.makedirs(output_dir, exist_ok=True)

    # Configure logging to write to a file in the output directory
    log_file_path = os.path.join(output_dir, 'batch_sizes.txt')
    logging.basicConfig(
        level=logging.INFO,  # Set the logging level to INFO
        format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
        handlers=[
            logging.FileHandler(log_file_path)  # Save to a file in the output directory
        ]
    )

    # Log initial critical information
    agent.log_critical_info("Base Agent", 0, torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]))

    # Initialize buffer for batch updates
    batch_buffer = {
        'observations': [[], []],
        'actions': [[], []],
        'log_probs': [[], []],
        'rewards': [[], []],
        'values': [[], []],
        'dones': [[], []]
    }

    for game in tqdm(range(num_games)):
        # Play a game to get one rollout
        buffer, game_history = play_game(game_engine, vectorizer, agent, game)
        
         # Append current game's buffer to batch buffer
        for key in buffer:
            batch_buffer[key][0].extend(buffer[key][0])
            batch_buffer[key][1].extend(buffer[key][1])

        # Update policy every batch_size games
        if (game + 1) % batch_size == 0:
             # Log the sizes of the arrays in the batch buffer before update
            logging.info(f"Batch buffer sizes before update at game {game + 1}: " +
                         f"observations: {len(batch_buffer['observations'][0]) + len(batch_buffer['observations'][1])}, " +
                         f"actions: {len(batch_buffer['actions'][0]) + len(batch_buffer['actions'][1])}, " +
                         f"log_probs: {len(batch_buffer['log_probs'][0]) + len(batch_buffer['log_probs'][1])}, " +
                         f"rewards: {len(batch_buffer['rewards'][0]) + len(batch_buffer['rewards'][1])}, " +
                         f"values: {len(batch_buffer['values'][0]) + len(batch_buffer['values'][1])}, " +
                         f"dones: {len(batch_buffer['dones'][0]) + len(batch_buffer['dones'][1])}")

            combined_buffer = {key: batch_buffer[key][0] + batch_buffer[key][1] for key in batch_buffer}
            agent.update(
                "Base Agent",
                combined_buffer['observations'],
                combined_buffer['actions'],
                combined_buffer['log_probs'],
                combined_buffer['rewards'],
                combined_buffer['values'],
                combined_buffer['dones'],
                next_value=0,  # Terminal state
                epochs=update_epochs,
                vectorizer=vectorizer,
                game=game
            )
            # Clear batch buffer after update
            batch_buffer = {key: [[], []] for key in batch_buffer}

        # Save game history to CSV
        with open(os.path.join(output_dir, f"game_history_{game+1}.csv"), "w", newline='') as f:
            fieldnames = list(game_history[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(game_history)

        # Reset game history for the next game
        game_history = []

    return agent
