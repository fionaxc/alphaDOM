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
from multiprocessing import Pool
import time
from vectorized_env import VectorizedDominionEnv
import numpy as np

def save_game_history(output_dir: str, game_id: int, game_history: list):
    """Save game history to a CSV file."""
    file_path = os.path.join(output_dir, f"game_history_{game_id}.csv")
    print(f"Saving game history to: {file_path}")
    with open(os.path.join(output_dir, f"game_history_{game_id}.csv"), "w", newline='') as f:
        fieldnames = list(game_history[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(game_history)

# def play_game(game_engine: Game, vectorizer: DominionVectorizer, agent: PPOAgent, game_id: int):
#     # start_game_time = time.time()  # Start timing the action selection
#     # Initialize new buffer to store experiences of one rollout
#     buffer = {
#         'observations': [[], []],  # For each player
#         'actions': [[], []],
#         'log_probs': [[], []],
#         'rewards': [[], []],
#         'values': [[], []],
#         'dones': [[], []],
#         'action_masks': [[], []]
#     }

#     game_engine = Game()  # Reset the game state
#     done = False

#     # Initialize game history
#     game_history = []

#     # Initialize variables for tracking game state
#     cumulative_rewards_after_action = [0, 0]

#     # start_play_time = time.time()  # Start timing the action selection
#     while not done:
#         current_player = game_engine.current_player_turn

#         obs = vectorizer.vectorize_observation(game_engine)

#         # start_time = time.time()  # Start timing the action selection
#         action, log_prob, probs, action_mask = agent.get_action(obs, game_engine, vectorizer)
#         # action_selection_time = time.time() - start_time  # Calculate the time taken for action selection
#         # print(f"Action selection time: {action_selection_time:.6f} seconds")  # Log the action selection time

#         # start_time = time.time()  # Start timing the value estimation
#         value = agent.get_value(obs)
#         # value_estimation_time = time.time() - start_time  # Calculate the time taken for value estimation
#         # print(f"Value estimation time: {value_estimation_time:.6f} seconds")  # Log the value estimation time

#         # start_time = time.time()  # Start timing the game state copy
#         game_engine_observation_state_copy = copy.deepcopy(game_engine.get_observation_state())
#         # game_state_copy_time = time.time() - stiart_time  # Calculate the time taken for game state copy
#         # print(f"Game state copy time: {game_state_copy_time:.6f} seconds")  # Log the game state copy time

#         # Log the game state before we take an action
#         game_history.append({
#             'game': game_id,
#             'game_over': game_engine_observation_state_copy['game_state']['game_over'],
#             'reward': 0,
#             #'cumulative_rewards_after_action': 0,
#             'current_turns': game_engine_observation_state_copy['game_state']['turn_number'],
#             'from_state_chose_action': str(action),
#             'action_probs': convert_action_probs_to_readable(probs, vectorizer, game_engine),
#             #'vectorized_observation': obs,
#             'current_player_name': game_engine_observation_state_copy['game_state']['current_player_name'],
#             'current_phase': game_engine_observation_state_copy['game_state']['current_phase'],
#             'current_player_state': game_engine_observation_state_copy['current_player_state'],
#             'opponent_state': game_engine_observation_state_copy['opponent_state'],
#             'supply_piles': game_engine_observation_state_copy['game_state']['supply_piles']
#         })

#         # start_time = time.time()  # Start timing the action application
#         action.apply()
#         # action_application_time = time.time() - start_time  # Calculate the time taken for action application
#         # print(f"Action application time: {action_application_time:.6f} seconds")  # Log the action application time

#         done = game_engine.game_over
#         reward = agent.calculate_reward(game_engine, current_player, done, action)
#         cumulative_rewards_after_action[current_player] += reward

#         # Now that we've taken an action, update the rewards and cumulative rewards
#         game_history[-1]['reward'] = reward
#         game_history[-1]['cumulative_rewards_after_action'] = cumulative_rewards_after_action[current_player]

#         buffer['observations'][current_player].append(obs)
#         buffer['actions'][current_player].append(vectorizer.vectorize_action(action))
#         buffer['log_probs'][current_player].append(log_prob)
#         buffer['rewards'][current_player].append(reward)
#         buffer['values'][current_player].append(value)
#         buffer['dones'][current_player].append(done)
#         buffer['action_masks'][current_player].append(action_mask)

#     # end_play_time = time.time()  # End timing the action selection
#     # play_duration = end_play_time - start_play_time  # Calculate the total game duration
#     # print(f"Total play duration: {play_duration:.2f} seconds")  # Log the total game duration

#     # Since the game is over, update the reward for the game_history and the buffer reward of the winner's + loser's last action that was appended
#     # NOTE: I'm not sure if this is a problem because if the other player was dumb and just bought the last card, then assigning 
#     # the reward to the other playe's last action doesn't feel right
#     winner = game_engine.winner()
#     winning_reward = 1
#     tie_reward = 0.2
#     losing_reward = -1
    
#     def update_rewards(player_name, reward):
#         for entry in reversed(game_history):
#             if entry['current_player_name'] == player_name:
#                 entry['cumulative_rewards_after_action'] += reward - entry['reward']
#                 entry['reward'] = reward
#                 break

#     if winner is not None:
#         winner_index = 0 if winner.name == game_engine.players[0].name else 1
#         loser_index = 1 - winner_index
#         update_rewards(winner.name, winning_reward)
#         update_rewards(game_engine.players[loser_index].name, losing_reward)
#         buffer['rewards'][winner_index][-1] = winning_reward
#         buffer['rewards'][loser_index][-1] = losing_reward
#     else:
#         for i in [0, 1]:
#             update_rewards(game_engine.players[i].name, tie_reward)
#             buffer['rewards'][i][-1] = tie_reward
    
#     # end_game_time = time.time()  # End timing the game
#     # game_duration = end_game_time - start_game_time  # Calculate the total game duration
#     # print(f"Total game duration: {game_duration:.2f} seconds")  # Log the total game duration

#     return game_id, buffer, game_history


def play_game(game_engine: Game, vectorizer: DominionVectorizer, agent: PPOAgent, game_id: int):
    buffer = {
        'observations': [np.empty((0, vectorizer.observation_space_size), dtype=np.float32) for _ in range(2)],
        'actions': [np.empty((0,), dtype=np.int32) for _ in range(2)],
        'log_probs': [np.empty((0,), dtype=np.float32) for _ in range(2)],
        'rewards': [np.empty((0,), dtype=np.float32) for _ in range(2)],
        'values': [np.empty((0,), dtype=np.float32) for _ in range(2)],
        'dones': [np.empty((0,), dtype=bool) for _ in range(2)],
        'action_masks': [np.empty((0, vectorizer.action_space_size), dtype=bool) for _ in range(2)]
    }

    game_engine = Game()  # Reset the game state
    done = False
    game_history = []
    cumulative_rewards_after_action = np.zeros(2, dtype=np.float32)

    while not done:
        current_player = game_engine.current_player_turn
        obs = vectorizer.vectorize_observation(game_engine)
        action, log_prob, probs, action_mask = agent.get_action(obs, game_engine, vectorizer)
        value = agent.get_value(obs)

        game_engine_observation_state_copy = copy.deepcopy(game_engine.get_observation_state())

        game_history.append({
            'game': game_id,
            'game_over': game_engine_observation_state_copy['game_state']['game_over'],
            'reward': 0,
            'current_turns': game_engine_observation_state_copy['game_state']['turn_number'],
            'from_state_chose_action': str(action),
            'action_probs': convert_action_probs_to_readable(probs, vectorizer, game_engine),
            'current_player_name': game_engine_observation_state_copy['game_state']['current_player_name'],
            'current_phase': game_engine_observation_state_copy['game_state']['current_phase'],
            'current_player_state': game_engine_observation_state_copy['current_player_state'],
            'opponent_state': game_engine_observation_state_copy['opponent_state'],
            'supply_piles': game_engine_observation_state_copy['game_state']['supply_piles']
        })

        action.apply()
        done = game_engine.game_over
        reward = agent.calculate_reward(game_engine, current_player, done, action)
        cumulative_rewards_after_action[current_player] += reward

        game_history[-1]['reward'] = reward
        game_history[-1]['cumulative_rewards_after_action'] = cumulative_rewards_after_action[current_player]

        buffer['observations'][current_player] = np.vstack((buffer['observations'][current_player], obs))
        buffer['actions'][current_player] = np.append(buffer['actions'][current_player], vectorizer.vectorize_action(action))
        buffer['log_probs'][current_player] = np.append(buffer['log_probs'][current_player], log_prob)
        buffer['rewards'][current_player] = np.append(buffer['rewards'][current_player], reward)
        buffer['values'][current_player] = np.append(buffer['values'][current_player], value)
        buffer['dones'][current_player] = np.append(buffer['dones'][current_player], done)
        buffer['action_masks'][current_player] = np.vstack((buffer['action_masks'][current_player], action_mask))

    winner = game_engine.winner()
    if winner is not None:
        winner_index = 0 if winner.name == game_engine.players[0].name else 1
        loser_index = 1 - winner_index
        buffer['rewards'][winner_index][-1] = 1  # winning_reward
        buffer['rewards'][loser_index][-1] = -1  # losing_reward
    else:
        buffer['rewards'][0][-1] = buffer['rewards'][1][-1] = 0.2  # tie_reward

    return game_id, buffer, game_history

def run_all_games_in_parallel(game_engine: Game, vectorizer: DominionVectorizer, agent: PPOAgent, num_games: int, batch_size: int, update_epochs: int, output_dir: str):
    # Initialize buffer for batch updates
    batch_buffer = {
        'observations': [[], []],
        'actions': [[], []],
        'log_probs': [[], []],
        'rewards': [[], []],
        'values': [[], []],
        'dones': [[], []],
        'action_masks': [[], []]
    }

    with Pool() as pool:
        with tqdm(total=num_games) as pbar:
            game_counter = 1  # Initialize game counter
            num_batch_updates = 0  # Initialize batch update counter
            while game_counter < num_games:
                start_time = time.time()

                # Submit batch_size games to be played in parallel
                results = pool.starmap(play_game, [(game_engine, vectorizer, agent, game_counter + i) for i in range(batch_size)])
                
                # Collect results and update progress bar
                for result in results:
                    pbar.update(1)
                    game_counter += 1

                # Sort results by game_id to ensure correct order
                results.sort(key=lambda x: x[0])

                # Save game histories in parallel
                save_futures = [pool.apply_async(save_game_history, (output_dir, game_id, game_history)) for game_id, _, game_history in results if game_id % 500 == 0]
                for future in save_futures:
                    future.get()  # Ensure all save operations complete

                for game_id, buffer, game_history in results:
                    # Append current game's buffer to batch buffer
                    for key in buffer:
                        batch_buffer[key][0].extend(buffer[key][0])
                        batch_buffer[key][1].extend(buffer[key][1])

                # Log the sizes of the arrays in the batch buffer before update
                logging.info(f"Batch buffer sizes before update: " +
                             f"observations: {len(batch_buffer['observations'][0]) + len(batch_buffer['observations'][1])}, " +
                             f"actions: {len(batch_buffer['actions'][0]) + len(batch_buffer['actions'][1])}, " +
                             f"log_probs: {len(batch_buffer['log_probs'][0]) + len(batch_buffer['log_probs'][1])}, " +
                             f"rewards: {len(batch_buffer['rewards'][0]) + len(batch_buffer['rewards'][1])}, " +
                             f"values: {len(batch_buffer['values'][0]) + len(batch_buffer['values'][1])}, " +
                             f"dones: {len(batch_buffer['dones'][0]) + len(batch_buffer['dones'][1])}")
                
                # End timing the game-playing process
                end_game_time = time.time()
                game_duration = end_game_time - start_time
                logging.info(f"Game playing duration: {game_duration:.2f} seconds")

                # Start timing the update process
                update_start_time = time.time()

                # Combine buffers from all games in the batch
                combined_buffer = {key: batch_buffer[key][0] + batch_buffer[key][1] for key in batch_buffer}
                
                # Perform the update with the combined buffer
                agent.update(
                    "Base Agent",
                    combined_buffer['observations'],
                    combined_buffer['actions'],
                    combined_buffer['log_probs'],
                    combined_buffer['rewards'],
                    combined_buffer['values'],
                    combined_buffer['dones'],
                    combined_buffer['action_masks'],
                    next_value=0,  # Terminal state
                    epochs=update_epochs,
                    vectorizer=vectorizer,
                    batch_size=batch_size,
                    game=game_counter - 1 # -1 because we increment game_counter before using it
                )
                num_batch_updates += 1

                # End timing the update process
                update_end_time = time.time()
                update_duration = update_end_time - update_start_time
                logging.info(f"Update duration: {update_duration:.2f} seconds")

                # Save the model every 50 batch updates (50 * batch_size games)
                if num_batch_updates % 50 == 0:
                    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
                    if not os.path.exists(checkpoint_dir):
                        os.makedirs(checkpoint_dir, exist_ok=True)
                    agent_save_path = os.path.join(checkpoint_dir, f"checkpoint_game_{game_counter-1}.pth")
                    model_dict = {
                        'actor': agent.actor.state_dict(),
                        'critic': agent.critic.state_dict(),
                        'optimizer': agent.actor_optimizer.state_dict(),
                        'critic_optimizer': agent.critic_optimizer.state_dict(),
                        'lr': agent.lr,
                        'gamma': agent.gamma,
                        'epsilon': agent.epsilon,
                        'value_coef': agent.value_coef,
                        'entropy_coef': agent.entropy_coef,
                        'gae_lambda': agent.gae_lambda
                    }
                    torch.save(model_dict, agent_save_path)
                
                # Clear batch buffer after update
                batch_buffer = {key: [[], []] for key in batch_buffer}

def run_all_games_sequentially(game_engine: Game, vectorizer: DominionVectorizer, agent: PPOAgent, num_games: int, batch_size: int, update_epochs: int, output_dir: str):
    # Initialize buffer for batch updates
    batch_buffer = {
        'observations': [[], []],
        'actions': [[], []],
        'log_probs': [[], []],
        'rewards': [[], []],
        'values': [[], []],
        'dones': [[], []],
        'action_masks': [[], []]
    }

    for game in tqdm(range(num_games)):
        # Start timing the entire process for the batch
        if game % batch_size == 0:
            batch_start_time = time.time()

        # Play a game to get one rollout
        game_id, buffer, game_history = play_game(game_engine, vectorizer, agent, game)
        
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
                combined_buffer['action_masks'],
                next_value=0,  # Terminal state
                epochs=update_epochs,
                vectorizer=vectorizer,
                batch_size=batch_size,
                game=game
            )
            # Clear batch buffer after update
            batch_buffer = {key: [[], []] for key in batch_buffer}

        if game % 100 == 0:
            save_game_history(output_dir, game + 1, game_history)

        # End timing the entire process for the batch
        if (game + 1) % batch_size == 0:
            batch_end_time = time.time()
            batch_duration = batch_end_time - batch_start_time
            logging.info(f"Batch update duration: {batch_duration:.2f} seconds")

        # Reset game history for the next game
        game_history = []

def ppo_train(
        game_engine: Game,
        vectorizer: DominionVectorizer,
        run_id: str,
        output_dir: str,
        num_games: int,
        batch_size: int,
        update_epochs: int,
        hidden_size: int,
        checkpoint_path: str = None
) -> PPOAgent:
    obs_dim = vectorizer.vectorize_observation(game_engine).shape[0]
    action_dim = vectorizer.action_space_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = PPOAgent(obs_dim, action_dim, hidden_size, device=device, output_dir=output_dir, checkpoint_path=checkpoint_path)
    agent.to(device)

    # Create a directory for storing output files
    os.makedirs(output_dir, exist_ok=True)

    # Configure logging to write to a file in the output directory
    log_file_path = os.path.join(output_dir, '_batch_sizes.txt')
    logging.basicConfig(
        level=logging.INFO,  # Set the logging level to INFO
        format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
        handlers=[
            logging.FileHandler(log_file_path)  # Save to a file in the output directory
        ]
    )

    # Log initial critical information
    agent.log_critical_info("Base Agent", 0, torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]))

    # run_all_games_sequentially(game_engine, vectorizer, agent, num_games, batch_size, update_epochs, output_dir)
    run_all_games_in_parallel(game_engine, vectorizer, agent, num_games, batch_size, update_epochs, output_dir)

    return agent

# def ppo_train(
#         env: VectorizedDominionEnv,
#         agent: PPOAgent,
#         vectorizer: DominionVectorizer,
#         run_id: str,
#         output_dir: str,
#         num_games: int,
#         batch_size: int,
#         update_epochs: int
# ) -> PPOAgent:
#     """
#     Train the agent using the vectorized environment.
#     """
#     device = agent.device
#     total_steps = 0
#     num_updates = num_games // batch_size

#     for update in range(num_updates):
#         observations = env.reset()
        
#         for _ in range(batch_size):
#             action_masks = env.get_action_masks()
#             actions, log_probs, _, _ = agent.get_action(observations, action_masks)
#             values = agent.get_value(observations)
            
#             next_observations, rewards, dones = env.step(actions)
            
#             agent.store_transition(observations, actions, log_probs, rewards, values, dones, action_masks)
            
#             observations = next_observations
#             total_steps += env.num_envs

#         agent.update(update_epochs)

#         if update % 10 == 0:
#             agent.log_critical_info("Base Agent", total_steps)

#         if update % 50 == 0:
#             checkpoint_dir = os.path.join(output_dir, 'checkpoints')
#             os.makedirs(checkpoint_dir, exist_ok=True)
#             agent_save_path = os.path.join(checkpoint_dir, f"checkpoint_step_{total_steps}.pth")
#             agent.save(agent_save_path)

#     return agent

