import torch
from game_engine.game import Game
from vectorization.vectorizer import DominionVectorizer
from babyDOM.ppo import PPOAgent
from babyDOM.utils import convert_action_probs_to_readable
import os

def load_agent(checkpoint_path: str, game_engine: Game, vectorizer: DominionVectorizer, hidden_size: int) -> PPOAgent:
    obs_dim = vectorizer.vectorize_observation(game_engine).shape[0]
    action_dim = vectorizer.action_space_size
    agent = PPOAgent(obs_dim, action_dim, hidden_size, checkpoint_path=checkpoint_path)
    return agent

def display_game_state(game_engine: Game):
    state = game_engine.get_game_state_string()
    print(state)

def display_action_probs(probs, vectorizer, game_engine):
    readable_probs = convert_action_probs_to_readable(probs, vectorizer, game_engine)
    print(f"Action Probabilities: {readable_probs}")

def get_user_action(valid_actions):
    print("Possible Actions:")
    for i, action in enumerate(valid_actions):
        print(f"{i}: {action}")
    choice = int(input("Choose an action: "))
    return valid_actions[choice]

def play_against_bot(agent: PPOAgent, game_engine: Game, vectorizer: DominionVectorizer):
    done = False
    while not done:
        display_game_state(game_engine)
        current_player = game_engine.current_player_turn

        if current_player == 0:  # Human player
            valid_actions = game_engine.get_valid_actions()
            user_action = get_user_action(valid_actions)
            user_action.apply()
        else:  # Bot player
            obs = vectorizer.vectorize_observation(game_engine)
            action, log_prob, probs, action_mask = agent.get_action(obs, game_engine, vectorizer)
            display_action_probs(probs, vectorizer, game_engine)
            print(f"Bot chose action: {action}")  # Display bot's chosen action
            action.apply()

        done = game_engine.game_over

    winner = game_engine.winner()
    if winner:
        print(f"The winner is: {winner.name}")
    else:
        print("The game is a tie.")

if __name__ == "__main__":
    checkpoint_path = "src/sample_agents/village_smithy_moneylender_festival_laboratory_market_witch/3M_0912_SIMPLE2_run3_games1000000_batchsize32_updateepochs5_hidden256/checkpoint_game_652800.pth"
    # checkpoint_path = "src/output/0912_SIMPLE2_run3_games1000000_batchsize32_updateepochs5_hidden256/checkpoints/checkpoint_game_755200.pth"
    hidden_size = 256  # Adjust based on your model
    game_engine = Game()
    vectorizer = DominionVectorizer(game_engine.all_card_names)
    agent = load_agent(checkpoint_path, game_engine, vectorizer, hidden_size)
    play_against_bot(agent, game_engine, vectorizer)
