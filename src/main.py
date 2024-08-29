import numpy as np
import os
from babyDOM.ppo import ppo_train
from game_engine.game import Game
from vectorization.vectorizer import DominionVectorizer
from utils import save_params_to_csv

def __main__():
    # Initialize the game engine
    game_engine = Game()

    # Initialize the vectorizer
    vectorizer = DominionVectorizer(game_engine.all_card_names)

    # Generate a unique run ID
    run_id = "0829_run1"

    # Set up training parameters
    num_episodes = 100
    batch_size = 32
    update_epochs = 10
    hidden_size = 64

    # Create output directory
    output_dir = os.path.join(os.path.expanduser('~/Documents/'), 'dominion_output', run_id)
    os.makedirs(output_dir, exist_ok=True)

    trained_agent1, trained_agent2 = ppo_train(
        game_engine=game_engine,
        vectorizer=vectorizer,
        run_id=run_id,
        output_dir=output_dir,
        num_episodes=num_episodes,
        batch_size=batch_size,
        update_epochs=update_epochs,
        hidden_size=hidden_size
    )

    print(f"Training completed for run {run_id}")
    print(f"Output files saved in: {output_dir}")

    # Save training parameters as CSV
    params = {
        "num_episodes": num_episodes,
        "batch_size": batch_size,
        "update_epochs": update_epochs,
        "hidden_size": hidden_size
    }

    params_file = save_params_to_csv(params, output_dir, "training_params.csv")
    print(f"Training parameters saved to {params_file}")

if __name__ == "__main__":
    __main__()
