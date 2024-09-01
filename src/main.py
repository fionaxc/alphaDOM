import os
import argparse
from babyDOM.ppo import ppo_train
from game_engine.game import Game
from vectorization.vectorizer import DominionVectorizer
from utils import save_params_to_csv
import torch

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a Dominion AI agent using PPO")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to train")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--update_epochs", type=int, default=10, help="Number of epochs for each update")
    parser.add_argument("--hidden_size", type=int, default=64, help="Hidden size of the neural network")
    parser.add_argument("--run_id", type=str, default="default_run", help="Unique identifier for this run")
    return parser.parse_args()

def __main__():
    args = parse_arguments()
    args.run_id = "0831_run1"

    # Initialize the game engine
    game_engine = Game()

    # Initialize the vectorizer
    vectorizer = DominionVectorizer(game_engine.all_card_names)

    # Create output directory within the alphaDom repository
    output_dir = os.path.join(os.path.dirname(__file__), 'output', args.run_id)
    os.makedirs(output_dir, exist_ok=True)

    trained_agent1, trained_agent2 = ppo_train(
        game_engine=game_engine,
        vectorizer=vectorizer,
        run_id=args.run_id,
        output_dir=output_dir,
        num_episodes=args.num_episodes,
        batch_size=args.batch_size,
        update_epochs=args.update_epochs,
        hidden_size=args.hidden_size,
    )

    print(f"Training completed for run {args.run_id}")
    print(f"Output files saved in: {output_dir}")

    # Save training parameters as CSV
    params = vars(args)
    params_file = save_params_to_csv(params, output_dir, "training_params.csv")
    print(f"Training parameters saved to {params_file}")

    # Save the trained agents
    torch.save(trained_agent1, os.path.join(output_dir, "trained_agent1.pth"))
    torch.save(trained_agent2, os.path.join(output_dir, "trained_agent2.pth"))

if __name__ == "__main__":
    __main__()
