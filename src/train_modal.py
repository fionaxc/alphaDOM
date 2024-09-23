import modal
from modal import App, web_endpoint, Image
from babyDOM.train import ppo_train
from game_engine.game import Game
from vectorization.vectorizer import DominionVectorizer
import os
import torch
from babyDOM.ppo import PPOAgent

def create_image():
    return (
        Image.debian_slim()
        .pip_install("torch", "numpy", "tqdm")
    )

volume = modal.Volume.from_name("dominion-training-output", create_if_missing=True)
app = App("dominion-ppo-training", image=create_image())

@app.function(gpu=modal.gpu.T4(count=1), volumes={"/root/output": volume})
def train_dominion(num_games, batch_size, update_epochs, hidden_size, run_id, checkpoint_path=None):
    game_engine = Game()
    vectorizer = DominionVectorizer(game_engine.all_card_names)
    if run_id is None:
        run_id = "0922_SIMPLE_run1_games{}_batchsize{}_updateepochs{}_hidden{}".format(num_games, batch_size, update_epochs, hidden_size)

    # Create output directory within the alphaDom repository
    output_dir = os.path.join("/root/output", run_id)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the agent
    obs_dim = vectorizer.vectorize_observation(game_engine).shape[0]
    action_dim = vectorizer.action_space_size
    agent = PPOAgent(obs_dim, action_dim, hidden_size, device=device, output_dir=output_dir, checkpoint_path=checkpoint_path)
    agent.to(device)

    # Create a VectorizedDominionEnv
    env = VectorizedDominionEnv(batch_size, game_engine, vectorizer, device)

    # Train the agent
    trained_agent = ppo_train(
        env=env,
        agent=agent,
        vectorizer=vectorizer,
        run_id=run_id,
        output_dir=output_dir,
        num_games=num_games,
        batch_size=batch_size,
        update_epochs=update_epochs
    )

    # Save the trained agent to the output directory
    torch.save(trained_agent, os.path.join(output_dir, "trained_agent.pth"))
    return output_dir

@app.local_entrypoint()
def main(num_games: int = 10000000, 
         batch_size: int = 32, 
         update_epochs: int = 5, 
         hidden_size: int = 128, 
         run_id: str = "default_run", 
         checkpoint_path: str = None):
    output_dir = train_dominion.remote(num_games, batch_size, update_epochs, hidden_size, run_id, checkpoint_path)
    print(f"Training completed. Output saved in: {output_dir}")