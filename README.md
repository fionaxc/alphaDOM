# AlphaDOM

AlphaDOM is a project by Fiona Cai and Kevin Jiang focused on building a Deep Reinforcement Learning AI to play Dominion! Dominion is a deck-building card game where players compete to accumulate the most victory points by purchasing cards from a shared supply and strategically building their personal decks to optimize their actions, buys, and overall strategy.

## Agents

- **BabyDOM**: Our first RL agent that uses Proximal Policy Optimization with Clipped Surrogate Objective (PPO-Clip) and self-play to learn and play a basic Kingdom (Chapel, Festival, Village, Smithy, Market, Moneylender, Witch, Lab)

## Repository Structure

The repository is generally organized as follows:

- `src/`: Main source code for RL agents
  - `babyDOM/`: Our first RL agent that uses PPO and self-play
  - `game_engine/`: Core game engine for running a game of Dominion
    - `cards/`: Definitions and instances of all cards used in the game.
    - `action.py`: Defines a possible action a player can take
    - `effects.py`: Defines various effects that can be applied by cards (e.g. drawing cards)
    - `game.py`: Game state information
    - `phase.py`: Represents phases in the game
    - `player.py`: Player state and actions.
  - `vectorization/`: Converts game engine states to vectors for RL agents to use
  - `main.py`: Main script to run the training process
  - `utils.py`: Utility functions
- `tests/`: Contains unit tests for the game engine

## Running Unit Tests

To ensure the game engine is functioning correctly, we have included a suite of unit tests:

```bash
python3 -m unittest discover -s tests
```

## Running the Training Process

1. Create a virtual environment:

   ```bash
   python3 -m venv alphaDOM_env
   ```

2. Activate the virtual environment:

   ```bash
   source alphaDOM_env/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. To run the main training process, use the following command from the root directory of the project:

   ```bash
   python3 src/main.py [arguments]
   ```

5. Results will be saved to the src/output directory!

### Command-line Arguments

The script accepts the following command-line arguments:

| Argument          | Type | Default       | Description                        |
| ----------------- | ---- | ------------- | ---------------------------------- |
| `--num_games`     | int  | 100           | Number of games to train           |
| `--batch_size`    | int  | 32            | Batch size (in games) for training |
| `--update_epochs` | int  | 10            | Number of epochs for each update   |
| `--hidden_size`   | int  | 64            | Hidden size of the neural network  |
| `--run_id`        | str  | "default_run" | Unique identifier for this run     |

Example:

```bash
python3 main.py --num_games 200 --batch_size 64 --run_id my_custom_run
```

This will start the training process using the specified parameters. The script will:

1. Parse command-line arguments
2. Initialize the game engine and vectorizer
3. Create an output directory for the run
4. Run the PPO training algorithm
5. Save the trained agents and output files
6. Save the training parameters to a CSV file
