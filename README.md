# AlphaDOM

AlphaDOM is a project focused on building a Deep Reinforcement Learning AI to play Dominion! [Dominion](<https://en.wikipedia.org/wiki/Dominion_(card_game)>) is a deck-building card game where players compete to accumulate the most victory points by purchasing cards from a shared supply and strategically building their personal decks to optimize their actions, buys, and overall strategy.

## Agents

- **BabyDOM**: Our first RL agent built from scratch that uses Proximal Policy Optimization with Clipped Surrogate Objective (PPO-Clip) and self-play to learn and play one basic kingdom: Village, Smithy, Festival, Market, Moneylender, Witch, Lab.
  - After 100K games of self-play, BabyDOM discovered a faster strategy on this board that we've never seen before! It would do a Witch Big Money strategy and clear the Curse, Duchy, and Estate piles before any player can get a reliable engine working in time (note that the curse pile gets cleared quite quickly due to both players trying to witch each other). We've provided a model version trained on 3M+ games of this board that you can play with in the `sample_agents` folder using `play.py`.
- **BabyDOM v2**: Our second RL agent generalizes BabyDOM to all base game kingdoms by randomly selecting a board for each game simulation in training (the observation state is extended similarly).

## Repository Structure

The repository is generally organized as follows:

- `src/`: Main source code for RL agents
  - `babyDOM/`: Our first RL agent that uses PPO and self-play for one kingdom
  - `game_engine/`: Core game engine for running a game of Dominion
    - `cards/`: Definitions and instances of all cards used in the game.
    - `action.py`: Defines a possible action a player can take
    - `effects.py`: Defines various effects that can be applied by cards (e.g. drawing cards)
    - `game.py`: Game state information
    - `phase.py`: Represents phases in the game
    - `player.py`: Player state and actions.
  - `sample_agents`: Sample torch models trained on different board setups
  - `vectorization/`: Converts game engine states to vectors for RL agents to use
  - `main.py`: Main script to run the training process
  - `play.py`: Script to play test against sample agents
- `tests/`: Contains unit tests for the game engine

## Play Testing

Play against any of our sample agents by running the following from the root directory:

```bash
python3 src/play.py
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

#### Command-line Arguments

The script accepts the following command-line arguments:

| Argument            | Type | Default       | Description                                          |
| ------------------- | ---- | ------------- | ---------------------------------------------------- |
| `--num_games`       | int  | 1000000       | Number of games to train                             |
| `--batch_size`      | int  | 32            | Batch size (in games) for training                   |
| `--update_epochs`   | int  | 5             | Number of epochs for each update                     |
| `--hidden_size`     | int  | 256           | Hidden size of the neural network                    |
| `--run_id`          | str  | "default_run" | Unique identifier for this run                       |
| `--checkpoint_path` | str  | None          | Path to a checkpoint file to load model weights from |

## Running Unit Tests

To ensure the game engine is functioning correctly, we have included a suite of unit tests:

```bash
python3 -m unittest discover -s tests
```
