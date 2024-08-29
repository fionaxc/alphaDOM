# AlphaDOM

AlphaDOM is a project by Fiona Cai and Kevin Jiang focused on building a Deep Reinforcement Learning AI to play Dominion! Dominion is a deck-building card game where players compete to accumulate the most victory points by purchasing cards from a shared supply and strategically building their personal decks to optimize their actions, buys, and overall strategy.

### Agents

- **BabyDOM**: Our first RL agent that uses Proximal Policy Optimization with Clipped Surrogate Objective (PPO-Clip) and self-play to learn and play a basic Kingdom (Chapel, Festival, Village, Smithy, Market, Moneylender, Witch, Lab)

### Repository Structure

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

### Running Unit Tests

To ensure the game engine is functioning correctly, we have included a suite of unit tests:

```bash
python3 -m unittest discover -s tests
```

### Running the Training Process

To run the main training process, use the following command from the root directory of the project:

```bash
python3 main.py
```

This will start the training process using the parameters defined in the `main.py` file. The script will:

1. Initialize the game engine and vectorizer
2. Set up training parameters
3. Run the PPO training algorithm
4. Save the trained agents and output files
