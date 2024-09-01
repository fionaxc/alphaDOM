import unittest
import sys
import os
import random
from pprint import pprint  # Import pprint for pretty-printing

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from game_engine.game import Game

class TestGameEngine(unittest.TestCase):
    def print_title(self, title):
        print("\n" + "="*35)
        print(f"Test: {title}")
        print("="*35 + "\n")

    def simplified_observation_state(self, observation_state):
        # Redundant to include card_costs every time
        if 'game_state' in observation_state:
            observation_state['game_state'].pop('card_costs', None)
        return observation_state
        

    def test_initial_game_state(self):
        self.print_title("Initial Game State")

        # Initialize a new game
        game = Game()

        # Print the initial game state and valid actions
        print("Initial Game State:")
        pprint(self.simplified_observation_state(game.get_observation_state()))  # Pretty-print the game state
        print("\nInitial Valid Actions:")
        pprint(game.get_valid_actions())  # Pretty-print the valid actions
    
    def run_n_randomized_actions(self, n):
        self.print_title(f"{n} Randomized Actions")

        # Initialize a new game
        game = Game()

        # Print the initial game state and valid actions
        print("Initial Game State:")
        pprint(self.simplified_observation_state(game.get_observation_state()))
        print("\nInitial Valid Actions:")
        pprint(game.get_valid_actions())

        for i in range(1, n+1):
            if game.game_over:
                print(f"Game over after {i-1} actions on turn {game.turn_number}")
                pprint(self.simplified_observation_state(game.get_observation_state()))
                winner = game.winner()
                if winner is None:
                    print("Winner: None")
                else:
                    print(f"Winner: {winner.name}")
                print(f"Player 1 VP: {game.players[0].victory_points()}, Player 2 VP: {game.players[1].victory_points()}")
                break

            # Choose and apply a random action from the valid actions
            action = random.choice(game.get_valid_actions())
            action.apply()

            # Print the game state and valid actions after the action
            print(f"\nGame State after {action}:")
            pprint(self.simplified_observation_state(game.get_observation_state()))  # Pretty-print the game state
            print(f"\nValid Actions after {action}:")
            pprint(game.get_valid_actions())  # Pretty-print the valid actions
    
    def test_five_randomized_actions(self):
        self.run_n_randomized_actions(5)
    
    def test_fifty_randomized_actions(self):
        self.run_n_randomized_actions(50)
    
    def test_five_hundred_randomized_actions(self):
        self.run_n_randomized_actions(500)

if __name__ == '__main__':
    unittest.main()