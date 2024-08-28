import unittest
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from game_engine.game import Game

class TestGameEngine(unittest.TestCase):
    def test_initial_game_state(self):
        # Initialize a new game
        game = Game()

        # Print the initial game state
        print("Initial Game State:")
        print(game.get_observation_state())

if __name__ == '__main__':
    unittest.main()