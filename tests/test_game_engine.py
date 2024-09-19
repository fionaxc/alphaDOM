import unittest
import sys
import os
import random
from pprint import pprint  # Import pprint for pretty-printing
from src.game_engine.cards.card_instances import *

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
    
    def pretty_print_initial_game_state_and_valid_actions(self, game):
        print("Initial Game State:")
        pprint(self.simplified_observation_state(game.get_observation_state()))  # Pretty-print the game state
        print("\nInitial Valid Actions:")
        pprint(game.get_valid_actions())  # Pretty-print the valid actions

    def pretty_print_intermediate_game_state_and_valid_actions(self, action, game, opponent_state=False):
        print(f"\n{'Opponent' if opponent_state else 'Game'} State after {action}:")
        if opponent_state:
            pprint(self.simplified_observation_state(game.get_other_player().get_player_observation_state()))  # Pretty-print the game state
        else:
            pprint(self.simplified_observation_state(game.get_observation_state()))  # Pretty-print the game state
        print(f"\nValid Actions after {action}:")
        pprint(game.get_valid_actions())  # Pretty-print the valid actions

    def test_initial_game_state(self):
        self.print_title("Initial Game State")

        # Initialize a new game
        game = Game()

        # Print the initial game state and valid actions
        self.pretty_print_initial_game_state_and_valid_actions(game)
    
    def run_n_randomized_actions(self, n):
        self.print_title(f"{n} Randomized Actions")

        # Initialize a new game
        game = Game(["Village", "Smithy", "Moneylender", "Festival", "Laboratory", "Market", "Merchant", "Council Room", "Witch", "Gardens"])

        # Print the initial game state and valid actions
        self.pretty_print_initial_game_state_and_valid_actions(game)

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
            self.pretty_print_intermediate_game_state_and_valid_actions(action, game)
    
    def test_five_randomized_actions(self):
        self.run_n_randomized_actions(5)
    
    def test_fifty_randomized_actions(self):
        self.run_n_randomized_actions(50)
    
    def test_five_hundred_randomized_actions(self):
        self.run_n_randomized_actions(500)

    def test_merchant_card(self):
        self.print_title("Merchant Card")

        # Initialize a new game
        game = Game(["Village", "Smithy", "Moneylender", "Festival", "Laboratory", "Market", "Merchant"])
        current_player = game.current_player()

        # Initialize current player's hand with merchant and silver card objects
        current_player.hand = [CARD_MAP['Merchant'], CARD_MAP['Silver'], CARD_MAP['Copper'], CARD_MAP['Silver']]
        current_player.draw_pile = []

        # Print the initial game state and valid actions
        self.pretty_print_initial_game_state_and_valid_actions(game)

        # Play merchant
        [a for a in game.get_valid_actions() if a.card and a.card.name == 'Merchant'][0].apply()
        self.pretty_print_intermediate_game_state_and_valid_actions("Playing Merchant", game)

        # End action
        game.get_valid_actions()[0].apply()
        self.pretty_print_intermediate_game_state_and_valid_actions("Ending Action", game)

        # End Buy for both players to get back to original player
        [a for a in game.get_valid_actions() if a.card is None][0].apply()
        [a for a in game.get_valid_actions() if a.card is None][0].apply()
        self.pretty_print_intermediate_game_state_and_valid_actions("Ending Buy for both players", game)

        # Play merchant second time
        [a for a in game.get_valid_actions() if a.card and a.card.name == 'Merchant'][0].apply()
        self.pretty_print_intermediate_game_state_and_valid_actions("Playing Merchant second time", game)

        # End action second time
        game.get_valid_actions()[0].apply()
        self.pretty_print_intermediate_game_state_and_valid_actions("Ending Action second time", game)
    
    def test_council_room_card(self):
        self.print_title("Council Room Card")

        # Initialize a new game
        game = Game(["Village", "Smithy", "Moneylender", "Festival", "Laboratory", "Market", "Council Room"])
        current_player = game.current_player()

        # Initialize current player's hand
        current_player.hand = [CARD_MAP['Council Room']]
        current_player.draw_pile = [CARD_MAP['Silver'], CARD_MAP['Copper'], CARD_MAP['Estate'], CARD_MAP['Gold'], CARD_MAP['Duchy']]

        # Print the initial game state and valid actions
        self.pretty_print_initial_game_state_and_valid_actions(game)

        # Play council room
        [a for a in game.get_valid_actions() if a.card and a.card.name == 'Council Room'][0].apply()
        self.pretty_print_intermediate_game_state_and_valid_actions("Playing Council Room", game)

        # End action
        game.get_valid_actions()[0].apply()
        self.pretty_print_intermediate_game_state_and_valid_actions("Ending Action", game, opponent_state=True)
    
    def test_gardens_card(self):
        self.print_title("Gardens Card")

        # Initialize a new game
        game = Game(["Village", "Smithy", "Moneylender", "Festival", "Laboratory", "Market", "Gardens"])
        current_player = game.current_player()
        print(f'Max vp: {game.get_maximum_possible_vp()}')

        # Initialize current player's deck with copper cards and gardens
        current_player.hand = [CARD_MAP['Gardens']]
        current_player.draw_pile = [CARD_MAP['Copper']] * 46

        # Print the initial game state and valid actions
        self.pretty_print_initial_game_state_and_valid_actions(game)

        # Initialize current player's deck with copper cards and gardens
        current_player.hand = [CARD_MAP['Gardens']]
        current_player.draw_pile = [CARD_MAP['Copper']] * 2

        # Print the initial game state and valid actions
        self.pretty_print_initial_game_state_and_valid_actions(game)

                # Initialize current player's deck with copper cards and gardens
        current_player.hand = [CARD_MAP['Gardens']]
        current_player.draw_pile = [CARD_MAP['Copper']] * 74

        # Print the initial game state and valid actions
        self.pretty_print_initial_game_state_and_valid_actions(game)
    

        
if __name__ == '__main__':
    unittest.main()