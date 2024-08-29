import random
from game_engine.cards.card_instances import CARD_MAP, SUPPLY_CARD_LIMITS
from .player import PlayerState
from .phase import Phase

DEFAULT_SUPPLY = ["Copper", "Silver", "Gold", "Estate", "Duchy", "Province", "Curse"]
SIMPLE_SETUP = ["Village", "Smithy", "Moneylender", "Festival", "Laboratory", "Market", "Witch"]
# SIMPLE_CHAPEL_SETUP = ["Chapel", "Village", "Smithy", "Moneylender", "Festival", "Laboratory", "Market", "Witch"]

class Game:
    def __init__(self, kingdom_cards=SIMPLE_SETUP, num_players=2):
        """
        Initialize the game with the given kingdom cards and number of players.

        Args:
            kingdom_cards (list): List of kingdom card names to include in the game.
            num_players (int): Number of players in the game.
        """
        # Game setup
        self.all_card_names = DEFAULT_SUPPLY + kingdom_cards
        # Supply piles maps "card names" to their maximum count
        self.supply_piles = {card: SUPPLY_CARD_LIMITS[card] for card in DEFAULT_SUPPLY + kingdom_cards}
        self.players = [PlayerState(f"Player {i+1}", self) for i in range(num_players)]
        self.current_player_turn = 0
        self.current_phase = Phase.ACTION
        self.game_over = False
        self.turn_number = 1

        # Start game
        self.start_game()
    
    def get_observation_state(self):
        """
        Get the observation state for the current player.

        Returns:
            dict: A dictionary containing the current player's observation state and the game state.
        """
        observation_state = {
            **self.current_player().get_player_observation_state(),
            'game_state': {
                'current_phase': self.current_phase.value,
                'supply_piles': self.supply_piles,
                'card_costs': {card: CARD_MAP[card].cost for card in self.supply_piles.keys()},
                'game_over': self.game_over,
            }
        }
        return observation_state
    
    def get_valid_actions(self):
        """
        Get the valid actions for the current player.

        Returns:
            list: A list of valid actions that the current player can perform.
        """
        return self.current_player().get_valid_actions()
    
    def get_other_player(self):
        return self.players[(self.current_player_turn + 1) % len(self.players)]
    
    def start_game(self):
        # Randomly select player to go first
        self.current_player_turn = random.randint(0, len(self.players) - 1)

        # Initialize each player with 7 coppers, 3 estates and then draw 5 cards
        for player in self.players:
            # Remove 7 coppers and 3 estates from supply
            self.supply_piles["Copper"] -= 7
            self.supply_piles["Estate"] -= 3

            player.discard_pile = [CARD_MAP["Copper"]] * 7 + [CARD_MAP["Estate"]] * 3
            player.cleanup_cards()
    
    def next_phase(self):
        # Update the current phase to the next phase in the enum
        phases = list(Phase)
        current_index = phases.index(self.current_phase)
        self.current_phase = phases[(current_index + 1) % len(phases)]

    def next_player(self):
        # Before moving to next player, check if the game is over (no provinces, or at least 3 empty supply piles)
        if self.supply_piles["Province"] <= 0 or len([card for card, count in self.supply_piles.items() if count == 0]) >= 3:
            self.game_over = True
            return

        self.current_player_turn = (self.current_player_turn + 1) % len(self.players)
        self.turn_number += 1

    def current_player(self):
        return self.players[self.current_player_turn]
    
    def winner(self):
        if self.game_over:
            # Find the player with the most VPs
            return max(self.players, key=lambda player: player.victory_points())
        return None