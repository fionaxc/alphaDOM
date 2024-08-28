import random
from enum import Enum
from cards.card_instances import CARD_MAP, SUPPLY_CARD_LIMITS
from player import PlayerState

DEFAULT_SUPPLY = ["Copper", "Silver", "Gold", "Estate", "Duchy", "Province", "Curse"]
SIMPLE_SETUP = ["Chapel", "Village", "Smithy", "Moneylender", "Festival", "Laboratory", "Market", "Witch"]

class Phase(Enum):
    ACTION = "action"
    BUY = "buy"
    CLEANUP = "cleanup"

class Game:
    def __init__(self, kingdom_cards = SIMPLE_SETUP, num_players = 2):
        # Game setup
        self.supply_piles = {CARD_MAP[card]: SUPPLY_CARD_LIMITS[card] for card in DEFAULT_SUPPLY + kingdom_cards}
        self.players = [PlayerState(f"Player {i+1}", self) for i in range(num_players)]
        self.current_player_turn = 0
        self.current_phase = Phase.ACTION
        
        # Start game
        self.start_game()
    
    def get_other_player(self):
        return self.players[(self.current_player_turn + 1) % len(self.players)]
    
    def start_game(self):
        # Randomly select player to go first
        self.current_player_turn = random.randint(0, len(self.players) - 1)

        # Initialize each player with 7 coppers, 3 estates and then draw 5 cards
        for player in self.players:
            player.discard_pile = [CARD_MAP["Copper"]] * 7 + [CARD_MAP["Estate"]] * 3
            player.cleanup_cards()

    def next_phase(self):
        self.current_phase = Phase((self.current_phase + 1) % len(Phase))

    def next_player(self):
        self.current_player_turn = (self.current_player_turn + 1) % len(self.players)

    def current_player(self):
        return self.players[self.current_player_turn]