from enum import Enum
from cards.card_instances import CARD_MAP, SUPPLY_CARD_LIMITS

DEFAULT_SUPPLY = ["Copper", "Silver", "Gold", "Estate", "Duchy", "Province", "Curse"]
SIMPLE_SETUP = ["Chapel", "Village", "Smithy", "Moneylender", "Festival", "Laboratory", "Market", "Witch"]


class Phase(Enum):
    ACTION = "action"
    BUY = "buy"
    CLEANUP = "cleanup"

class Game:
    def __init__(self, kingdom_cards = SIMPLE_SETUP):
        # Game setup
        self.supply_piles = {CARD_MAP[card]: SUPPLY_CARD_LIMITS[card] for card in DEFAULT_SUPPLY + kingdom_cards}
        self.players = []
        self.current_player_turn = 0
        self.current_phase = Phase.ACTION