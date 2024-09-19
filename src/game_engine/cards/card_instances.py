from .card import Card, CardType
from ..effects import CompositeEffect, DrawCardsEffect, AddMoneyEffect, AddActionsEffect, AddBuysEffect, ConditionalEffect, TrashCardInHandEffect, AddCurseEffect, FirstPlayedEffect, DrawOpponentsCardsEffect

def has_copper(player, game):
    # Check if the player has a Copper card in hand
    return any(card == DEFAULT_CARDS["Copper"] for card in player.hand)

class Const():
    def __init__(self, value):
        self.value = value

    def get_value(self, player):
        return self.value

class DeckSize():
    def __init__(self, divide_by=1):
        self.divide_by = divide_by

    def get_value(self, player):
        return len(player.all_cards()) // self.divide_by

# Constants mapping card names to Card objects
DEFAULT_CARDS = {
    "Copper": Card(
        name="Copper",
        type=CardType.TREASURE,
        cost=0,
        victory_points=Const(0),
        effect=CompositeEffect(effects=[AddMoneyEffect(amount=1)])
    ),
    "Silver": Card(
        name="Silver",
        type=CardType.TREASURE,
        cost=3,
        victory_points=Const(0),
        effect=CompositeEffect(effects=[AddMoneyEffect(amount=2), ])
    ),
    "Gold": Card(
        name="Gold",
        type=CardType.TREASURE,
        cost=6,
        victory_points=Const(0),
        effect=CompositeEffect(effects=[AddMoneyEffect(amount=3)])
    ),
    "Estate": Card(
        name="Estate",
        type=CardType.VICTORY,
        cost=2,
        victory_points=Const(1),
        effect=None
    ),
    "Duchy": Card(
        name="Duchy",
        type=CardType.VICTORY,
        cost=5,
        victory_points=Const(3),
        effect=None
    ),
    "Province": Card(
        name="Province",
        type=CardType.VICTORY,
        cost=8,
        victory_points=Const(6),
        effect=None
    ),
    "Curse": Card(
        name="Curse",
        type=CardType.CURSE,
        cost=0,
        victory_points=Const(-1),
        effect=None
    ),
}

BASIC_KINGDOM_CARDS = {
    "Chapel": Card(
        name="Chapel",
        type=CardType.ACTION,
        cost=2,
        victory_points=Const(0),
        effect=None
    ),
    "Village": Card(
        name="Village",
        type=CardType.ACTION,
        cost=3,
        victory_points=Const(0),
        effect=CompositeEffect(effects=[DrawCardsEffect(num_cards=1), AddActionsEffect(num_actions=2)])
    ),
    "Merchant": Card(
        name="Merchant",
        type=CardType.ACTION,
        cost=3,
        victory_points=Const(0),
        effect=CompositeEffect(effects=[DrawCardsEffect(num_cards=1), AddActionsEffect(num_actions=1)]),
        activation_effect=FirstPlayedEffect(card=DEFAULT_CARDS["Silver"], effect=AddMoneyEffect(amount=1))
    ),
    "Smithy": Card(
        name="Smithy",
        type=CardType.ACTION,
        cost=4,
        victory_points=Const(0),
        effect=CompositeEffect(effects=[DrawCardsEffect(num_cards=3)])
    ),
    "Gardens": Card(
        name="Gardens",
        type=CardType.VICTORY,
        cost=4,
        victory_points=DeckSize(divide_by=10),
        effect=CompositeEffect(effects=[DrawCardsEffect(num_cards=3)])
    ),
    "Moneylender": Card(
        name="Moneylender",
        type=CardType.ACTION,
        cost=4,
        victory_points=Const(0),
        effect=CompositeEffect(effects=[
            ConditionalEffect(
                condition=has_copper,
                effect=CompositeEffect(effects=[
                    TrashCardInHandEffect(card=DEFAULT_CARDS["Copper"]),
                    AddMoneyEffect(amount=3)
                ])
            )
        ])
    ),
    "Festival": Card(
        name="Festival",
        type=CardType.ACTION,
        cost=5,
        victory_points=Const(0),
        effect=CompositeEffect(effects=[AddActionsEffect(num_actions=2), AddBuysEffect(num_buys=1), AddMoneyEffect(amount=2)])
    ),
    "Laboratory": Card(
        name="Laboratory",
        type=CardType.ACTION,
        cost=5,
        victory_points=Const(0),
        effect=CompositeEffect(effects=[DrawCardsEffect(num_cards=2), AddActionsEffect(num_actions=1)])
    ),
    "Council Room": Card(
        name="Council Room",
        type=CardType.ACTION,
        cost=5,
        victory_points=Const(0),
        effect=CompositeEffect(effects=[DrawCardsEffect(num_cards=4), AddBuysEffect(num_buys=1), DrawOpponentsCardsEffect(num_cards=1)])
    ),
    "Market": Card(
        name="Market",
        type=CardType.ACTION,
        cost=5,
        victory_points=Const(0),
        effect=CompositeEffect(effects=[DrawCardsEffect(num_cards=1), AddActionsEffect(num_actions=1), AddBuysEffect(num_buys=1), AddMoneyEffect(amount=1)])
    ),
    "Witch": Card(
        name="Witch",
        type=CardType.ACTION | CardType.ATTACK,
        cost=5,
        victory_points=Const(0),
        effect=CompositeEffect(effects=[DrawCardsEffect(num_cards=2), AddCurseEffect(num_curses=1)])
    ),
}

# Define the supply card limits for each card
SUPPLY_CARD_LIMITS = {
    "Copper": 60,
    "Silver": 40,
    "Gold": 30,
    "Estate": 14,
    "Duchy": 8,
    "Province": 8,
    "Curse": 10,
    "Chapel": 10,
    "Village": 10,
    "Merchant": 10,
    "Smithy": 10,
    "Gardens": 8,
    "Moneylender": 10,
    "Festival": 10,
    "Laboratory": 10,
    "Council Room": 10,
    "Market": 10,
    "Witch": 10,
}

# Combine default and kingdom cards into a single map
CARD_MAP = {**DEFAULT_CARDS, **BASIC_KINGDOM_CARDS}
