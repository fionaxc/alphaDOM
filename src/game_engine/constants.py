from .cards.card_instances import DEFAULT_CARDS, BASIC_KINGDOM_CARDS

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
    "Smithy": 10,
    "Moneylender": 10,
    "Festival": 10,
    "Laboratory": 10,
    "Market": 10,
    "Witch": 10,
}

# Combine default and kingdom cards into a single map
CARD_MAP = {**DEFAULT_CARDS, **BASIC_KINGDOM_CARDS}