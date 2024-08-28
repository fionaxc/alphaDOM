from .card import Card, CardType

DEFAULT_CARDS = {
    "Copper": Card(
        name="Copper",
        type=CardType.TREASURE,
        cost=0,
        victory_points=0,
        effect=None
    ),
    "Silver": Card(
        name="Silver",
        type=CardType.TREASURE,
        cost=3,
        victory_points=0,
        effect=None
    ),
    "Gold": Card(
        name="Gold",
        type=CardType.TREASURE,
        cost=6,
        victory_points=0,
        effect=None
    ),
    "Estate": Card(
        name="Estate",
        type=CardType.VICTORY,
        cost=2,
        victory_points=1,
        effect=None
    ),
    "Duchy": Card(
        name="Duchy",
        type=CardType.VICTORY,
        cost=5,
        victory_points=3,
        effect=None
    ),
    "Province": Card(
        name="Province",
        type=CardType.VICTORY,
        cost=8,
        victory_points=6,
        effect=None
    ),
    "Curse": Card(
        name="Curse",
        type=CardType.CURSE,
        cost=0,
        victory_points=-1,
        effect=None
    ),
}

BASIC_KINGDOM_CARDS = {
    "Chapel": Card(
        name="Chapel",
        type=CardType.ACTION,
        cost=2,
        victory_points=0,
        effect=None
    ),
    "Village": Card(
        name="Village",
        type=CardType.ACTION,
        cost=3,
        victory_points=0,
        effect=None
    ),
    "Smithy": Card(
        name="Smithy",
        type=CardType.ACTION,
        cost=4,
        victory_points=0,
        effect=None
    ),
    "Moneylender": Card(
        name="Moneylender",
        type=CardType.ACTION,
        cost=4,
        victory_points=0,
        effect=None
    ),
    "Festival": Card(
        name="Festival",
        type=CardType.ACTION,
        cost=5,
        victory_points=0,
        effect=None
    ),
    "Laboratory": Card(
        name="Laboratory",
        type=CardType.ACTION,
        cost=5,
        victory_points=0,
        effect=None
    ),
    "Market": Card(
        name="Market",
        type=CardType.ACTION,
        cost=5,
        victory_points=0,
        effect=None
    ),
    "Witch": Card(
        name="Witch",
        type=CardType.ACTION | CardType.ATTACK,
        cost=5,
        victory_points=0,
        effect=None
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
    "Smithy": 10,
    "Moneylender": 10,
    "Festival": 10,
    "Laboratory": 10,
    "Market": 10,
    "Witch": 10,
}

CARD_MAP = {**DEFAULT_CARDS, **BASIC_KINGDOM_CARDS}