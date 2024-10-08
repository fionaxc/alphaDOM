from enum import Flag, auto
from ..effects import Effect

class CardType(Flag):
    TREASURE = auto()
    CURSE = auto()
    VICTORY = auto()
    ACTION = auto()
    ATTACK = auto()

class Card:
    def __init__(self, name, type, cost, victory_points, effect: Effect, activation_effect: Effect=None):
        """
        Initialize a card (e.g. Village) with its attributes.

        Args:
            name (str): The name of the card.
            type (CardType): The type of the card (e.g., TREASURE, CURSE, VICTORY, ACTION, ATTACK).
            cost (int): The cost to buy the card.
            victory_points (int): The victory points the card provides.
            effect (Effect): The effect(s) the card has when played (e.g. +2 actions, +1 buy, etc.), defined in effects.py
            activation_effect (Effect): The effect(s) the card has when activated (e.g. merchant giving +1 money when silver is first played), defined in effects.py
        """
        self.name = name
        self.type = type
        self.cost = cost 
        self._victory_points = victory_points
        self.effect = effect
        self.activation_effect = activation_effect

    def is_type(self, card_type):
        return bool(self.type & card_type)

    def is_action(self):
        return self.is_type(CardType.ACTION)

    def is_victory(self):
        return self.is_type(CardType.VICTORY)
    
    def is_curse(self):
        return self.is_type(CardType.CURSE)

    def is_treasure(self):
        return self.is_type(CardType.TREASURE)

    def get_victory_points(self, player):
        return self._victory_points.get_value(player)

    def play(self, player, game):
        self.effect.apply(player, game)
    
    def activate(self, player, game):
        self.activation_effect.apply(player, game)

    def __eq__(self, other):
        return self.name == other.name

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name