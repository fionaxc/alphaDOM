from enum import Enum
from .cards.card import Card

class ActionType(Enum):
    PLAY = "play"
    BUY = "buy"
    END_ACTION = "end_action"
    END_BUY = "end_buy"

class Action:
    def __init__(self, player, action_type: ActionType, card: Card = None):
        """
        Initialize an action that a player can perform. E.g. Play a Village card, buy a Silver card, etc.

        Args:
            player (Player): The player performing the action.
            action_type (ActionType): The type of action being performed (e.g., PLAY, BUY, END_ACTION, END_BUY).
            card (Card, optional): The card involved in the action, if any. Defaults to None.
        """
        self.action_type = action_type
        self.card = card
        self.player = player
    
    def apply(self):
        action_methods = {
            ActionType.PLAY: self.player.play_card,
            ActionType.BUY: self.player.buy_card,
            ActionType.END_ACTION: self.player.end_action_phase,
            ActionType.END_BUY: self.player.end_buy_phase
        }
        
        # Call the appropriate method based on action_type
        action_methods[self.action_type](self.card) if self.card else action_methods[self.action_type]()

    def __repr__(self):
        return f"Action(player={self.player.name}, type={self.action_type}, card={self.card})"