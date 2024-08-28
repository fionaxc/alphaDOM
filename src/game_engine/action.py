from enum import Enum

class ActionType(Enum):
    PLAY = "play"
    BUY = "buy"
    END_ACTION = "end_action"
    END_BUY = "end_buy"

class Action:
    def __init__(self, player, action_type: ActionType, card=None):
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