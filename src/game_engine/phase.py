from enum import Enum

# Enum representing the different phases of the game
class Phase(Enum):
    ACTION = "action"
    BUY = "buy"
    CLEANUP = "cleanup"