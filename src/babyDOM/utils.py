import torch
from game_engine.game import Game
from vectorization.vectorizer import DominionVectorizer

def convert_action_probs_to_readable(action_probs: torch.Tensor, vectorizer: DominionVectorizer, game: Game):
    """
    Convert the action probabilities to a readable format.
    """
    readable_action_probs = ", ".join(
        f"{vectorizer.devectorize_action(i, game.current_player()).shorthand()}: {round(prob, 4)}"
        for i, prob in enumerate(action_probs.tolist()) if prob > 0
    )
    return readable_action_probs