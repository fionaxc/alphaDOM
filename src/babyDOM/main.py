import numpy as np
from typing import Any, Dict, List
from ppo import PPOAgent, PPOActor, PPOCritic, ppo_train
from game_engine.game import Game
from game_engine.cards.card import CardType
from game_engine.cards.card_instances import CARD_MAP
from vectorization.vectorizer import DominionVectorizer


def __main__():
    #Initialize the game engine
    game_engine = Game()

    #List of card types in the game
    card_types = list(CARD_MAP.keys())

    #Initialize the vectorizer
    vectorizer = DominionVectorizer(game_engine.all_card_names)

    trained_agent1, trained_agent2 = ppo_train(game_engine, vectorizer)

if __name__ == "__main__":
    __main__()
