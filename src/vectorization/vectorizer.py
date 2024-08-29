import numpy as np
from typing import List, Dict
from game_engine.game import Game
from game_engine.action import Action, ActionType
from game_engine.cards.card import CardType, Card
from game_engine.cards.card_instances import CARD_MAP

class DominionVectorizer:
    def __init__(self, card_types: List[str]):
        self.card_types = card_types
        self.action_space_size = len(card_types) * 2 + 2  # Play card, buy card, end action, end buy
        
    def vectorize_observation(self, game: Game) -> np.ndarray:
        observation = game.get_observation_state()
        current_player = observation['current_player_state']
        opponent = observation['opponent_state']
        game_state = observation['game_state']

        # Vectorize card counts
        hand = self._vectorize_cards(current_player['hand'])
        draw_pile = self._vectorize_cards(current_player['draw_pile_count'])
        discard_pile = self._vectorize_cards(current_player['discard_pile_count'])
        played_cards = self._vectorize_cards(current_player['played_cards'])
        opponent_deck = self._vectorize_cards(opponent['deck_count'])
        supply = self._vectorize_cards(game_state['supply_piles'])

        # Vectorize other game state information
        other_info = np.array([
            observation['current_player_name'] == 'Player 1',  # 0 if Player 2, 1 if Player 1
            game_state['current_phase'],
            current_player['actions'],
            current_player['buys'],
            current_player['coins'],
            game_state['game_over']
        ])

        # Concatenate all information into a single vector
        return np.concatenate([hand, draw_pile, discard_pile, played_cards, opponent_deck, supply, other_info])

    def _vectorize_cards(self, cards: Dict[str, int]) -> np.ndarray:
        vector = np.zeros(len(self.card_types))
        for card_name, count in cards.items():
            if card_name in self.card_types:
                vector[self.card_types.index(card_name)] = count
        return vector

    def vectorize_action(self, action: Action) -> int:
        if action.action_type == ActionType.END_ACTION:
            return self.action_space_size - 2
        elif action.action_type == ActionType.END_BUY:
            return self.action_space_size - 1
        else:  # PLAY or BUY
            card_index = self.card_types.index(action.card.name)
            action_offset = 0 if action.action_type == ActionType.PLAY else 1
            return card_index * 2 + action_offset

    def devectorize_action(self, action_index: int, player) -> Action:
        if action_index == self.action_space_size - 2:
            return Action(player, ActionType.END_ACTION)
        elif action_index == self.action_space_size - 1:
            return Action(player, ActionType.END_BUY)
        else:
            card_index = action_index // 2
            action_type = ActionType.PLAY if action_index % 2 == 0 else ActionType.BUY
            card_name = self.card_types[card_index]
            return Action(player, action_type, CARD_MAP[card_name])

    def get_action_mask(self, game: Game) -> np.ndarray:
        valid_actions = game.get_valid_actions()
        mask = np.zeros(self.action_space_size, dtype=np.int8)
        
        for action in valid_actions:
            mask[self.vectorize_action(action)] = 1
        
        return mask