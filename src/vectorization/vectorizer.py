import numpy as np
from typing import List, Dict, Union
from game_engine.game import Game
from game_engine.action import Action, ActionType
from game_engine.cards.card import Card
from game_engine.cards.card_instances import CARD_MAP
from game_engine.phase import Phase

class DominionVectorizer:
    def __init__(self, card_types: List[str]):
        """
        The Vectorizer converts game states into vectors and vice versa for RL algorithms to use.
        """
        self.card_types = card_types
        self.action_space_size = len(card_types) * 2 + 2  # For each card, there are 2 actions (PLAY and BUY) + 2 end actions (END_ACTION and END_BUY)

    def vectorize_observation(self, game: Game) -> np.ndarray:
        """
        Converts the game state into a vector representation. The resulting vector has the following components (let n = len(card_types)):

        - Vectors where information is a list (because we know the exact order of cards, but that's not used yet)
            - Hand cards (n)
            - Played cards (n)
        - Vectors where we don't know the exact order of cards, so it's a count of each card.
            - Draw pile count (n)
            - Discard pile count (n)
            - Opponent deck count (n)
            - Supply piles (n)
        - Other game state information 
            - Current phase (1 element)
            - # of actions (1 element)
            - # of buys (1 element)
            - # of coins (1 element)
            - Game over (1 element)

        Total vector length: 6 * n + 5
        """
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

        phase_encoding = 0 if game_state['current_phase'] == Phase.ACTION.value else 1 if game_state["current_phase"] == Phase.BUY.value else 2

        # Vectorize other game state information
        other_info = np.array([
            phase_encoding, # What phase of the game are we in?
            current_player['actions'], # How many actions does the current player have left?
            current_player['buys'], # How many buys does the current player have left?
            current_player['coins'], # How many coins does the current player have?
            1 if game_state['game_over'] else 0 # Is the game over?
        ], dtype=np.float32)

        # Concatenate all information into a single vector
        return np.concatenate([hand, draw_pile, discard_pile, played_cards, opponent_deck, supply, other_info]).astype(np.float32)

    def _vectorize_cards(self, cards: Union[Dict[str, int], List[Card]]) -> np.ndarray:
        """
        Converts a list or dictionary of cards into a vector where each card has a count.
        """
        vector = np.zeros(len(self.card_types), dtype=np.float32)
        if isinstance(cards, dict):
            for card_name, count in cards.items():
                if card_name in self.card_types:
                    vector[self.card_types.index(card_name)] = count
        elif isinstance(cards, list):
            for card in cards:
                if card.name in self.card_types:
                    vector[self.card_types.index(card.name)] += 1
        return vector

    def vectorize_action(self, action: Action) -> int:
        """
        Converts an action into an index in the action space.
        """
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
        else: # PLAY or BUY
            card_index = action_index // 2
            action_type = ActionType.PLAY if action_index % 2 == 0 else ActionType.BUY
            card_name = self.card_types[card_index]
            return Action(player, action_type, CARD_MAP[card_name])

    def get_action_mask(self, game: Game) -> np.ndarray:
        """
        Returns a mask of valid actions for the current game state (1 for valid actions, 0 for invalid actions).
        """
        valid_actions = game.get_valid_actions()

        mask = np.zeros(self.action_space_size, dtype=np.int8)
        
        for action in valid_actions:
            mask[self.vectorize_action(action)] = 1
    
        return mask
    
