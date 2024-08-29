import numpy as np
from typing import Dict, List, Tuple
from dominion_engine import GameState, Action, CardType  # Import from your game engine

class DominionVectorizer:
    def __init__(self, card_types: List[CardType]):
        self.card_types = card_types
        self.action_space_size = len(card_types) * 2 + 1  # Play card, buy card, or end phase

    def vectorize_observation(self, game_state: GameState, player_index: int) -> np.ndarray:
        current_player = game_state.players[player_index]
        opponent = game_state.players[1 - player_index]

        # Vectorize card counts
        hand = self._vectorize_cards(current_player.hand)
        deck = self._vectorize_cards(current_player.deck)
        discard = self._vectorize_cards(current_player.discard)
        opponent_deck = self._vectorize_cards(opponent.deck)
        opponent_discard = self._vectorize_cards(opponent.discard)
        supply = self._vectorize_cards(game_state.supply)
        trash = self._vectorize_cards(game_state.trash)

        # Vectorize other game state information
        other_info = np.array([
            player_index,
            game_state.phase,
            current_player.vp,
            current_player.money,
            current_player.buys,
            current_player.actions,
            opponent.vp
        ])

        # Concatenate all information into a single vector
        return np.concatenate([hand, deck, discard, opponent_deck, opponent_discard, supply, trash, other_info])

    def _vectorize_cards(self, cards: List[CardType]) -> np.ndarray:
        vector = np.zeros(len(self.card_types))
        for card in cards:
            vector[self.card_types.index(card)] += 1
        return vector

    def vectorize_action(self, action: Action) -> int:
        if action.type == "end_phase":
            return self.action_space_size - 1
        else:  # "play" or "buy"
            card_index = self.card_types.index(action.card)
            action_offset = 0 if action.type == "play" else 1
            return card_index * 2 + action_offset

    def devectorize_action(self, action_index: int) -> Action:
        if action_index == self.action_space_size - 1:
            return Action("end_phase")
        else:
            card_index = action_index // 2
            action_type = "play" if action_index % 2 == 0 else "buy"
            return Action(action_type, self.card_types[card_index])

    def get_action_mask(self, game_state: GameState, player_index: int) -> np.ndarray:
        mask = np.zeros(self.action_space_size, dtype=np.int8)
        player = game_state.players[player_index]

        # Check if player can play action cards
        if game_state.phase == 0:  # Action phase
            for i, card in enumerate(self.card_types):
                if card in player.hand and card.is_action:
                    mask[i*2] = 1  # Can play this action card

        # Check if player can buy cards
        if game_state.phase <= 1:  # Action or Buy phase
            for i, card in enumerate(self.card_types):
                if game_state.can_buy_card(player_index, card):
                    mask[i*2 + 1] = 1  # Can buy this card

        # Player can always choose to end their phase/turn
        mask[-1] = 1

        return mask