import random
from .phase import Phase
from .utils import cards_to_dict

class PlayerState:
    def __init__(self, name, game):
        self.name = name
        self.game = game

        # Turn state
        self.actions = 0
        self.buys = 0
        self.coins = 0

        # Deck information
        self.draw_pile = []
        self.hand = []
        self.played_cards = []
        self.discard_pile = []
    
    def get_observation_state(self):
        # Get the current player's state
        current_player_state = {
            'actions': self.actions,
            'buys': self.buys,
            'coins': self.coins,
            'hand': self.hand,
            'played_cards': self.played_cards,
            # For rest of deck information, I should only know the count of each card
            'draw_pile_count': cards_to_dict(self.draw_pile),
            'discard_pile_count': cards_to_dict(self.discard_pile),
        }

        # Get the opponent's state
        opponent = self.game.get_other_player()
        opponent_state = {
            # I should only know the count of each card in his entire deck and the discard pile
            'deck_count': cards_to_dict(opponent.all_cards()),
            'discard_pile_count': cards_to_dict(opponent.discard_pile),
            # TODO: Figure out if we want to include # of cards in hand or draw pile
        }

        # Get the game state
        game_state = {
            'current_phase': self.game.current_phase.value,
            'supply_piles': self.game.supply_piles,
            'game_over': self.game.game_over,
        }

        return {
            'current_player_state': current_player_state,
            'opponent_state': opponent_state,
            'game_state': game_state
        }

    def all_cards(self):
        return self.draw_pile + self.hand + self.played_cards + self.discard_pile
    
    def victory_points(self):
        return sum(card.victory_points for card in self.all_cards())

    def play_remaining_treasures(self):
        # Filter out treasure cards from hand and play them
        for card in filter(lambda card: card.is_treasure(), self.hand):
            self.handle.play(card.name)
    
    def play_card(self, card):
        # Check if the card is in hand  
        if card in self.hand:
            # Remove card from hand
            self.hand.remove(card)

            # Add card to played_cards
            self.played_cards.append(card)

            # Play the card
            card.play(self, self.game)
    
    def draw_card(self):
        # Check if draw_pile is empty
        if not self.draw_pile:
            # If draw_pile is empty, shuffle discard_pile and set it as draw_pile
            self.draw_pile = self.discard_pile[:]
            random.shuffle(self.draw_pile)
            self.discard_pile.clear()
        
        # Draw the top card from draw_pile and add it to hand
        if self.draw_pile:
            self.hand.append(self.draw_pile.pop())
    
    def buy_card(self, card):
        # Can only buy if in the buy phase
        if self.game.current_phase != Phase.BUY:
            return

        # Check if the card is in supply
        if card.name in self.game.supply_piles and self.game.supply_piles[card.name] > 0:
            # Check if the player has enough coins and buys to buy the card
            if self.coins >= card.cost and self.buys > 0:
                # Buy the card
                self.coins -= card.cost
                self.buys -= 1
                self.game.supply_piles[card.name] -= 1
                self.discard_pile.append(card)
    
    def cleanup_cards(self):
        self.actions = 1
        self.buys = 1
        self.coins = 0

        # Move all cards from hand to discard pile
        self.discard_pile.extend(self.hand)
        self.hand.clear()

        # Move all cards from play area to discard pile
        self.discard_pile.extend(self.played_cards)
        self.played_cards.clear()

        # Draw 5 cards
        for _ in range(5):
            self.draw_card()
    
    def end_action_phase(self):
        if self.game.current_phase == Phase.ACTION:
            self.game.next_phase()
    
    def end_buy_phase(self):
        if self.game.current_phase == Phase.BUY:
            # Move to cleanup phase, clean up cards, and then go to next player with reset turn state
            self.game.next_phase()
            self.cleanup_cards()
            self.game.next_phase()
            self.game.next_player()

    def __str__(self):
        return 'PlayerState(actions={}, buys={}, coins={}, draw_pile={}, hand={}, played_cards={}, discard_pile={})'.format(
            self.actions, self.buys, self.coins, self.draw_pile, self.hand, self.played_cards, self.discard_pile)
