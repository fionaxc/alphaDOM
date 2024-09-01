import random
from .phase import Phase
from .utils import cards_to_dict
from .action import Action, ActionType
from game_engine.cards.card_instances import CARD_MAP

class PlayerState:
    def __init__(self, name, game):
        """
        Initialize the state of a player.

        Args:
            name (str): The name of the player.
            game (Game): The game instance the player is part of.
        """
        self.name = name
        self.game = game

        # Turn state - number of actions, buys, and coins they have
        self.actions = 0
        self.buys = 0
        self.coins = 0

        # Deck information
        self.draw_pile = []
        self.hand = []
        self.played_cards = []
        self.discard_pile = []
    
    def get_player_observation_state(self):
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
            # I should only know the count of each card in his entire deck
            'deck_count': cards_to_dict(opponent.all_cards()),
            # TODO: Figure out if we want to include # of cards in hand or draw pile
            # TODO: Eventually, add information about some of the known cards in the discard pile
        }

        return {
            'current_player_name': self.name,
            'current_player_state': current_player_state,
            'opponent_state': opponent_state,
        }
    
    def get_valid_actions(self):
        # Initialize list of valid actions
        valid_actions = []

        # If the game is over, return empty list
        if self.game.game_over:
            return []

        # If in action phase, add playable action cards and end action phase action
        if self.game.current_phase == Phase.ACTION:
            valid_actions.extend([Action(self, ActionType.PLAY, card) for card in self.hand if card.is_action()])
            valid_actions.append(Action(self, ActionType.END_ACTION))

        # If in buy phase, add buyable cards and end buy phase action
        elif self.game.current_phase == Phase.BUY:
            valid_actions.extend([Action(self, ActionType.BUY, CARD_MAP[card]) for card in self.game.supply_piles if self.coins >= CARD_MAP[card].cost and self.buys > 0 and self.game.supply_piles[card] > 0])
            valid_actions.append(Action(self, ActionType.END_BUY))

        return valid_actions

    def all_cards(self):
        return self.draw_pile + self.hand + self.played_cards + self.discard_pile
    
    def victory_points(self):
        return sum(card.victory_points for card in self.all_cards())

    def play_remaining_treasures(self):
        # Play all treasure cards in hand
        for card in [card for card in self.hand if card.is_treasure()]:
            self.play_card(card)
    
    def play_card(self, card):
        # Check if the card is in hand  
        if card in self.hand:
            # Remove card from hand
            self.hand.remove(card)

            # Add card to played_cards
            self.played_cards.append(card)

            # Play the card
            self.actions -= 1
            card.play(self, self.game)

            # If we're in the action phase and no actions left, end action phase
            if self.game.current_phase == Phase.ACTION and self.actions <= 0:
                self.end_action_phase()
    
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

                # Move to cleanup phase if no more buys
                if self.buys <= 0:
                    self.end_buy_phase()
    
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
            # Play remaining treasures and reset actions
            self.play_remaining_treasures()
            self.actions = 0
    
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

    def __eq__(self, other):
        return self.name == other.name and self.game == other.game
    