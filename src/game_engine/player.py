import random

class PlayerState:
    def __init__(self, name):
        self.name = name

        # Game state
        self.actions = 0
        self.buys = 0
        self.coins = 0

        # Deck information
        self.draw_pile = []
        self.hand = []
        self.played_cards = []
        self.discard_pile = []

    def all_cards(self):
        return self.draw_pile + self.hand + self.played_cards + self.discard_pile

    def num_cards(self):
        return len(self.all_cards())
    
    def victory_points(self):
        return sum(card.victory_points for card in self.all_cards())

    def play_remaining_treasures(self):
        # Filter out treasure cards from hand and play them
        for card in filter(lambda card: card.is_treasure(), self.hand):
            self.handle.play(card.name)
    
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

    def __str__(self):
        return 'PlayerState(actions={}, buys={}, coins={}, draw_pile={}, hand={}, played_cards={}, discard_pile={})'.format(
            self.actions, self.buys, self.coins, self.draw_pile, self.hand, self.played_cards, self.discard_pile)
