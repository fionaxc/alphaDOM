class PlayerState:
    def __init__(self):
        self.actions = 0
        self.buys = 0
        self.coins = 0

        self.deck = []
        self.hand = []
        self.play_area = []
        self.discard = []
        self.temp_area = []

    def all_cards(self):
        cards = []
        for collection in [self.deck, self.hand, self.play_area, self.discard, self.temp_area]:
            cards.extend(collection)
        return cards

    def num_cards(self):
        return len(self.all_cards())

    def play_remaining_treasures(self):
        remaining_treasures = list(map(lambda card: card.name, filter(lambda card: card.is_treasure(), self.hand)))
        for card_name in remaining_treasures:
            self.handle.play(card_name)

    def victory_points(self):
        points = 0
        for card in self.all_cards():
            points += card.victory_points.invoke(self.handle, self.game, None)
        return points

    def __str__(self):
        return 'PlayerState(actions={}, buys={}, coins={}, deck={}, hand={},play_area={}, discard={}, temp_area={}'.format(
            self.actions, self.buys, self.coins, self.deck, self.hand, self.play_area, self.discard, self.temp_area)
