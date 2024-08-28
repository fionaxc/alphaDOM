# Convert a list of cards to a dictionary of card names and counts
def cards_to_dict(cards):
    return {card.name: cards.count(card) for card in set(cards)}