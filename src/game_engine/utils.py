from collections import Counter

# Convert a list of cards to a dictionary of card names and counts
def cards_to_dict(cards):
    # Create a dictionary with card names as keys and their counts as values
    return dict(Counter(card.name for card in cards))