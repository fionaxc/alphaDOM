# This module defines various effects that can be applied to players and the game state.

class Effect:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def apply(self, player, game):
        raise NotImplementedError("This method should be overridden by subclasses")

class DrawCardsEffect(Effect):
    def apply(self, player, game):
        # Draw num_cards from the deck
        for _ in range(self.num_cards):
            player.draw_card()
        
class AddActionsEffect(Effect):
    def apply(self, player, game):
        # Add num_actions to the player's action count
        player.actions += self.num_actions

class AddBuysEffect(Effect):
    def apply(self, player, game):
        # Add num_buys to the player's buy count
        player.buys += self.num_buys

class AddMoneyEffect(Effect):
    def apply(self, player, game):
        # Add amount to the player's money
        player.coins += self.amount

class AddCurseEffect(Effect):
    def apply(self, player, game):
        # Import CARD_MAP here to avoid circular dependency
        from game_engine.cards.card_instances import CARD_MAP

        # Add minimum of num_curses or remaining curses in supply to other player's deck
        if game.supply_piles["Curse"] > 0:
            num_curses_to_add = min(self.num_curses, game.supply_piles["Curse"])
            game.get_other_player().discard_pile.extend([CARD_MAP["Curse"]] * num_curses_to_add)
            game.supply_piles["Curse"] -= num_curses_to_add  # Update the supply pile

class TrashCardInHandEffect(Effect):
    def apply(self, player, game):
        # Remove the first card that's equal to 'card' from player's hand
        if self.card in player.hand:
            player.hand.remove(self.card)

# class TrashCardsEffect(Effect):
#     def apply(self, player, game):
#         # Allow player to select num_cards to trash
#         trashed_cards = player.select_cards_to_trash(self.num_cards)
#         for card in trashed_cards:
#             player.hand.remove(card)
#             game.trash_pile.append(card)

class CompositeEffect(Effect):
    def apply(self, player, game):
        # Apply all effects in the composite
        for effect in self.effects:
            effect.apply(player, game)

class ConditionalEffect(Effect):
    def __init__(self, condition, effect):
        self.condition = condition
        self.effect = effect

    def apply(self, player, game):
        # Check the condition and apply the effect if true
        if self.condition(player, game):
            self.effect.apply(player, game)