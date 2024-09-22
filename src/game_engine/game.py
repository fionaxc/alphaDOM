import random
from game_engine.cards.card_instances import CARD_MAP, SUPPLY_CARD_LIMITS
from game_engine.action import ActionType
from .player import PlayerState
from .phase import Phase

DEFAULT_SUPPLY = ["Copper", "Silver", "Gold", "Estate", "Duchy", "Province", "Curse"]
# SIMPLE_SETUP = ["Village", "Smithy", "Moneylender"]
# SIMPLE_SETUP_2 = ["Village", "Smithy", "Moneylender", "Festival", "Laboratory", "Market"]
SIMPLE_SETUP = ["Village", "Smithy", "Moneylender", "Festival", "Laboratory", "Market", "Witch"]
# SIMPLE_CHAPEL_SETUP = ["Chapel", "Village", "Smithy", "Moneylender", "Festival", "Laboratory", "Market", "Witch"]

class Game:
    def __init__(self, kingdom_cards=SIMPLE_SETUP, num_players=2):
        """
        Initialize the game with the given kingdom cards and number of players.

        Args:
            kingdom_cards (list): List of kingdom card names to include in the game.
            num_players (int): Number of players in the game.
        """
        # Game setup
        self.all_card_names = DEFAULT_SUPPLY + kingdom_cards
        # Supply piles maps "card names" to their maximum count
        self.supply_piles = {card: SUPPLY_CARD_LIMITS[card] for card in DEFAULT_SUPPLY + kingdom_cards}
        self.players = [PlayerState(f"Player {i+1}", self) for i in range(num_players)]
        self.current_player_turn = 0
        self.current_phase = Phase.ACTION
        self.game_over = False
        self.turn_number = [0 for _ in range(num_players)] # Start with 0 for each player

        # Start game
        self.start_game()
    
    def get_observation_state(self):
        """
        Get the observation state for the current player. Example:

        {
            'current_player_state': {
                'victory_points': 1,
                'actions': 1,
                'buys': 1,
                'coins': 0,
                'discard_pile_count': {
                    'Copper': 10,
                    'Curse': 6,
                    ...
                },
                'draw_pile_count': {
                    'Copper': 8,
                    'Estate': 1,
                    ...
                },
                'hand': [Moneylender, Curse, Estate, Gold, Copper],
                'played_cards': []
            },
            'game_state': {
                'current_player_name': 'Player 1',
                'turn_number': [1, 1],
                'current_phase': 'action',
                'game_over': True,
                'card_costs': {
                    'Copper': 0,
                    'Curse': 0,
                    ...
                },
                'supply_piles': {
                    'Copper': 20,
                    'Curse': 0,
                    ...
                },
                'game_over': False,
            },
            'opponent_state': {
                'deck_count': {
                    'Copper': 19,
                    'Curse': 3,
                    ...
                },
                'victory_points': 1,
            }
        }

        Returns:
            dict: A dictionary containing the current player's observation state and the game state.
        """
        observation_state = {
            **self.current_player().get_player_observation_state(),
            'game_state': {
                'current_player_name': self.current_player().name,
                'turn_number': self.turn_number,
                'current_phase': self.current_phase.value,
                'supply_piles': self.supply_piles,
                'card_costs': {card: CARD_MAP[card].cost for card in self.supply_piles.keys()},
                'game_over': self.game_over,
            }
        }
        return observation_state
    
    def get_valid_actions(self):
        """
        Get the valid actions for the current player. Example:

        [
            Action(player=Player 1, type=ActionType.PLAY, card=Smithy),
            Action(player=Player 1, type=ActionType.PLAY, card=Laboratory),
            Action(player=Player 1, type=ActionType.END_ACTION, card=None)
        ]

        Returns:
            list: A list of valid actions that the current player can perform.
        """
        return self.current_player().get_valid_actions()
    
    def get_other_player(self):
        return self.players[(self.current_player_turn + 1) % len(self.players)]
    
    def get_maximum_possible_vp(self):
        """
        Get the maximum possible VPs for the game given the kingdom cards.
        """
        # Create a new player with a deck size of 60 for Gardens
        max_vp_player = PlayerState("Max VP Player", self)
        max_vp_player.draw_pile = [CARD_MAP["Copper"]] * 60

        max_victory_points = 0
        for card_name in SUPPLY_CARD_LIMITS.keys():
            if card_name in self.all_card_names:
                card = CARD_MAP[card_name]
                if card.get_victory_points(max_vp_player) > 0:
                    max_victory_points += card.get_victory_points(max_vp_player) * SUPPLY_CARD_LIMITS[card_name]
        return max_victory_points
    
    def start_game(self):
        # Randomly select player to go first
        self.current_player_turn = random.randint(0, len(self.players) - 1)
        self.turn_number[self.current_player_turn] += 1

        # Initialize each player with 7 coppers, 3 estates and then draw 5 cards
        for player in self.players:
            # Remove 7 coppers and 3 estates from supply
            self.supply_piles["Copper"] -= 7
            self.supply_piles["Estate"] -= 3

            player.discard_pile = [CARD_MAP["Copper"]] * 7 + [CARD_MAP["Estate"]] * 3
            player.cleanup_cards()
    
    def next_phase(self):
        # Update the current phase to the next phase in the enum
        phases = list(Phase)
        current_index = phases.index(self.current_phase)
        self.current_phase = phases[(current_index + 1) % len(phases)]

        # Check if the current phase is the action phase
        if self.current_phase == Phase.ACTION:
            # Get the current player's valid actions
            valid_actions = self.current_player().get_valid_actions()
            
            # If the only valid action is to end the action phase, then do that
            if len(valid_actions) == 1 and valid_actions[0].action_type == ActionType.END_ACTION:
                self.current_player().end_action_phase()

    def next_player(self):
        # Before moving to next player, check if the game is over (no provinces, or at least 3 empty supply piles)
        if self.supply_piles["Province"] <= 0 or len([card for card, count in self.supply_piles.items() if count == 0]) >= 3:
            self.game_over = True
            return

        self.current_player_turn = (self.current_player_turn + 1) % len(self.players)
        self.turn_number[self.current_player_turn] += 1

    def current_player(self):
        return self.players[self.current_player_turn]
    
    def winner(self):   
        """
        Return the winner of the game if there is one, otherwise return None. In case of a tie, return None.
        """
        if not self.game_over:
            return None
        
        # Find the player(s) with the most VPs
        max_vp = max(player.victory_points() for player in self.players)
        candidates = [player for player in self.players if player.victory_points() == max_vp]
        
        # If there's only one candidate, they are the winner
        if len(candidates) == 1:
            return candidates[0]
        
        # Find the player(s) with the fewest turns among the candidates
        fewest_turns = min(self.turn_number[self.players.index(player)] for player in candidates)
        fewest_turns_candidates = [player for player in candidates if self.turn_number[self.players.index(player)] == fewest_turns]
        
        # If there's only one candidate with the fewest turns, they are the winner
        if len(fewest_turns_candidates) == 1:
            return fewest_turns_candidates[0]
        
        # If there's still a tie, return None
        return None
        
    def get_game_state_string(self):
        state = f"\n{'='*20} Turn {self.turn_number}, Phase: {self.current_phase.value} {'='*20}\n"
        state += f"Current Player: {self.current_player().name}\n"
        state += "Supply Piles:\n"
        for card, count in self.supply_piles.items():
            state += f"  {card}: {count}\n"
        state += "\nPlayers:\n"
        for player in self.players:
            state += f"  {player.name}:\n"
            state += f"    Actions: {player.actions}, Buys: {player.buys}, Coins: {player.coins}\n"
            state += f"    Victory Points: {player.victory_points()}\n"  # Added victory points
            state += f"    Hand: {', '.join(str(card) for card in player.hand)}\n"
            state += f"    Played Hand: {', '.join(str(card) for card in player.played_cards)}\n"  # Added played hand
            state += f"    Deck: {len(player.draw_pile)} cards ({', '.join(str(card) for card in player.draw_pile)})\n"
            state += f"    Discard: {len(player.discard_pile)} cards ({', '.join(str(card) for card in player.discard_pile)})\n"
        return state