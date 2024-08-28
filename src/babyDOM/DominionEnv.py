import gym
from gym import spaces
import numpy as np
from typing import List, Tuple, Dict, Any

# Assume we have these imports from our game engine
from dominion_engine import DominionGameEngine, Card, Player, GameState, Action

class CardType(Enum):
    FESTIVAL = 0
    VILLAGE = 1
    SMITHY = 2
    MARKET = 3
    MONEYLENDER = 4
    WITCH = 5
    LAB = 6
    CHAPEL = 7
    COPPER = 8
    SILVER = 9
    GOLD = 10
    ESTATE = 11
    DUCHY = 12
    PROVINCE = 13
    CURSE = 14

class DominionEnv(gym.Env):
    def __init__(self, cards: List[Card]):
        super().__init__()

        self.card_types = [card for card in CardType]
        self.action_cards = self.card_types[:8]  # First 8 are action cards
        self.game_engine = DominionGameEngine(self.action_cards, num_players=2)
        
        self.max_card_counts = [
            10, # Festival
            10, # Village
            10, # Smithy
            10, # Market
            10, # Moneylender
            10, # Witch
            10, # Lab
            10, # Chapel
            53, # Copper
            40, # Silver
            30, # Gold
            8, # Estate
            8, # Duchy
            8, # Province
            10, # Curse
        ]


        # Define the observation space
        self.observation_space = spaces.Dict({
            'current_player': spaces.Discrete(2),
            'current_phase': spaces.Discrete(3), # Action, Buy, Cleanup
            'player_hand': spaces.MultiDiscrete(self.max_card_counts),
            'player_deck': spaces.MultiDiscrete(self.max_card_counts),
            'player_discard': spaces.MultiDiscrete(self.max_card_counts),
            'player_vp': spaces.Discrete(100), # Victory points
            'player_money': spaces.Discrete(100), # Money
            'player_buys': spaces.Discrete(10), # Number of buys
            'player_actions': spaces.Discrete(10), # Number of actions

            'opponent_deck': spaces.MultiDiscrete(self.max_card_counts),
            'opponent_discard': spaces.MultiDiscrete(self.max_card_counts),
            'opponent_vp': spaces.Discrete(100), # Victory points
            'opponent_money': spaces.Discrete(100), # Money

            'supply': spaces.MultiDiscrete(self.max_card_counts),
            'trash': spaces.MultiDiscrete(self.max_card_counts),

        })

        # Define the action space
        self.action_space = spaces.Discrete(len(self.action_cards)*2 + 1)

        self.current_player_index = 0

    def reset(self) -> Dict[str, Any]:
        self.game_engine.reset()
        self.current_player_index = 0
        return self._get_observation()

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        # Add action masking
        action_mask = self.get_action_mask()
        if action_mask[action] == 0:
            return self._get_observation(), -1, False, {"invalid_action": True}

        # Rest of the step function remains the same
        action = self._action_index_to_action(action)
        reward = self.game_engine.take_action(self.current_player_index, action)

        if self.game_engine.is_turn_end():
            self.current_player_index = 1 - self.current_player_index # Switch player
            self.game_engine.start_turn(self.current_player_index)

        observation = self._get_observation()
        done = self.game_engine.is_game_over()
        info = {}

        if done:
            info['final_scores'] = self._calculate_scores()

        return observation, reward, done, info

    def get_action_mask(self) -> np.ndarray:
        game_state = self.game_engine.get_game_state()
        current_player = game_state.players[self.current_player_index]
        
        mask = np.zeros(self.action_space.n, dtype=np.int8)
        
        # Check if player can play action cards
        if game_state.phase == 0:  # Action phase
            for i, card in enumerate(self.action_cards):
                if card in current_player.hand:
                    mask[i*2] = 1  # Can play this action card
        
        # Check if player can buy cards
        if game_state.phase <= 1:  # Action or Buy phase
            for i, card in enumerate(self.card_types):
                if self.game_engine.can_buy_card(self.current_player_index, card):
                    mask[i*2 + 1] = 1  # Can buy this card
        
        # Player can always choose to end their phase/turn
        mask[-1] = 1
        
        return mask
    
    def _get_observation(self) -> Dict[str, Any]:
        game_state = self.game_engine.get_game_state()
        current_player = game_state.players[self.current_player_index]
        opponent = game_state.players[1 - self.current_player_index]

        return {
            'current_player': self.current_player_index,
            'current_phase': game_state.phase,
            'player_hand': self._get_card_counts(current_player.hand),
            'player_deck': self._get_card_counts(current_player.deck),
            'player_discard': self._get_card_counts(current_player.discard),
            'player_vp': current_player.vp,
            'player_money': current_player.money,
            'player_buys': current_player.buys,
            'player_actions': current_player.actions,

            'opponent_deck': self._get_card_counts(opponent.deck),
            'opponent_discard': self._get_card_counts(opponent.discard),
            'opponent_vp': opponent.vp,
            'opponent_money': opponent.money,

            'supply': self._get_card_counts(game_state.supply),
            'trash': self._get_card_counts(game_state.trash),
        }
    
    def _get_card_counts(self, cards: List[Card]) -> List[int]:
        card_counts = [0] * len(self.max_card_counts)
        for card in cards:
            card_counts[self.card_types.index(card)] += 1
        return card_counts

    def _action_index_to_action(self, action_index: int) -> Action:
        if action_index == len(self.action_cards)*2:
            return Action("end_phase")
        else:
            card = self.action_cards[action_index // 2]
            action_type = "play" if action_index % 2 == 0 else "buy"
            return Action(action_type, card)

    def _calculate_scores(self) -> Dict[int, int]:
        scores = {}
        for i, player in enumerate(self.game_engine.players):
            scores[i] = player.calculate_score()
        return scores
    
    def render(self, mode='human'):
        print(self.game_engine.get_game_state_string())

    

    
