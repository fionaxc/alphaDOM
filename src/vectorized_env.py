# src/vectorized_env.py
import torch
from game_engine.game import Game
from vectorization.vectorizer import DominionVectorizer

class VectorizedDominionEnv:
    """
    Vectorized environment for Dominion for running many game environements in parallel.
    """
    def __init__(self, num_envs, game_engine, vectorizer, device):
        self.num_envs = num_envs
        self.game_engine = game_engine
        self.vectorizer = vectorizer
        self.device = device
        self.envs = [Game() for _ in range(num_envs)]
        self.current_player_turns = torch.zeros(num_envs, dtype=torch.long, device=device)

    def reset(self):
        for env in self.envs:
            env.start_game()
        obs = self._get_observations()
        return obs

    def step(self, actions):
        rewards = torch.zeros(self.num_envs, device=self.device)
        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            action_obj = self.vectorizer.devectorize_action(action.item(), env.current_player())
            action_obj.apply()
            
            rewards[i] = env.current_player().calculate_reward(env, action_obj)
            dones[i] = env.game_over
            
            if env.game_over:
                env.start_game()
            else:
                env.end_turn()

        obs = self._get_observations()
        return obs, rewards, dones

    def _get_observations(self):
        obs = torch.stack([torch.FloatTensor(self.vectorizer.vectorize_observation(env)) for env in self.envs])
        return obs.to(self.device)

    def get_action_masks(self):
        masks = torch.stack([torch.BoolTensor(self.vectorizer.get_action_mask(env)) for env in self.envs])
        return masks.to(self.device)