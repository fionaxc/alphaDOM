import torch
from babyDOM.ppo import PPOActor, PPOCritic, PPOAgent

def main():
    model_path = "src/output/0912_SIMPLE2_run2_games250000_batchsize25_updateepochs5_hidden256/checkpoints/checkpoint_game_100.pth"
    checkpoint = torch.load(model_path)
    actor_state_dict = checkpoint['actor']
    critic_state_dict = checkpoint['critic']
    lr = checkpoint['lr']
    gamma = checkpoint['gamma']
    epsilon = checkpoint['epsilon']
    value_coef = checkpoint['value_coef']
    entropy_coef = checkpoint['entropy_coef']
    gae_lambda = checkpoint['gae_lambda']

    print("checkpoint loaded")
    print("lr:", lr)
    print("gamma:", gamma)
    print("epsilon:", epsilon)
    print("value_coef:", value_coef)
    print("entropy_coef:", entropy_coef)
    print("gae_lambda:", gae_lambda)

    print("actor_state_dict:", actor_state_dict)
    print("critic_state_dict:", critic_state_dict)

if __name__ == "__main__":
    main()
