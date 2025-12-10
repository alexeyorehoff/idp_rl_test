import gymnasium as gym
import numpy as np
import torch
from model import Actor
from utils import make_robust_env

def make_env(render=True):
    base_env = gym.make('InvertedDoublePendulum-v5', render_mode='human')

    base_env.unwrapped.model.opt.timestep = 0.005
    base_env.unwrapped.model.opt.iterations = 50
    base_env.unwrapped.model.vis.global_.offwidth = 1200
    base_env.unwrapped.model.vis.global_.offheight = 900

    env = make_robust_env(base_env, force_scale=0.02, force_prob=0.1)
    print("Episode length limit:", env.max_swing_steps)
    return env


def test_policy(model_path='actor_doublependulum_robust.pt', episodes=10, device='cpu'):
    env = make_env(render=True)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    actor = Actor(obs_dim, act_dim, hidden_sizes=[256, 128]).to(device)
    actor.load_state_dict(torch.load(model_path, map_location=device))
    actor.eval()

    returns = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        ep_ret, ep_len = 0, 0
        print(f"\nEpisode {ep+1}")

        while True:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                # Test both stochastic and deterministic
                if ep % 2 == 0:  # Stochastic
                    mu, std = actor(obs_t)
                    dist = torch.distributions.Normal(mu, std)
                    action = dist.sample()
                else:  # Deterministic
                    mu, _ = actor(obs_t)
                    action = mu

                action = action.cpu().numpy().flatten()

            obs, rew, term, trunc, _ = env.step(action)
            ep_ret += rew
            ep_len += 1

            if term or trunc or ep_len > 2000:  # Safety limit
                break

        returns.append(ep_ret)
        print(f"  Length: {ep_len:4d}, Return: {ep_ret:6.1f}")

    print(f"\nFinal Stats - Mean: {np.mean(returns):6.1f}, Std: {np.std(returns):5.1f}")
    env.close()

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Testing double pendulum on device: {device}")
    test_policy(episodes=5, device=device)
