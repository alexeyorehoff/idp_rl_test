import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model import Actor, Critic
from utils import RolloutBuffer, make_robust_env
import torch.nn.functional as F
import os

def make_env(render=False, force_scale=0.02, force_prob=0.025):
    render_mode = 'human' if render else None
    base_env = gym.make('InvertedDoublePendulum-v5', render_mode=render_mode,
                        healthy_reward=10.0, reset_noise_scale=0.1)
    env = make_robust_env(base_env, force_scale=force_scale, force_prob=force_prob)
    return env

def load_pretrained_models(obs_dim, act_dim, device, actor_path='actor_doublependulum.pt',
                           critic_path='critic_doublependulum.pt'):
    """Load existing models or create new ones"""
    actor = Actor(obs_dim, act_dim, hidden_sizes=[256, 128]).to(device)
    critic = Critic(obs_dim, hidden_sizes=[256, 128]).to(device)

    # Try loading pre-trained weights
    if os.path.exists(actor_path):
        actor.load_state_dict(torch.load(actor_path, map_location=device))
        print(f"Loaded pre-trained actor from {actor_path}")
    else:
        print(f"No pre-trained actor found at {actor_path}, starting fresh")

    if os.path.exists(critic_path):
        critic.load_state_dict(torch.load(critic_path, map_location=device))
        print(f"Loaded pre-trained critic from {critic_path}")
    else:
        print(f"No pre-trained critic found at {critic_path}, starting fresh")

    return actor, critic

def ppo_train(total_steps=500000, device='cpu', load_pretrained=True, force_prob_start=0.1,
              force_prob_end=0.1):
    env = make_env(render=False, force_prob=force_prob_start)
    obs, _ = env.reset(seed=42)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    print(f"Double Pendulum: obs_dim={obs_dim}, act_dim={act_dim}")

    # Load pre-trained OR initialize new
    actor, critic = load_pretrained_models(obs_dim, act_dim, device)

    # Lower learning rates for fine-tuning
    pi_optimizer = optim.Adam(actor.parameters(), lr=1e-4)
    vf_optimizer = optim.Adam(critic.parameters(), lr=3e-4)

    buf = RolloutBuffer(size=2048, obs_dim=obs_dim, act_dim=act_dim, device=device)
    ep_returns = []
    steps = 0

    while steps < total_steps:
        # Curriculum: gradually increase force_prob
        current_force_prob = force_prob_start + (force_prob_end - force_prob_start) * (steps / total_steps)
        env.force_prob = current_force_prob  # Dynamic adjustment

        buf_start_ptr = buf.ptr
        episode_count = 0

        while buf.ptr < buf.size and episode_count < 10:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                mu, std = actor(obs_t)
                dist = torch.distributions.Normal(mu, std)
                act = dist.sample()
                logp = dist.log_prob(act).sum(-1)
                val = critic(obs_t)

            act_np = act.cpu().numpy().flatten()
            next_obs, rew, term, trunc, _ = env.step(act_np)
            done = term or trunc

            buf.store(obs, act_np, rew, val.item(), logp.item())
            obs = next_obs
            steps += 1

            if done or buf.ptr >= buf.size:
                last_val = 0.0
                buf.finish_path(last_val)

                path_len = buf.ptr - buf.path_start_idx
                ep_return = np.sum(buf.rew_buf[buf_start_ptr:buf.ptr])
                ep_returns.append(ep_return)
                episode_count += 1

                print(f"Ep {len(ep_returns)}: len={path_len}, ret={ep_return:.1f}, "
                      f"force_p={current_force_prob:.3f}")

                if buf.ptr < buf.size:
                    obs, _ = env.reset(seed=steps)

        if buf.ptr >= buf.size:
            print("=== PPO Update ===")
            data = buf.get()

            for epoch in range(8):
                idx = torch.randperm(data['obs'].shape[0], device=device)
                for start in range(0, data['obs'].shape[0], 256):
                    end = min(start + 256, data['obs'].shape[0])
                    mb_idx = idx[start:end]

                    # Policy loss
                    obs_b = data['obs'][mb_idx]
                    act_b = data['act'][mb_idx]
                    adv_b = data['adv'][mb_idx]
                    logp_old_b = data['logp'][mb_idx]

                    dist = actor.distribution(obs_b)
                    logp = dist.log_prob(act_b).sum(-1)
                    ratio = torch.exp(logp - logp_old_b)
                    pi_loss = -torch.min(ratio * adv_b, torch.clamp(ratio, 0.8, 1.2) * adv_b).mean()
                    entropy = dist.entropy().mean()
                    pi_loss -= 0.02 * entropy

                    pi_optimizer.zero_grad()
                    pi_loss.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                    pi_optimizer.step()

                    # Value loss
                    v = critic(obs_b).squeeze(-1)
                    vf_loss = F.mse_loss(v, data['ret'][mb_idx])

                    vf_optimizer.zero_grad()
                    vf_loss.backward()
                    nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                    vf_optimizer.step()

            print(f"Update complete. force_prob={current_force_prob:.3f}")

        if len(ep_returns) >= 10:
            mean_ret = np.mean(ep_returns[-10:])
            print(f"Steps: {steps:06d}, Mean Return: {mean_ret:6.1f} (last 10 eps), "
                  f"force_p={current_force_prob:.3f}")

    # Save robust model
    torch.save(actor.state_dict(), 'actor_doublependulum_robust.pt')
    torch.save(critic.state_dict(), 'critic_doublependulum_robust.pt')
    print("Robust training complete! Models saved as *_robust.pt")
    env.close()

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    ppo_train(total_steps=200000, device=device, load_pretrained=True,
              force_prob_start=0.05, force_prob_end=0.20)
