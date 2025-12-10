import numpy as np
import torch
import gymnasium as gym
import mujoco

class RolloutBuffer:
    def __init__(self, size, obs_dim, act_dim, gamma=0.99, lam=0.95, device='cpu'):
        self.size = size
        self.gamma = gamma
        self.lam = lam
        self.device = device

        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.ptr = 0
        self.path_start_idx = 0

    def store(self, obs, act, rew, val, logp):
        assert self.ptr < self.size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        adv = 0
        for t in reversed(range(len(deltas))):
            adv = deltas[t] + self.gamma * self.lam * adv
            self.adv_buf[path_slice][t] = adv

        ret = 0
        for t in reversed(range(len(rews) - 1)):
            ret = rews[t] + self.gamma * ret
            self.ret_buf[path_slice][t] = ret

        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.size

        adv_mean = np.mean(self.adv_buf)
        adv_std = np.std(self.adv_buf) + 1e-8
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std

        data = dict(
            obs=torch.as_tensor(self.obs_buf, device=self.device),
            act=torch.as_tensor(self.act_buf, device=self.device),
            ret=torch.as_tensor(self.ret_buf, device=self.device),
            adv=torch.as_tensor(self.adv_buf, device=self.device),
            logp=torch.as_tensor(self.logp_buf, device=self.device)
        )
        self.ptr = 0
        self.path_start_idx = 0
        return data

class SwingUpEnv(gym.Wrapper):
    """COMPLETE override - no early termination from down position"""
    def __init__(self, env, force_scale=2.0, force_prob=0.1):
        super().__init__(env)
        self.model = self.env.unwrapped.model
        self.data = self.env.unwrapped.data
        self.steps_from_down = 0
        self.max_swing_steps = 2000  # Allow full swing-up

        self.force_scale = force_scale
        self.force_prob = force_prob

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # FORCE DOWN POSITION
        self.data.qpos[1] = np.pi  # Pole 1 down
        self.data.qpos[2] = np.pi  # Pole 2 down
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        obs = self.env.unwrapped._get_obs()

        self.steps_from_down = 0
        return obs, info

    def step(self, action):
        if np.random.rand() < self.force_prob:
            extra_force = np.random.uniform(-1.0, 1.0) * self.force_scale
            action = np.clip(action + extra_force, -1.0, 1.0)

        obs, rew_base, term_base, trunc_base, info = self.env.step(action)

        self.steps_from_down += 1

        term = False
        trunc = self.steps_from_down >= self.max_swing_steps

        qpos = self.data.qpos
        qvel = self.data.qvel

        upright = 12.0 * (np.cos(qpos[1]) + np.cos(qpos[2]))
        swing_vel = 0.05 * (np.abs(np.sin(qpos[1]) * qvel[1]) +
                            np.abs(np.sin(qpos[2]) * qvel[2]))
        alive_bonus = 1.0
        action_cost = 0.001 * np.sum(np.square(action))
        ang_vel_penalty = 0.15 * (qvel[1]**2 + qvel[2]**2)

        # Cart centering reward (keep cart near x=0)
        cart_center_bonus = -0.5 * (qpos[0] ** 2)  # Quadratic penalty for cart displacement

        rew = (rew_base + upright + swing_vel + alive_bonus - action_cost -
               ang_vel_penalty + cart_center_bonus)

        # Stability bonus
        if (np.cos(qpos[1]) > 0.95 and np.cos(qpos[2]) > 0.95 and
                np.abs(qvel[1]) < 1.0 and np.abs(qpos[0]) < 0.1):  # ALSO near center
            rew += 5.0

        return obs, rew, term, trunc, info


def make_robust_env(base_env, force_scale=2.0, force_prob=0.1):
    env = SwingUpEnv(base_env, force_scale, force_prob)
    return env
