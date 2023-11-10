import argparse
import time
import gym
import torch
import numpy as np
from itertools import count
from os.path import join

import logging
import wandb
import dill as pickle 

import os
import os.path as osp
import json

from sac.replay_memory import ReplayMemory
from sac.sac import SAC
from dwm.a2c import ActorCritic
from model import EnsembleDynamicsModel
from predict_env import PredictEnv
from sample_env import EnvSampler
from env import create_env
from utils import compute_traj_errors

def readParser():
    parser = argparse.ArgumentParser(description='MBPO')
    parser.add_argument('--env_name', default="Hopper-v3",
                        help='Mujoco Gym environment (default: hopper_hop)')
    parser.add_argument('--suite', default="gym")
    parser.add_argument('--group', default="default")
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                        help='random seed (default: 123456)')
    parser.add_argument('--load_path', default="logs"),
    parser.add_argument('--load_step', type=int, default=0),
    parser.add_argument('--use_decay', type=bool, default=True, metavar='G',
                        help='discount factor for reward (default: 0.99)')

    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                        help='Automaically adjust α (default: False)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')

    parser.add_argument('--num_networks', type=int, default=7, metavar='E',
                        help='ensemble size (default: 7)')
    parser.add_argument('--num_elites', type=int, default=5, metavar='E',
                        help='elite size (default: 5)')
    parser.add_argument('--pred_hidden_size', type=int, default=200, metavar='E',
                        help='hidden size for predictive model')
    parser.add_argument('--reward_size', type=int, default=1, metavar='E',
                        help='environment reward size')

    parser.add_argument('--replay_size', type=int, default=2000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')

    parser.add_argument('--model_retain_epochs', type=int, default=1, metavar='A',
                        help='retain epochs')
    parser.add_argument('--model_train_freq', type=int, default=250, metavar='A',
                        help='frequency of training')
    parser.add_argument('--rollout_batch_size', type=int, default=500, metavar='A',
                        help='rollout number M')
    parser.add_argument('--epoch_length', type=int, default=20000, metavar='A',
                        help='steps per epoch')
    parser.add_argument('--rollout_min_epoch', type=int, default=20, metavar='A',
                        help='rollout min epoch')
    parser.add_argument('--rollout_max_epoch', type=int, default=150, metavar='A',
                        help='rollout max epoch')
    parser.add_argument('--rollout_length', type=int, default=100, metavar='A',
                        help='rollout length')
    parser.add_argument('--num_epoch', type=int, default=50, metavar='A',
                        help='total number of epochs')
    parser.add_argument('--min_pool_size', type=int, default=1000, metavar='A',
                        help='minimum pool size')
    parser.add_argument('--real_ratio', type=float, default=0.05, metavar='A',
                        help='ratio of env samples / model samples')
    parser.add_argument('--train_every_n_steps', type=int, default=1, metavar='A',
                        help='frequency of training policy')
    parser.add_argument('--num_train_repeat', type=int, default=20, metavar='A',
                        help='times to training policy per step')
    parser.add_argument('--max_train_repeat_per_step', type=int, default=5, metavar='A',
                        help='max training times per step')
    parser.add_argument('--policy_train_batch_size', type=int, default=256, metavar='A',
                        help='batch size for training policy')
    parser.add_argument('--init_exploration_steps', type=int, default=5000, metavar='A',
                        help='exploration steps initially')
    parser.add_argument('--max_path_length', type=int, default=1000, metavar='A',
                        help='max length of path')
    parser.add_argument('--model_type', default='pytorch', metavar='A',
                        help='predict model -- pytorch or tensorflow')

    parser.add_argument('--cuda', default=True, action="store_true",
                        help='run on CUDA (default: True)')
    return parser.parse_args()


def train(args, env_sampler, predict_env, a2c, env_pool, model_pool, env):
    total_step = 0
    reward_sum = 0
    rollout_length = 1

    for epoch_step in range(args.num_epoch):
        train_predict_model(args, env_pool, predict_env, num_steps=args.epoch_length)
        rollout_states, rollout_actions, rollout_rewards, init_sim_state = rollout_model(args, predict_env, a2c, model_pool, env_pool, args.rollout_length)
        metrics = compute_traj_errors(env, rollout_states, rollout_actions, rollout_rewards, init_sim_state)
        wandb.log(metrics, step=(epoch_step + 1) * args.epoch_length)
        


def exploration_before_start(args, env_sampler, env_pool, agent):
    for i in range(args.init_exploration_steps):
        cur_state, action, next_state, reward, done, info = env_sampler.sample(agent)
        env_pool.push(cur_state, action, reward, next_state, done, info["sim_state"])


def set_rollout_length(args, epoch_step):
    rollout_length = (min(max(args.rollout_min_length + (epoch_step - args.rollout_min_epoch)
                              / (args.rollout_max_epoch - args.rollout_min_epoch) * (args.rollout_max_length - args.rollout_min_length),
                              args.rollout_min_length), args.rollout_max_length))
    return int(rollout_length)


def train_predict_model(args, env_pool, predict_env, num_steps=1000):
    # Get all samples from environment
    state, action, reward, next_state, done, _ = env_pool.sample(len(env_pool))
    delta_state = next_state - state
    inputs = np.concatenate((state, action), axis=-1)
    labels = np.concatenate((np.reshape(reward, (reward.shape[0], -1)), delta_state), axis=-1)
    predict_env.model.train(inputs, labels, batch_size=256, holdout_ratio=0.1, max_grad_updates=num_steps, max_epochs_since_update=10000)


def resize_model_pool(args, rollout_length, model_pool):
    rollouts_per_epoch = args.rollout_batch_size * args.epoch_length / args.model_train_freq
    model_steps_per_epoch = int(rollout_length * rollouts_per_epoch)
    new_pool_size = args.model_retain_epochs * model_steps_per_epoch

    sample_all = model_pool.return_all()
    new_model_pool = ReplayMemory(new_pool_size)
    new_model_pool.push_batch(sample_all)

    return new_model_pool




def rollout_model(args, predict_env, agent, model_pool, env_pool, rollout_length):
    state, action, reward, next_state, done, sim_state = env_pool.sample_all_batch(args.rollout_batch_size)

    all_state = np.zeros((args.rollout_batch_size, rollout_length, state.shape[1])) # B x T x D
    all_action = np.zeros((args.rollout_batch_size, rollout_length, action.shape[1])) # B x T x A
    all_reward = np.zeros((args.rollout_batch_size, rollout_length, 1)) # B x T x R
    all_sim_state = np.zeros((args.rollout_batch_size, 1, sim_state.shape[1])) # B x 1 x S
    all_sim_state[:, 0, :] = sim_state

    for i in range(rollout_length):
        policy_distr = agent.forward_actor(torch.Tensor(state).to("cuda:0"), normed_input=False)
        action = policy_distr.sample().detach().cpu().numpy()
        next_states, rewards, terminals, info = predict_env.step(state, action)
        all_state[:, i] = state
        all_action[:, i] = action
        all_reward[:, i] = rewards
        state = next_states
    return all_state, all_action, all_reward, all_sim_state


def train_policy_repeats(args, total_step, train_step, cur_step, env_pool, model_pool, agent):
    if total_step % args.train_every_n_steps > 0:
        return 0, {}

    if train_step > args.max_train_repeat_per_step * total_step:
        return 0, {}

    for i in range(args.num_train_repeat):
        env_batch_size = int(args.policy_train_batch_size * args.real_ratio)
        model_batch_size = args.policy_train_batch_size - env_batch_size

        env_state, env_action, env_reward, env_next_state, env_done, sim_state = env_pool.sample(int(env_batch_size))

        if model_batch_size > 0 and len(model_pool) > 0:
            model_state, model_action, model_reward, model_next_state, model_done, sim_state = model_pool.sample_all_batch(int(model_batch_size))
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = np.concatenate((env_state, model_state), axis=0), \
                                                                                    np.concatenate((env_action, model_action),
                                                                                                   axis=0), np.concatenate(
                (np.reshape(env_reward, (env_reward.shape[0], -1)), model_reward), axis=0), \
                                                                                    np.concatenate((env_next_state, model_next_state),
                                                                                                   axis=0), np.concatenate(
                (np.reshape(env_done, (env_done.shape[0], -1)), model_done), axis=0)
        else:
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = env_state, env_action, env_reward, env_next_state, env_done

        batch_reward, batch_done = np.squeeze(batch_reward), np.squeeze(batch_done)
        batch_done = (~batch_done).astype(int)
        metrics = agent.update_parameters((batch_state, batch_action, batch_reward, batch_next_state, batch_done), args.policy_train_batch_size, i)

    return args.num_train_repeat, metrics


from gym.spaces import Box


class SingleEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super(SingleEnvWrapper, self).__init__(env)
        obs_dim = env.observation_space.shape[0]
        obs_dim += 2
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        torso_height, torso_ang = self.env.sim.data.qpos[1:3]  # Need this in the obs for determining when to stop
        obs = np.append(obs, [torso_height, torso_ang])

        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        torso_height, torso_ang = self.env.sim.data.qpos[1:3]
        obs = np.append(obs, [torso_height, torso_ang])
        return obs
    
def dwm_dataset_to_env_pool(dwm_dataset, env_pool):
    print(dwm_dataset.data_buffer._dict.keys())

    path_lengths = dwm_dataset.data_buffer._dict["path_lengths"]
    steps_added = 0
    for i, length in enumerate(path_lengths):
        for j in range(length):
            cur_state = dwm_dataset.data_buffer._dict["observations"][i][j]
            action = dwm_dataset.data_buffer._dict["actions"][i][j]
            reward = dwm_dataset.data_buffer._dict["rewards"][i][j]
            next_state = dwm_dataset.data_buffer._dict["next_observations"][i][j]
            done = dwm_dataset.data_buffer._dict["terminals"][i][j]
            sim_state = dwm_dataset.data_buffer._dict["sim_states"][i][j]
            env_pool.push(cur_state, action, reward, next_state, done, sim_state)
            steps_added += 1
    print(f"Added {steps_added} steps to env pool")

def main(args=None):
    if args is None:
        args = readParser()

    # Initial environment
    env = create_env(args.env_name, suite=args.suite)
    wandb.init(entity="a2i", project="diffusion_world_models", group=args.group, config=args)

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # env.seed(args.seed)

    # Initial ensemble model
    state_size = np.prod(env.observation_space.shape)
    action_size = np.prod(env.action_space.shape)

    # Intial agent
    a2c = ActorCritic(in_dim=state_size, out_actions=action_size, normalizer=None)
    ac_path = join(args.load_path, f"step-{args.load_step}-ac.pt")
    a2c.load_state_dict(torch.load(ac_path))
    print(f"Loaded actor critic from {ac_path}")
    print(f"Actor std dev: ", a2c.logstd(torch.Tensor([0.0])).exp().mean().item() + a2c.min_std)

    env_model = EnsembleDynamicsModel(args.num_networks, args.num_elites, state_size, action_size, args.reward_size, args.pred_hidden_size,
                                        use_decay=args.use_decay)

    # Predict environments
    predict_env = PredictEnv(env_model, args.env_name, args.model_type)

    # Initial pool for env
    env_pool = ReplayMemory(args.replay_size)

    dataset_path = join(args.load_path, f"step-{args.load_step}-dataset.pkl")
    with open(dataset_path, 'rb') as f:
        dwm_dataset = pickle.load(f)

    dwm_dataset_to_env_pool(dwm_dataset, env_pool)

    # # Initial pool for model
    rollouts_per_epoch = args.rollout_batch_size * args.epoch_length / args.model_train_freq
    model_steps_per_epoch = int(1 * rollouts_per_epoch)
    new_pool_size = args.model_retain_epochs * model_steps_per_epoch
    model_pool = ReplayMemory(new_pool_size)

    # # Sampler of environment
    env_sampler = EnvSampler(env, max_path_length=args.max_path_length)

    train(args, env_sampler, predict_env, a2c, env_pool, model_pool, env)


if __name__ == '__main__':
    main()
