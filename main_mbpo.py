import argparse
import time
import gym
import torch
import numpy as np
from itertools import count

import logging
import wandb

import os
import os.path as osp
import json

from sac.replay_memory import ReplayMemory
from sac.sac import SAC
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

    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')

    parser.add_argument('--model_retain_epochs', type=int, default=1, metavar='A',
                        help='retain epochs')
    parser.add_argument('--model_train_freq', type=int, default=1000, metavar='A',
                        help='frequency of training')
    parser.add_argument('--rollout_batch_size', type=int, default=10000, metavar='A',
                        help='rollout number M')
    parser.add_argument('--epoch_length', type=int, default=1000, metavar='A',
                        help='steps per epoch')
    parser.add_argument('--rollout_min_epoch', type=int, default=20, metavar='A',
                        help='rollout min epoch')
    parser.add_argument('--rollout_max_epoch', type=int, default=150, metavar='A',
                        help='rollout max epoch')
    parser.add_argument('--rollout_min_length', type=int, default=1, metavar='A',
                        help='rollout min length')
    parser.add_argument('--rollout_max_length', type=int, default=15, metavar='A',
                        help='rollout max length')
    parser.add_argument('--num_epoch', type=int, default=1000, metavar='A',
                        help='total number of epochs')
    parser.add_argument('--min_pool_size', type=int, default=1000, metavar='A',
                        help='minimum pool size')
    parser.add_argument('--real_ratio', type=float, default=0.0, metavar='A',
                        help='ratio of env samples / model samples')
    parser.add_argument('--train_every_n_steps', type=int, default=1, metavar='A',
                        help='frequency of training policy')
    parser.add_argument('--num_train_repeat', type=int, default=5, metavar='A',
                        help='times to training policy per step')
    parser.add_argument('--max_train_repeat_per_step', type=int, default=5, metavar='A',
                        help='max training times per step')
    parser.add_argument('--policy_train_batch_size', type=int, default=256, metavar='A',
                        help='batch size for training policy')
    parser.add_argument('--init_exploration_steps', type=int, default=5000, metavar='A',
                        help='exploration steps initially')
    parser.add_argument('--max_path_length', type=int, default=1000, metavar='A',
                        help='max length of path')
    parser.add_argument('--eval_error_interval', type=int, default=5000, metavar='A',
                        help='max length of path')

    parser.add_argument('--model_type', default='pytorch', metavar='A',
                        help='predict model -- pytorch or tensorflow')

    parser.add_argument('--cuda', default=True, action="store_true",
                        help='run on CUDA (default: True)')
    return parser.parse_args()


def train(args, env_sampler, predict_env, agent, env_pool, model_pool, error_env):
    total_step = 0
    reward_sum = 0
    rollout_length = 1
    exploration_before_start(args, env_sampler, env_pool, agent)

    for epoch_step in range(args.num_epoch):
        start_step = total_step
        train_policy_steps = 0
        all_metrics = dict()
        for i in count():
            cur_step = total_step - start_step

            if cur_step >= args.epoch_length and len(env_pool) > args.min_pool_size:
                break

            if cur_step >= 0 and total_step % args.model_train_freq == 0 and args.real_ratio < 1.0:
                print(f"Current Step {total_step}: Training predict model")
                train_predict_model(args, env_pool, predict_env)

                new_rollout_length = set_rollout_length(args, epoch_step)
                if rollout_length != new_rollout_length:
                    rollout_length = new_rollout_length
                    model_pool = resize_model_pool(args, rollout_length, model_pool)

                print(f"Rollout model")
                rollout_states, rollout_actions, rollout_rewards, init_sim_state = rollout_model(args, predict_env, agent, model_pool, env_pool, rollout_length)
                
                if total_step % args.eval_error_interval == 0:
                    metrics = compute_traj_errors(error_env, rollout_states, rollout_actions, rollout_rewards, init_sim_state)
                    all_metrics.update(metrics)

            cur_state, action, next_state, reward, done, info = env_sampler.sample(agent)
            env_pool.push(cur_state, action, reward, next_state, done, info["sim_state"])

            if len(env_pool) > args.min_pool_size:
                steps, metrics = train_policy_repeats(args, total_step, train_policy_steps, cur_step, env_pool, model_pool, agent)
                train_policy_steps += steps
                # all_metrics.update(metrics)
            total_step += 1

            if total_step % args.epoch_length == 0:
                '''
                avg_reward_len = min(len(env_sampler.path_rewards), 5)
                avg_reward = sum(env_sampler.path_rewards[-avg_reward_len:]) / avg_reward_len
                logging.info("Step Reward: " + str(total_step) + " " + str(env_sampler.path_rewards[-1]) + " " + str(avg_reward))
                print(total_step, env_sampler.path_rewards[-1], avg_reward)
                '''
                total_rewards = []
                for i in range(5):
                    env_sampler.current_state = None
                    sum_reward = 0
                    done = False
                    test_step = 0

                    while (not done) and (test_step != args.max_path_length):
                        cur_state, action, next_state, reward, done, info = env_sampler.sample(agent, eval_t=True)
                        sum_reward += reward
                        test_step += 1
                    total_rewards.append(sum_reward)
                    # logger.record_tabular("total_step", total_step)
                    # logger.record_tabular("sum_reward", sum_reward)
                    # logger.dump_tabular()
                    print("Step Reward: " + str(total_step) + " " + str(sum_reward))
                all_metrics.update({"eval/eval_reward: ": sum(total_rewards) / len(total_rewards)})
                    
                # print(total_step, sum_reward)
        wandb.log(all_metrics, step=epoch_step)


def exploration_before_start(args, env_sampler, env_pool, agent):
    for i in range(args.init_exploration_steps):
        cur_state, action, next_state, reward, done, info = env_sampler.sample(agent)
        env_pool.push(cur_state, action, reward, next_state, done, info["sim_state"])


def set_rollout_length(args, epoch_step):
    rollout_length = (min(max(args.rollout_min_length + (epoch_step - args.rollout_min_epoch)
                              / (args.rollout_max_epoch - args.rollout_min_epoch) * (args.rollout_max_length - args.rollout_min_length),
                              args.rollout_min_length), args.rollout_max_length))
    return int(rollout_length)


def train_predict_model(args, env_pool, predict_env):
    # Get all samples from environment
    state, action, reward, next_state, done, sim_state = env_pool.sample(len(env_pool))
    delta_state = next_state - state
    inputs = np.concatenate((state, action), axis=-1)
    labels = np.concatenate((np.reshape(reward, (reward.shape[0], -1)), delta_state), axis=-1)

    predict_env.model.train(inputs, labels, batch_size=256, holdout_ratio=0.2, max_grad_updates=5000)


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
    
    eval_traj = 100
    all_state = np.zeros((eval_traj, rollout_length, state.shape[1])) # B x T x D
    all_action = np.zeros((eval_traj, rollout_length, action.shape[1])) # B x T x A
    all_reward = np.zeros((eval_traj, rollout_length, 1)) # B x T x R
    all_sim_state = np.zeros((eval_traj, 1, sim_state.shape[1])) # B x 1 x S
    all_sim_state[:, 0, :] = sim_state[:eval_traj]
    
    for i in range(rollout_length):
        # TODO: Get a batch of actions
        action = agent.select_action(state)
        next_states, rewards, terminals, info = predict_env.step(state, action)
        all_state[:, i] = state[:eval_traj]
        all_action[:, i] = action[:eval_traj]
        all_reward[:, i] = rewards[:eval_traj]
        # TODO: Push a batch of samples
        model_pool.push_batch([(state[j], action[j], rewards[j], next_states[j], terminals[j], None) for j in range(state.shape[0])])
        nonterm_mask = ~terminals.squeeze(-1)
        if nonterm_mask.sum() == 0:
            break
        state = next_states[nonterm_mask]
    return all_state, all_action, all_reward, all_sim_state


def train_policy_repeats(args, total_step, train_step, cur_step, env_pool, model_pool, agent):
    if total_step % args.train_every_n_steps > 0:
        return 0, {}

    if train_step > args.max_train_repeat_per_step * total_step:
        return 0, {}

    for i in range(args.num_train_repeat):
        env_batch_size = int(args.policy_train_batch_size * args.real_ratio)
        model_batch_size = args.policy_train_batch_size - env_batch_size

        if env_batch_size > 0:
            env_state, env_action, env_reward, env_next_state, env_done, sim_state = env_pool.sample(int(env_batch_size))

        if model_batch_size > 0 and len(model_pool) > 0:
            model_state, model_action, model_reward, model_next_state, model_done, sim_state = model_pool.sample_all_batch(int(model_batch_size))
            
            if env_batch_size > 0:
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = np.concatenate((env_state, model_state), axis=0), \
                                                                                        np.concatenate((env_action, model_action),
                                                                                                    axis=0), np.concatenate(
                    (np.reshape(env_reward, (env_reward.shape[0], -1)), model_reward), axis=0), \
                                                                                        np.concatenate((env_next_state, model_next_state),
                                                                                                    axis=0), np.concatenate(
                    (np.reshape(env_done, (env_done.shape[0], -1)), model_done), axis=0)
            else:
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = model_state, model_action, model_reward, model_next_state, model_done
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


def main(args=None):
    if args is None:
        args = readParser()

    # Initial environment
    env = create_env(args.env_name, suite=args.suite)
    eval_env = create_env(args.env_name, suite=args.suite)
    wandb.init(entity="a2i", project="MBPO_baseline", group=args.group, config=args)

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # env.seed(args.seed)

    # Intial agent
    agent = SAC(env.observation_space.shape[0], env.action_space, args)

    # Initial ensemble model
    state_size = np.prod(env.observation_space.shape)
    action_size = np.prod(env.action_space.shape)
    env_model = EnsembleDynamicsModel(args.num_networks, args.num_elites, state_size, action_size, args.reward_size, args.pred_hidden_size,
                                        use_decay=args.use_decay)

    # Predict environments
    predict_env = PredictEnv(env_model, args.env_name, args.model_type)

    # Initial pool for env
    env_pool = ReplayMemory(args.replay_size)
    # Initial pool for model
    rollouts_per_epoch = args.rollout_batch_size * args.epoch_length / args.model_train_freq
    model_steps_per_epoch = int(1 * rollouts_per_epoch)
    new_pool_size = args.model_retain_epochs * model_steps_per_epoch
    model_pool = ReplayMemory(new_pool_size)

    # Sampler of environment
    env_sampler = EnvSampler(env, max_path_length=args.max_path_length)

    train(args, env_sampler, predict_env, agent, env_pool, model_pool, eval_env)


if __name__ == '__main__':
    main()
