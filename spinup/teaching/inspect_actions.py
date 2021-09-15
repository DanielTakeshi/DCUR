import matplotlib
#matplotlib.use('Agg')  # prevents plt.show()
import matplotlib.pyplot as plt
plt.style.use('seaborn-white') # actually i find plots are easier to see this way
import seaborn as sns  # Daniel: not sure if needed
# https://stackoverflow.com/questions/43080259/
# no-outlines-on-bins-of-matplotlib-histograms-or-seaborn-distplots/43080772
plt.rcParams["patch.force_edgecolor"] = True
plt.rcParams['agg.path.chunksize'] = 100000

import time
import joblib
import gym
import os
import os.path as osp
import torch
import numpy as np
import pickle
from collections import defaultdict
from scipy.stats import gaussian_kde
from spinup import EpochLogger
from spinup.algos.pytorch.td3.td3 import ReplayBuffer
from spinup.algos.pytorch.td3.core import MLPActorCritic
from spinup.user_config import DEFAULT_DATA_DIR
from spinup.teaching.offline_rl import sanity_check_args
from spinup.teaching.load_policy import (
        load_policy_and_env, sample_std, get_buffer_base_name)
np.set_printoptions(linewidth=200)
gym.logger.set_level(40)

BUFFER_DATA_DIR = "/data/spinup-data/data"

def load_policy_model(fpath, itr='last'):
    if itr == 'last':
        # check filenames for epoch (AKA iteration) numbers, find maximum value
        pytsave_path = osp.join(fpath, 'pyt_save')
        saves = [int(x.split('.')[0][6:]) for x in os.listdir(pytsave_path) if len(x)>8 and 'model' in x]
        itr = '%d'%max(saves) if len(saves) > 0 else ''
    else:
        assert isinstance(itr, int), \
            "Bad value provided for itr (needs to be int or 'last')."
        itr = '%d'%itr

    # load the entire model
    fname = osp.join(fpath, 'pyt_save', 'model-'+itr+'.pt')
    print('\n\nLoading actor-critic model from %s.\n\n'%fname)
    actor_critic = torch.load(fname)
    
    return actor_critic

def load_one_buffer(env_fn, args):
    buffer_path = args.buffer_path
    buffer_size = int(1e6)
    np_path = args.np_path
    replay_size = int(1e6)
    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]


    # Load buffer.
    params_desired = dict(buffer_path=buffer_path, np_path=np_path, env_arg=args.env)
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
    replay_buffer.load_from_disk(buffer_path, buffer_size, params_desired=params_desired)

    return replay_buffer, act_dim, act_limit

def smooth(values, weight=0.5):
    last = values[0]
    smoothed = []
    for point in values:
        new_val = last * weight + (1 - weight) * point
        smoothed.append(new_val)
        last = new_val
    return smoothed

if __name__ == '__main__':
    # Arguments mostly copied from OfflineRL since we're also loading a replay buffer.
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',        type=str)
    parser.add_argument('--hid',        type=int, default=256)
    parser.add_argument('--l',          type=int, default=2)
    parser.add_argument('--gamma',      type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--exp_name',   type=str, default='td3')
    parser.add_argument('--itr', '-i',  type=int, default=-1)
    
    parser.add_argument('--norender', '-nr', action='store_true')

    # Detects the replay buffer we used.
    parser.add_argument('--buffer_path', '-bp', type=str, default=None,
        help='Full path to the saved replay buffer we load for Offline RL.')
    parser.add_argument('--np_path', '-np', type=str, default=None,
        help='If enabled, we should load a noise predictor from this path.')
    parser.add_argument('--t_source', type=str, default=None,
        help='The exp_name of the teacher, with the seed please.')
    args = parser.parse_args()

    # load policy first
    exp_seed   = args.t_source
    exp        = (exp_seed.split('_'))[:-1]   # split and dump the '_sXYZ' in the exp_seed.
    exp        = '_'.join(exp)                # join them back (with underscores).
    args.fpath = osp.join(BUFFER_DATA_DIR, exp, exp_seed)
    assert os.path.exists(args.fpath), args.fpath
    assert 'data/' in args.fpath, f'Double check {args.fpath}'
    print(f'Loading:  {args.fpath}')
    actor_critic = load_policy_model(args.fpath, args.itr if args.itr >=0 else 'last')

    # load all same-sized buffer inside the given path
    path_to_buffers = osp.join(BUFFER_DATA_DIR, args.buffer_path)
    found_buffers = []
    for name in os.listdir(path_to_buffers):
        if 'steps-1000000' in name and '.p' in name and 'constant' not in name:
            if 'final' in name:
                found_buffers = [name] + found_buffers
            else:
                found_buffers.append(name)

    print(f'Found {len(found_buffers)} buffers for this agent')

    nrow, ncol = 6, len(found_buffers)
    fig, axs = plt.subplots(nrow, ncol, sharex=True, figsize=(4*ncol, 3*nrow))

    for name, idx in zip(found_buffers, range(ncol)):
        # load one buffer
        args.buffer_path = osp.join(path_to_buffers, name)
        assert os.path.exists(args.buffer_path), args.buffer_path
        args.fpath = (args.buffer_path).split('/buffer/')[0]  # only keep: /.../data/exp>/<exp_seed>
        buffer, act_dim, act_limit = load_one_buffer(lambda : gym.make(args.env), args)

        buffer_obses = torch.as_tensor(buffer.obs_buf, dtype=torch.float32)
        buffer_actions = torch.as_tensor(buffer.act_buf, dtype=torch.float32)

        with torch.no_grad():
            policy_actions = actor_critic.pi(buffer_obses)
            buffer_q_values = (actor_critic.q1(buffer_obses, buffer_actions) + actor_critic.q2(buffer_obses, buffer_actions)) / 2
            policy_q_values = (actor_critic.q1(buffer_obses, policy_actions) + actor_critic.q2(buffer_obses, policy_actions)) / 2

            action_dist = ((buffer_actions - policy_actions) ** 2).numpy()
            # mean, std = np.mean(action_dist, axis=0), np.std(action_dist, axis=0)
            # print(mean.shape, std.shape)
            
            
            action_dist = np.mean(action_dist, axis=1)
            # mean, std = np.zeros(action_dist.shape), np.zeros(action_dist.shape)
            
            # for j in range(1, 1+int(1e6)):
            #     mean[j], std[j] = np.mean(action_dist[:j]), np.std(action_dist[:j])
            # action_dist = (action_dist - mean)/std
            size = 12 
            ax = axs[0, idx]
            ax.scatter(range(int(1e6)), action_dist, s=1)
            ax.set_xlabel('Train steps', size=size)
            title = 'final buffer' if 'final' in name else name.split("-")[-3]
            ax.set_title( title+': action distance', size=size)

            smoothed = smooth(action_dist, weight=0.8)
            ax = axs[1, idx]
            ax.scatter(range(int(1e6)), smoothed, s=1)
            ax.set_ylim([0, 4])
            ax.set_xlabel('Train steps', size=size)
            ax.set_title( title+': action distance, smooth coeff 0.8', size=size)

            normalized = (action_dist - action_dist.mean()) / action_dist.std()
            ax = axs[2, idx]
            ax.set_title( title+': action distance, normalized', size=size)
            ax.set_xlabel('Train steps', size=size)
            ax.scatter(range(int(1e6)), normalized, s=1)


            q_dist = ((buffer_q_values - policy_q_values) ** 2).numpy()
            ax = axs[3, idx]
            ax.scatter(range(int(1e6)), q_dist, s=1)
            ax.set_ylim([0, 1e5])
            ax.set_title( title+': q value distance', size=size)
            ax.set_xlabel('Train steps', size=size)

            smoothed = smooth(q_dist, weight=0.8)
            ax = axs[4, idx]
            ax.scatter(range(int(1e6)), smoothed, s=1)
            ax.set_ylim([0, 1e5])
            ax.set_xlabel('Train steps', size=size)
            ax.set_title( title+': q distance, smooth coeff 0.8', size=size)

            normalized = (q_dist - q_dist.mean()) / q_dist.std()
            ax = axs[5, idx]
            #ax.set_ylim([0, 1e5])
            ax.scatter(range(int(1e6)), normalized, s=1)
            ax.set_title( title+': q distance, normalized', size=size)
            ax.set_xlabel('Train steps', size=size)

   

    fig.suptitle(exp_seed+' inspect different buffers with final policy', size=15)
    plt.show()
    plt.tight_layout()
    fig.subplots_adjust(top=0.9)
    plt.savefig(f'inspect_actions_{exp_seed}.png')
    print("plot saved")

            

        



