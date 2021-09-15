"""
Use this file to check and inspect the loaded policy. I think an easy way is to take
saved replay buffers, then plot the observations. But maybe we should support loading
a policy as well to visualize the rollouts? So this is both `teaching/offline_rl.py`
and `teaching/load_policy.py`.

Note: for now, passing in exp_name even though we don't actually use it -- we may want
to save videos in a programmatic way, and we'd use exp_name for that.
"""
import matplotlib
#matplotlib.use('Agg')  # prevents plt.show()
import matplotlib.pyplot as plt
#plt.style.use('seaborn') # actually i find plots are easier to see this way
import seaborn as sns  # Daniel: not sure if needed
# https://stackoverflow.com/questions/43080259/
# no-outlines-on-bins-of-matplotlib-histograms-or-seaborn-distplots/43080772
plt.rcParams["patch.force_edgecolor"] = True

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
from spinup.user_config import DEFAULT_DATA_DIR
from spinup.teaching.offline_rl import sanity_check_args
from spinup.teaching.load_policy import (
        load_policy_and_env, sample_std, get_buffer_base_name)
np.set_printoptions(linewidth=200)

# For the ENV, we're assuming we divide the obs dimension, then round down to get
# the positional portion (velocity is the remaining part).
ENV_NAMES = ['ant', 'halfcheetah', 'hopper', 'walker2d']
ENV_TO_POS = {'HalfCheetah-v3': 8,
              'Hopper-v3': 5,
              'Walker2d-v3': 8,}


def plot_obs_phases(args, stats, sigma, savefig=False):
    """Phase plot, 2D based on two observation components.

    For simplicity ignore O_i[k] x O_i[k] cases, since those are just boring.
    So if it's Hopper, we want a 4x4 plot, showing {(0,1), ..., (0,4)}, etc.

    Hopefully we can see a good pattern. To make it easier to understand, use a
    special red marker to indicate the _start_ of the joint's trajectory.
    """
    nrows = ENV_TO_POS[args.env] - 1
    ncols = nrows
    fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True, squeeze=False,
            figsize=(5*ncols, 5*nrows))

    # Just using normal matplotlib plotting with '->' connectors. Since we're
    # not using same-same components, need to adjust j by -1 for indexing.
    for i in range(nrows + 1):
        for j in range(i + 1, ncols + 1):
            sub_title = f'O_i[{i}] x O_i[{j}]'
            ax[i,j-1].plot(stats[i], stats[j], '->', lw=0.25, markersize=1.0)
            ax[i,j-1].set_title(sub_title, size=22)
            ax[i,j-1].scatter(stats[i][0], stats[j][0], s=100, c='red')

    plt.axis('square')
    for i in range(nrows):
        for j in range(ncols):
            ax[i,j].tick_params(axis='x', labelsize=22)
            ax[i,j].tick_params(axis='y', labelsize=22)

    sup_title = f'Phases, {args.env}, sigma: {sigma:0.2f}, len: {len(stats[0])-1}'
    fig.suptitle(sup_title, size=34)
    plt.tight_layout()
    fig.subplots_adjust(top=0.92)

    # Note that hese two differ. I optimize for `savefig`.
    if savefig:
        count = len([x for x in os.listdir('.') if x[-4:]=='.png' and 'phase' in x])
        count = str(count).zfill(4)
        plt.savefig(f'phase_{args.env}_c{count}.png')
    else:
        plt.show()


def plot_obs_multi_row(args, stats, sigma, savefig=False):
    """One row per observation component, or however we specify `ENV_TO_POS`.

    This is easy to spot-check, but also consider a phase plot.
    """
    nrows = ENV_TO_POS[args.env]
    ncols = 1
    fig, ax = plt.subplots(nrows, ncols, sharey=True, squeeze=False,
            figsize=(12*ncols, int(2.5*nrows)))

    # One row per observation component.
    xvals = np.arange(len(stats[0]))
    for i in range(nrows):
        label = f'O_i[{i}]'
        ax[i,0].plot(xvals, stats[i], lw=1.0, label=label)
        ax[i,0].legend(loc="best", ncol=1, prop={'size':25})
    title = f'Env: {args.env}, sigma: {sigma:0.2f}, len: {len(stats[0])-1}'
    ax[0,0].set_title(title, size=30)

    for i in range(nrows):
        for j in range(ncols):
            ax[i,j].tick_params(axis='x', labelsize=22)
            ax[i,j].tick_params(axis='y', labelsize=22)
    plt.tight_layout()

    # Note that hese two differ. I optimize for `savefig`.
    if savefig:
        count = len([x for x in os.listdir('.') if x[-4:]=='.png' and 'multirow' in x])
        count = str(count).zfill(4)
        plt.savefig(f'multirow_{args.env}_c{count}.png')
    else:
        plt.show()


def buffer_scatter(args, savefig, max_count, fig_index, replay_o,
        half='first', keyword='scatter'):
    """Makes a scatter plot of either position or velocity components.

    We could extend this to all the components, I guess?
    """
    if half == 'first':
        offset = 0
        nrows = ENV_TO_POS[args.env] - 1
        tail = 'pos'
    elif half == 'second':
        offset = ENV_TO_POS[args.env]
        nrows = ENV_TO_POS[args.env]
        tail = 'vel'
    else:
        raise ValueError(half)
    ncols = nrows
    fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True, squeeze=False,
            figsize=(5*ncols, 5*nrows))

    # Since we're not using same-same components, need to adjust j by -1 for indexing.
    for i in range(nrows + 1):
        for j in range(i + 1, ncols + 1):
            sub_title = f'O_i[{offset+i}] x O_i[{offset+j}]'
            ax[i,j-1].scatter(replay_o[:,offset+i],
                              replay_o[:,offset+j],
                              s=1, c='midnightblue', alpha=0.15)
            ax[i,j-1].set_title(sub_title, size=22)

    # Bells and whistles.
    plt.axis('square')
    for i in range(nrows):
        for j in range(ncols):
            ax[i,j].tick_params(axis='x', labelsize=22)
            ax[i,j].tick_params(axis='y', labelsize=22)
    sup_title = f'Obs state, {args.env}, {args.noise}, len: {max_count}'
    fig.suptitle(sup_title, size=34)
    plt.tight_layout()
    fig.subplots_adjust(top=0.92)

    # Note that these two differ. I optimize for `savefig`.
    if savefig:
        count = str(fig_index).zfill(4)
        figname = osp.join('figures',
                f'{keyword}_{args.env}_{args.noise}_c{count}_{tail}.png')
        plt.savefig(figname)
        print(f'See:  {figname}')
    else:
        plt.show()


def density_scatter(x, y, ax=None, fig=None, sort=True, bins=25, **kwargs):
    """Scatter plot colored by 2d histogram.

    https://numpy.org/doc/stable/reference/generated/numpy.histogram2d.html
    Simply computes the 2D histogram of the data x and y. Note the densit=True argument:

    If False, the default, returns the number of samples in each bin. If True, returns
    the probability density function at the bin, bin_count / sample_count / bin_area.
    Makes sense! This gives us a sense for densities. TODO(daniel) double check my
    understanding of this and how we can integrate this to 1.
    """
    from matplotlib import cm
    from matplotlib.colors import Normalize
    from scipy.interpolate import interpn

    # x_e and y_e are bin edges along x and y dimensions.
    data, x_e, y_e = np.histogram2d(x, y, bins=bins, density=True)
    z = interpn(
        (0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])),
        data,
        np.vstack([x,y]).T,
        method="splinef2d",
        bounds_error=False)

    # To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort:
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter(x, y, c=z, **kwargs)
    norm = Normalize(vmin=np.min(z), vmax=np.max(z))
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm), ax=ax)
    cbar.ax.set_ylabel('Density')
    return ax


def buffer_gaussian_kde(args, savefig, max_count, fig_index, replay_o,
        half='first', keyword='density'):
    """Estimates densities using Gaussian KDEs. Similar arguments as buffer_scatter.

    https://en.wikipedia.org/wiki/Kernel_density_estimation
    https://python-graph-gallery.com/86-avoid-overlapping-in-scatterplot-with-2d-density/
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
    https://stackoverflow.com/questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density-in-matplotlib

    For now I'm following: https://stackoverflow.com/a/53865762/3287820
    It approximates Gaussian KDEs for cases when we have 100K or more data points (as we
    do ... for EVERY observation pairing). This is far more computationally feasible, and
    we can cross-reference this with the scatter plots.
    """
    if half == 'first':
        offset = 0
        nrows = ENV_TO_POS[args.env] - 1
        tail = 'pos'
    elif half == 'second':
        offset = ENV_TO_POS[args.env]
        nrows = ENV_TO_POS[args.env]
        tail = 'vel'
    else:
        raise ValueError(half)
    ncols = nrows
    fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True, squeeze=False,
            figsize=(5*ncols, 5*nrows))

    # Since we're not using same-same components, need to adjust j by -1 for indexing.
    for i in range(nrows + 1):
        for j in range(i + 1, ncols + 1):
            x = replay_o[:,offset+i]
            y = replay_o[:,offset+j]
            density_scatter(x=x, y=y, ax=ax[i,j-1], fig=fig, s=1)
            sub_title = f'O_i[{offset+i}] x O_i[{offset+j}]'
            ax[i,j-1].set_title(sub_title, size=22)

    # Bells and whistles.
    plt.axis('square')
    for i in range(nrows):
        for j in range(ncols):
            ax[i,j].tick_params(axis='x', labelsize=22)
            ax[i,j].tick_params(axis='y', labelsize=22)

    # TODO(Daniel): why is this not working? Axes are still invisible.
    for a in ax.flatten():
        for tk in a.get_yticklabels():
            tk.set_visible(True)
        for tk in a.get_xticklabels():
            tk.set_visible(True)

    sup_title = f'Obs state, {args.env}, {args.noise}, len: {max_count}'
    fig.suptitle(sup_title, size=34)
    plt.tight_layout()
    fig.subplots_adjust(top=0.92)

    # Note that these two differ. I optimize for `savefig`.
    if savefig:
        count = str(fig_index).zfill(4)
        figname = osp.join('figures',
                f'{keyword}_{args.env}_{args.noise}_c{count}_{tail}.png')
        plt.savefig(figname)
        print(f'See:  {figname}')
    else:
        plt.show()


def run_policy(env, args, get_action, render=True, max_ep_len=1000, size=int(1e6)):
    """Actually runs the policy, but more importantly, gathers data.

    Max episode length should be 1000 for standard MuJoCo environments.
    https://github.com/openai/spinningup/issues/37
    https://github.com/openai/gym/blob/master/gym/envs/__init__.py

    Just run it, but don't actually save the data.
    """
    assert env is not None
    data_type = 'neither'
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=size)
    sigma = sample_std(noise_distr=args.noise)
    print(f'\nMade new buffer w/obs: {obs_dim}, act: {act_dim}, act_limit: {act_limit}.')
    print(f'Noise distr: {args.noise}, starting sigma: {sigma:0.3f}')

    # The usual initialization, but we should make the output more distinctive.
    base_pth = osp.join(args.fpath, 'rollout_buffer_txts')
    base_txt = get_buffer_base_name(args.noise, size, data_type=data_type, ending='.txt')
    logger = EpochLogger(output_dir=base_pth, output_fname=base_txt)
    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0

    # Start a fresh stats dict for recording observations, add 1 item to each list inside.
    # stats[k] = [obs_0[k], obs_1[k], obs_2[k], ..., obs_T[k]] for episode of length T+1.
    stats = defaultdict(list)
    for j in range(len(o)):
        stats[j].append(o[j])

    for _ in range(size):
        if render:
            # One can use mode='rgb_array' to save as images, then form GIFs at the end. However
            # using mode='human' makes it appear as a video for us to see.
            #obs_array = env.render(mode='rgb_array')  # defaults to 500x500
            env.render(mode='human')
            time.sleep(1e-3)

        # Get noise-free action from the policy.
        a = get_action(o)

        # Perturb action based on our desired noise parameter. We clip here. The policy
        # net ALREADY has a tanh at the end, but we need another one for the added noise.
        a += sigma * np.random.randn(act_dim)
        a = np.clip(a, -act_limit, act_limit)

        # Do the usual environment stepping.
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store information for usage later.
        replay_buffer.store(o, a, r, o2, d, std=sigma)

        # Update most recent observation [mainly to make rbuffer.store() just use one call].
        o = o2

        # Add more info.
        for j in range(len(o)):
            stats[j].append(o[j])

        # Need max_ep_len if we are also using it to assign d=False if we hit the max len.
        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            print('Episode %d \t EpRet %.3f \t EpLen %d \t Sigma %.3f' % (n, ep_ret, ep_len, sigma))
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            n += 1

            # Plot data from this finished episode, one row per specified value.
            plot_obs_multi_row(args, stats, sigma, savefig=True)

            # Plot phase data for this episode.
            plot_obs_phases(args, stats, sigma, savefig=True)

            # Refresh the stats dict for the next episode.
            stats = defaultdict(list)
            for j in range(len(o)):
                stats[j].append(o[j])

            # Whenever episode finishes, reset the noise.
            sigma = sample_std(noise_distr=args.noise)

    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()


def inspect_data(env_fn, args, savefig=True):
    """Given a replay buffer of data, plot observations in some way.

    More scalable than loading and visualizing policy on the fly. Perhaps we should
    limit analysis to constant sigma values, i.e., not mixing sigmas together as I
    normally do with Uniform_X_Y?

    As with the policy loading case, we'll define a `stats` defaultdict such that:
        stats[k] = [obs_0[k], obs_1[k], obs_2[k], ..., obs_T[k]]
    has observation components for the k-th time step, for an episode of length T+1.
    However, we want to combine multiple datasets together, hence we'll make a list
    of all these `stats` defaultdicts.
    """
    buffer_path = args.buffer_path
    buffer_size = args.buffer_size
    np_path = args.np_path
    replay_size = int(1e6)
    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    # Load buffer.
    params_desired = dict(buffer_path=buffer_path, np_path=np_path, env_arg=args.env)
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
    replay_buffer.load_from_disk(buffer_path, buffer_size, params_desired=params_desired)

    # Collect observation data. Not sure if we'll need `replay_d`.
    replay_o = replay_buffer.obs_buf   #  (N, env_obs_dim)
    max_count = 200000                 # NOTE may save compute
    replay_o = replay_o[:max_count]

    # Scatter plots of positions, then velocities. Then Gaussian KDEs (edit: an approximation to it).
    # TODO(daniel) figure indexing needs to be changed! E.g., hard to tell act=0.1 vs act=0.5, etc.
    fig_index = len([x for x in os.listdir('figures')
            if x[-4:]=='.png' and 'density' in x and 'pos' in x])
    print('Working on scatter plots...')
    #buffer_scatter(     args, savefig, max_count, fig_index, replay_o=replay_o, half='first')
    #buffer_scatter(     args, savefig, max_count, fig_index, replay_o=replay_o, half='second')
    print('Working on densities ...')
    buffer_gaussian_kde(args, savefig, max_count, fig_index, replay_o=replay_o, half='first')
    #buffer_gaussian_kde(args, savefig, max_count, fig_index, replay_o=replay_o, half='second')


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
    parser.add_argument('--noise',      type=str)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--plot_buffer', action='store_true', default=False)
    parser.add_argument('--plot_load_pol', action='store_true', default=False)

    # Detects the replay buffer we used.
    parser.add_argument('--buffer_size', '-bs', type=int, default=int(1e6),
        help='Size of the Offline RL data.')
    parser.add_argument('--buffer_path', '-bp', type=str, default=None,
        help='Full path to the saved replay buffer we load for Offline RL.')
    parser.add_argument('--np_path', '-np', type=str, default=None,
        help='If enabled, we should load a noise predictor from this path.')
    parser.add_argument('--t_source', type=str, default=None,
        help='The exp_name of the teacher, with the seed please.')
    args = parser.parse_args()

    # Inspect data (from a SAVED buffer) vs loading a policy and plotting 'on the fly'
    assert args.plot_buffer or args.plot_load_pol, 'One of these must be True'
    assert not (args.plot_buffer and args.plot_load_pol), 'Do not make both True'

    # Adding data dir to start only if `plot_buffer`, requires existing buffer present.
    if args.plot_buffer:
        if not os.path.exists(args.buffer_path):
            print(f'{args.buffer_path} does not exist, pre-pending {DEFAULT_DATA_DIR}')
            args.buffer_path = osp.join(DEFAULT_DATA_DIR, args.buffer_path)
            assert os.path.exists(args.buffer_path), args.buffer_path
            args.fpath = (args.buffer_path).split('/buffer/')[0]  # only keep: /.../data/exp>/<exp_seed>
        inspect_data(lambda : gym.make(args.env), args)

    elif args.plot_load_pol:
        # Here it is not necessary for the buffer path to exist, we just need the snapshot.
        exp_seed   = args.t_source
        exp        = (exp_seed.split('_'))[:-1]   # split and dump the '_sXYZ' in the exp_seed.
        exp        = '_'.join(exp)                # join them back (with underscores).
        args.fpath = osp.join(DEFAULT_DATA_DIR, exp, exp_seed)
        assert os.path.exists(args.fpath), args.fpath
        assert 'data/' in args.fpath, f'Double check {args.fpath}'
        print(f'Loading:  {args.fpath}')
        env, get_action = load_policy_and_env(args.fpath, args.itr if args.itr >=0 else 'last')
        run_policy(env, args, get_action, not(args.norender))
