"""
Plot noise predictor. Also includes time and xi/varepsilon predictor.
Save plots in the /.../teacher/experiments/ subdirectory, with the model, etc.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import seaborn as sns  # Daniel: not sure if needed
# https://stackoverflow.com/questions/43080259/
# no-outlines-on-bins-of-matplotlib-histograms-or-seaborn-distplots/43080772
plt.rcParams["patch.force_edgecolor"] = True

import argparse
import pandas as pd
from copy import deepcopy
import os
import os.path as osp
import itertools
import numpy as np
import sys
import gym
import json
import time
import pickle
import torch
import torch.nn as nn
from collections import defaultdict
from spinup.teaching.noise_predictor import MLPNoise
from spinup.teaching.load_policy import load_policy_and_env
from spinup.user_config import DEFAULT_DATA_DIR
ENV_NAMES = ['ant', 'halfcheetah', 'hopper', 'walker2d']

# Matplotlib stuff
titlesize = 32
xsize = 30
ysize = 30
ticksize = 28
legendsize = 24
er_alpha = 0.25
lw = 3


def smooth(data, window=1):
    """Try to smooth in a similar way as spinup's normal plotting code."""
    if window > 1:
        y = np.ones(window)
        x = np.asarray(data)
        z = np.ones(len(x))
        smoothed_x = np.convolve(x,y,'same') / np.convolve(z,y,'same')
        return smoothed_x
    else:
        return data


def parse_name(pt_file):
    """Parse the .txt file or adjust for legend label."""
    pt_file = (pt_file.replace('.txt','')).replace('progress-','')
    pt_file = pt_file.replace('uniform_','Unif_')
    pt_file = pt_file.replace('1000000','1M')
    pt_file = pt_file.replace('0500000','0.5M')
    pt_file = pt_file.replace('0200000','0.2M')
    return pt_file


def plot_and_hist(args, preds_labels, progress_tables, progress_tables_files):
    """The data is in pandas format. For now we make two side by side plots,
    one with loss curves and the other with histograms.

    If data augmentation was used, then separate based on real data (first half) and augmented
    (second half), assuming that's the train / valid split.
    """
    colors = ['blue', 'red', 'yellow', 'black']
    nrows, ncols = 1, 2
    if args.data_aug:
        ncols = 4
    fig, ax = plt.subplots(nrows, ncols, squeeze=False, figsize=(11*ncols, 8*nrows))

    # Track maxval so we get the maximum y coord to keep.
    maxval = -np.float('inf')

    for idx, (pt, pl, pt_file) in enumerate(zip(progress_tables, preds_labels, progress_tables_files)):
        pt_file_base = os.path.basename(pt_file)  # the '.txt base'
        parsed_base = parse_name(pt_file_base)
        print(pt_file, parsed_base)
        LossT = pt['LossT']
        LossV = pt['LossV']
        LossV_Baseline = pt['LossVNaive']  # baseline
        assert np.std(LossV_Baseline) < 1e-5, np.std(LossV_Baseline)
        LossV_Baseline = np.mean(LossV_Baseline)

        # Actually this works directly from the pandas table. Double check that this is correct ...
        label_t = f'{parsed_base}\nLossT [{np.min(LossT):0.3f}]'
        label_v = f'{parsed_base}\nLossV [{np.min(LossV):0.3f}]'
        label_naive = f'LossV Mean Pred: {LossV_Baseline:0.3f}'
        ax[0,0].plot(LossT, ls='--', lw=lw, color=colors[idx], label=label_t)
        ax[0,0].plot(LossV, ls='-',  lw=lw, color=colors[idx], label=label_v)
        ax[0,0].axhline(LossV_Baseline, ls='dashdot', lw=lw, color=colors[idx], label=label_naive)

        # Only for getting a reasonable set of y-axis values for loss values.
        maxval = max(maxval, np.max(LossT))
        maxval = max(maxval, np.max(LossV))
        maxval = max(maxval, LossV_Baseline)

        # Now the histograms? It's a bit hacky but let's only do this for the 1M training data
        # case, otherwise we have a lot of plots to deal with. Note: valid_preds_final is a
        # list of numbers, but valid_labels is an array, so convert preds_l to an array.
        if 't-1M' in parsed_base:
            n_bins = 30
            preds_l  = np.array(pl['valid_preds_final'])
            labels_l = pl['valid_labels']
            bins = np.histogram(np.hstack((preds_l,labels_l)), bins=n_bins)[1]
            label_p = 'preds, avg {:.3f}\nmin,max ({:.3f},{:.3f})'.format(
                    np.mean(preds_l), np.min(preds_l), np.max(preds_l))
            label_l = 'labels, avg {:.3f}\nmin,max ({:.3f},{:.3f})'.format(
                    np.mean(labels_l), np.min(labels_l), np.max(labels_l))
            ax[0,1].hist(preds_l,  color='r', alpha=0.8, rwidth=1.0, bins=bins, label=label_p)
            ax[0,1].hist(labels_l, color='b', alpha=0.8, rwidth=1.0, bins=bins, label=label_l)
            #ax[0,1].set_ylim( [0, int(len(preds_l)/2)])  # Only for this case

            if args.data_aug:
                # Now split this up further. REAL DATA, wherever it's not -1.
                labels_real = (pl['valid_labels'] != -1)
                preds  = preds_l[labels_real]
                labels = labels_l[labels_real]
                bins = np.histogram(np.hstack((preds,labels)), bins=n_bins)[1]
                label_p = 'preds, avg {:.3f}\nmin,max ({:.3f},{:.3f})'.format(
                        np.mean(preds), np.min(preds), np.max(preds))
                label_l = 'labels, avg {:.3f}\nmin,max ({:.3f},{:.3f})'.format(
                        np.mean(labels), np.min(labels), np.max(labels))
                ax[0,2].hist(preds,  color='r', alpha=0.8, rwidth=1.0, bins=bins, label=label_p)
                ax[0,2].hist(labels, color='b', alpha=0.8, rwidth=1.0, bins=bins, label=label_l)
                errors = np.mean((preds - labels) ** 2)
                title_2 = f'Real ({len(preds)}). Valid L2: {errors:0.3f}'
                ax[0,2].set_title(title_2, size=titlesize)

                # Then AUGMENTED / FAKE data, wherever it is -1. If we did random Gaussian noise
                # for these states, the histograms should be very closely matched w/ground truth.
                labels_fake = (pl['valid_labels'] == -1)
                preds  = preds_l[labels_fake]
                labels = labels_l[labels_fake]
                bins = np.histogram(np.hstack((preds,labels)), bins=n_bins*2)[1] # increase bins.
                label_p = 'preds, avg {:.3f}\nmin,max ({:.3f},{:.3f})'.format(
                        np.mean(preds), np.min(preds), np.max(preds))
                label_l = 'labels, avg {:.3f}\nmin,max ({:.3f},{:.3f})'.format(
                        np.mean(labels), np.min(labels), np.max(labels))
                ax[0,3].hist(preds,  color='r', alpha=0.8, rwidth=1.0, bins=bins, label=label_p)
                ax[0,3].hist(labels, color='b', alpha=0.8, rwidth=1.0, bins=bins, label=label_l)
                errors = np.mean((preds - labels) ** 2)
                title_3 = f'Fake ({len(preds)}). Valid L2: {errors:0.3f}'
                ax[0,3].set_title(title_3, size=titlesize)
                ax[0,3].set_ylim( [0, int(len(preds)/1)])  # Only for this case

    # Set ylim. Double check that we're setting a good maximum?
    ax[0,0].set_ylim([0.0, maxval+0.001])

    # Double check these as well ...
    filename = progress_tables_files[0]  # we need a better way
    plot_title = ''
    for env_n in ENV_NAMES:
        if env_n in filename:
            plot_title = env_n
            break
    if plot_title == '':
        raise ValueError('something went wrong')
    assert plot_title in args.exp_seed, f'{plot_title} vs {args.exp_seed}'
    if args.noise == 'time_predictor':
        xlabel = 'Fraction Time'
    elif 'nonaddunif' in args.noise:
        xlabel = 'The xi or varepsilon'
    else:
        xlabel = 'Sigma $\sigma$'
    ax[0,0].set_title(args.exp_seed, size=titlesize)
    ax[0,1].set_title(f'{plot_title}_hist_valid; {args.ss}', size=titlesize)
    ax[0,0].set_xlabel('Train Epoch', size=xsize)
    ax[0,1].set_xlabel(xlabel, size=xsize)
    ax[0,0].set_ylabel('Loss (nn.MSELoss())', size=ysize)
    ax[0,1].set_ylabel('Frequency', size=ysize)

    # Bells and whistles.
    for r in range(nrows):
        for c in range(ncols):
            leg = ax[r,c].legend(loc="best", ncol=1, prop={'size':legendsize})
            for legobj in leg.legendHandles:
                legobj.set_linewidth(5.0)
            ax[r,c].tick_params(axis='x', labelsize=ticksize)
            ax[r,c].tick_params(axis='y', labelsize=ticksize)
    plt.tight_layout()
    fig_suffix = 'plot_{}_{}-{}.png'.format(plot_title, args.noise, args.ss)
    if args.data_aug:
        fig_suffix = fig_suffix.replace('.png', '_data-aug.png')
    figname = osp.join(args.head, fig_suffix)
    plt.savefig(figname)
    print("Just saved: {}\n".format(figname))


def load_policies(args, env, predictor, itr_to_pol, max_ep_len=1000, max_episodes=10,
        std_err_mean=True):
    """Load policies (saved teacher snapshots), roll them out, then apply predictor.

    Args:
        predictor: The network we saved, the last epoch. Normally I do the best one,
            but for simplicity I did the last, which was almost always the best anyway.
        itr_to_pol: maps <saved_epoch> to <current_policy> from the original teacher
            training run. We can then take some env steps.
    """
    assert len(itr_to_pol) == 6, itr_to_pol
    nrows, ncols = 2, 3
    fig, ax = plt.subplots(nrows, ncols, squeeze=False, figsize=(11*ncols, 8*nrows))
    nrow, ncol = 0, 0

    for itr in sorted(list(itr_to_pol.keys())):
        print(f'\nNow testing policy from original teacher iteration at: {itr}')
        current_policy = itr_to_pol[itr]
        t_to_obs = defaultdict(list)

        # Roll out `max_episodes` episodes while accumulating saved observations.
        o, r, d, ep_ret, ep_len, num_eps = env.reset(), 0, False, 0, 0, 0
        while num_eps < max_episodes:
            t_to_obs[ep_len].append(o)
            a = current_policy(o)
            o2, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1
            d = False if ep_len==max_ep_len else d
            o = o2
            if d or (ep_len == max_ep_len):
                print('  Episode %d \t EpRet %.3f \t EpLen %d' % (num_eps, ep_ret, ep_len))
                o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
                num_eps += 1

        # Make predictions from those data points. Note: predictions are NOT squashed
        # between [0,1] so we could apply a clamping factor if needed.
        t_to_predavg = []
        t_to_predstd = []
        for t in range(max_ep_len):
            if len(t_to_obs[t]) == 0:
                break
            # Shape (B, obs_dim); B is the number of episodes that got to this time step.
            o_batch_np = np.array(t_to_obs[t])
            o_batch = torch.as_tensor(o_batch_np, dtype=torch.float32)
            p_batch = predictor(o_batch)
            p_batch = torch.clamp(p_batch, 0, 1)
            p_batch_np = p_batch.detach().numpy()
            t_to_predavg.append(np.mean(p_batch_np))
            if std_err_mean:
                t_to_predstd.append(np.std(p_batch_np) / np.sqrt(len(p_batch_np)))
            else:
                t_to_predstd.append(np.std(p_batch_np))
        t_to_predavg = np.array(t_to_predavg)
        t_to_predstd = np.array(t_to_predstd)

        # Plot predictions as a function of time.
        envname = str(env.unwrapped).replace(' instance', '')
        subp_title = f'{envname} Policy itr: {itr}'
        xvals = np.arange(len(t_to_predavg))
        label = 'pred seed: {}, avg: {:0.3f},\nfirst20: {:0.3f}, last20: {:0.3f}'.format(
                args.seed,
                np.mean(t_to_predavg),
                np.mean(t_to_predavg[:20]),
                np.mean(t_to_predavg[20:]),
        )
        if '_td3_' in args.fpath:
            label = '(td3) '+label
        if '_sac_' in args.fpath:
            label = '(sac) '+label
        ax[nrow, ncol].plot(xvals, t_to_predavg, lw=1, color='blue', label=label)
        ax[nrow, ncol].fill_between(xvals,
                                    t_to_predavg - t_to_predstd,
                                    t_to_predavg + t_to_predstd,
                                    color='blue',
                                    alpha=0.2)
        ax[nrow, ncol].set_title(subp_title, size=titlesize)
        ncol += 1
        if ncol == ncols:
            nrow, ncol = 1, 0

    # Bells and whistles.
    for r in range(nrows):
        for c in range(ncols):
            leg = ax[r,c].legend(loc="best", ncol=1, prop={'size':30})
            for legobj in leg.legendHandles:
                legobj.set_linewidth(5.0)
            ax[r,c].tick_params(axis='x', labelsize=ticksize)
            ax[r,c].tick_params(axis='y', labelsize=ticksize)
            ax[r,c].set_xlim([-1, max_ep_len + 1])
            ax[r,c].set_ylim([0 - 0.01, 1 + 0.01])
            ax[r,c].set_xlabel(f'Time (max={max_ep_len})', size=xsize)
            ax[r,c].set_ylabel(f'Predictions (epis: {max_episodes})', size=ysize)
    plt.tight_layout()
    fig_suffix = 'plot_preds_rollouts_{}-{}.png'.format(args.noise, args.ss)
    if args.data_aug:
        fig_suffix = fig_suffix.replace('.png', '_data-aug.png')
    figname = osp.join(args.head, fig_suffix)
    plt.savefig(figname)
    print("Just saved: {}\n".format(figname))


def predict_buffer(args, env, predictor, std_err_mean=True):
    """Now, let's just load the final buffer, and run prediction on those items.

    After all, we should hopefully see some variation -- if not, no real point in applying
    this time predictor.
    """
    envname = str(env.unwrapped).replace(' instance', '')
    _, t_source = os.path.split(args.fpath)
    nrows, ncols = 2, 3
    fig, ax = plt.subplots(nrows, ncols, squeeze=False, figsize=(11*ncols, 8*nrows))

    # Load the logged replay buffer from teacher training, see `ReplayBuffer.load_from_disk`.
    buffer_dir = osp.join(args.fpath, 'buffer')
    buffer_path = [osp.join(buffer_dir,x) for x in os.listdir(buffer_dir) if 'final_buffer' in x]
    assert len(buffer_path) == 1, buffer_path
    buffer_path = buffer_path[0]
    print(f'loading buffer: {buffer_path}')
    save_dict = torch.load(buffer_path)
    obs_buf = save_dict['obs']
    obs2_buf = save_dict['obs2']
    rew_buf = save_dict['rew']
    done_buf = save_dict['done']
    h_label = (f'Rewards. Len: {len(rew_buf)}\nMax: {np.max(rew_buf):0.2f}, '
        f'Min: {np.min(rew_buf):0.2f}, Medi: {np.median(rew_buf):0.2f}')
    ax[0,0].hist(rew_buf, color='blue', alpha=0.6, rwidth=1.0, bins=30, label=h_label)
    ax[0,0].set_xlabel(f'Rewards, Teacher Training', size=xsize)
    ax[0,0].set_ylabel(f'Frequency', size=ysize)
    ax[0,0].set_title(t_source, size=titlesize)
    min_h = np.min(rew_buf)
    max_h = np.max(rew_buf)

    # Just make o_batch and o2_batch giant batch sizes. :D
    o_batch  = torch.as_tensor(obs_buf, dtype=torch.float32)
    o2_batch = torch.as_tensor(obs2_buf, dtype=torch.float32)
    p_batch  = torch.clamp(predictor(o_batch), 0, 1)
    p2_batch = torch.clamp(predictor(o2_batch), 0, 1)
    p_batch_np  = p_batch.detach().numpy()
    p2_batch_np = p2_batch.detach().numpy()
    preds_diffs_all = p2_batch_np - p_batch_np

    # Filter out any with d=True. NOTE: we could filter those which hit the time limit, but
    # I don't have a way of extracting that from the data. However, those only happen, by
    # definition, 1 out of 1000 time steps at most. And since o -> a -> o2 is consistent
    # with env dynamics, we might actually want to predict o2 even though it's not stored.
    preds_diffs = preds_diffs_all[ np.where(done_buf != 1) ]

    # Now report those prediction differences in a histogram.
    h_label = (f'Pred Diffs. Len: {len(preds_diffs)}\nMax: {np.max(preds_diffs):0.2f}, '
        f'Min: {np.min(preds_diffs):0.2f}, Medi: {np.median(preds_diffs):0.2f}')
    ax[0,1].hist(preds_diffs, color='purple', alpha=0.6, rwidth=1.0, bins=30, label=h_label)
    ax[0,1].set_xlabel(f'Predictions: $f(o2) - f(o)$', size=xsize)
    ax[0,1].set_ylabel(f'Frequency', size=ysize)
    ax[0,1].set_title(f'Preds of: {t_source}', size=titlesize)
    ax[0,1].set_xlim([-0.70, 0.70])

    # Let's split this up into quartiles, to see the distribution throughout training.
    data = [dict(idx=(0,2), start=     0, end= 250000),
            dict(idx=(1,0), start=250000, end= 500000),
            dict(idx=(1,1), start=500000, end= 750000),
            dict(idx=(1,2), start=750000, end=1000000),]

    def subplot(item, all_preds, done_buf):
        start = item['start']
        end = item['end']
        preds = all_preds[start:end]  # partial
        dones = done_buf[start:end]   # partial
        preds = preds[ np.where(dones != 1) ]
        h_label = (f'Preds. Len: {len(preds)}\nMax: {np.max(preds):0.2f}, '
            f'Min: {np.min(preds):0.2f}, Medi: {np.median(preds):0.2f}')
        ax[item['idx']].hist(preds, color='r', alpha=0.6, rwidth=1.0, bins=30, label=h_label)
        ax[item['idx']].set_xlabel(f'Predictions: $f(o2) - f(o)$', size=xsize)
        ax[item['idx']].set_ylabel(f'Frequency', size=ysize)
        ax[item['idx']].set_title(f'Time Steps: {start} --> {end}', size=titlesize)
        ax[item['idx']].set_xlim([-0.70, 0.70])

    for item in data:
        subplot(item, preds_diffs_all, done_buf)

    # Bells and whistles.
    for r in range(nrows):
        for c in range(ncols):
            leg = ax[r,c].legend(loc="best", ncol=1, prop={'size':30})
            for legobj in leg.legendHandles:
                legobj.set_linewidth(5.0)
            ax[r,c].tick_params(axis='x', labelsize=ticksize)
            ax[r,c].tick_params(axis='y', labelsize=ticksize)
    plt.tight_layout()
    fig_suffix = 'plot_full_buffer_{}-{}.png'.format(args.noise, args.ss)
    if args.data_aug:
        fig_suffix = fig_suffix.replace('.png', '_data-aug.png')
    figname = osp.join(args.head, fig_suffix)
    plt.savefig(figname)
    print("Just saved: {}\n".format(figname))


def predict_buffer_raw(args, env, predictor, std_err_mean=True):
    """Now, let's just load the final buffer, and run prediction on those items.

    SIMPLE: predict on teacher data (i.e., what it trained on) and see performance
    as a function of time.
    """
    envname = str(env.unwrapped).replace(' instance', '')
    _, t_source = os.path.split(args.fpath)
    nrows, ncols = 1, 1
    fig, ax = plt.subplots(nrows, ncols, squeeze=False, figsize=(12*ncols, 9*nrows))

    # Load the logged replay buffer from teacher training, see `ReplayBuffer.load_from_disk`.
    buffer_dir = osp.join(args.fpath, 'buffer')
    buffer_path = [osp.join(buffer_dir,x) for x in os.listdir(buffer_dir) if 'final_buffer' in x]
    assert len(buffer_path) == 1, buffer_path
    buffer_path = buffer_path[0]
    print(f'loading buffer: {buffer_path}')
    save_dict = torch.load(buffer_path)
    obs_buf = save_dict['obs']
    obs2_buf = save_dict['obs2']
    rew_buf = save_dict['rew']
    done_buf = save_dict['done']

    # Just make o_batch a giant batch. :D
    o_batch    = torch.as_tensor(obs_buf, dtype=torch.float32)
    p_batch    = torch.clamp(predictor(o_batch), 0, 1)
    p_batch_np = p_batch.detach().numpy()

    # Actually this needs to be smoothed. A LOT.
    p_batch_np = smooth(p_batch_np, window=50)

    # Now report those predictions as a function of time.
    h_label = (f'Clamped Preds. Len: {len(p_batch_np)}\nMax: {np.max(p_batch_np):0.2f}, '
        f'Min: {np.min(p_batch_np):0.2f}, Medi: {np.median(p_batch_np):0.2f}')
    x_vals = np.arange(len(p_batch_np))
    ax[0,0].plot(x_vals, p_batch_np, lw=0.2, color='purple', label=h_label)
    ax[0,0].set_xlabel(f'Train Time (Teacher)', size=xsize)
    ax[0,0].set_ylabel(f'Time Predictor Output', size=ysize)
    plot_title = f'Preds: {t_source}'
    if args.data_aug:
        plot_title += ' data-aug'
    ax[0,0].set_title(plot_title, size=titlesize)

    # Bells and whistles.
    for r in range(nrows):
        for c in range(ncols):
            leg = ax[r,c].legend(loc="best", ncol=1, prop={'size':30})
            for legobj in leg.legendHandles:
                legobj.set_linewidth(5.0)
            ax[r,c].tick_params(axis='x', labelsize=ticksize)
            ax[r,c].tick_params(axis='y', labelsize=ticksize)
    plt.tight_layout()
    fig_suffix = 'plot_full_buffer_raw_{}-{}.png'.format(args.noise, args.ss)
    if args.data_aug:
        fig_suffix = fig_suffix.replace('.png', '_data-aug.png')
    figname = osp.join(args.head, fig_suffix)
    plt.savefig(figname)
    print("Just saved: {}\n".format(figname))


def check_reward(args, env):
    """Check reward of teacher buffer in more detail.

    These can add more understanding than the simpler plots that just show the reward
    statistics (i.e., the mean, max, min, etc.) from the logger's `progress.txt` file.
    """
    envname = str(env.unwrapped).replace(' instance', '')
    _, t_source = os.path.split(args.fpath)
    nrows, ncols = 2, 3
    fig, ax = plt.subplots(nrows, ncols, squeeze=False, figsize=(11*ncols, 8*nrows))

    # From the teacher's original training (so, not the time predictor training).
    # Contrast this with `plot.py` results, should give similar rewards modulo smoothing.
    # Note: for some SAC teachers that Mandi trained, they do not have AverageRew stored.
    progress_file = osp.join(args.fpath, 'progress.txt')
    train_table = pd.read_table(progress_file)
    if 'AverageRew' in train_table:
        rew_avg = train_table['AverageRew']
        rew_max = train_table['MaxRew']
        rew_min = train_table['MinRew']
        ax[0,0].plot(rew_avg, lw=3, label='AverageRew')
        ax[0,0].plot(rew_max, lw=3, label='MaxRew')
        ax[0,0].plot(rew_min, lw=3, label='MinRew')
        ax[0,0].set_xlabel(f'Train Epochs', size=xsize)
        ax[0,0].set_ylabel(f'Reward', size=ysize)
        ax[0,0].set_title(t_source, size=titlesize)

    # Load the logged replay buffer from teacher training, see `ReplayBuffer.load_from_disk`.
    buffer_dir = osp.join(args.fpath, 'buffer')
    buffer_path = [osp.join(buffer_dir,x) for x in os.listdir(buffer_dir) if 'final_buffer' in x]
    assert len(buffer_path) == 1, buffer_path
    buffer_path = buffer_path[0]
    print(f'loading buffer: {buffer_path}')
    save_dict = torch.load(buffer_path)
    rew_buf = save_dict['rew']
    h_label = (f'Rewards. Len: {len(rew_buf)}\nMax: {np.max(rew_buf):0.1f}, '
        f'Min: {np.min(rew_buf):0.1f}, Med: {np.median(rew_buf):0.1f}')
    ax[0,1].hist(rew_buf, color='b', alpha=0.8, rwidth=1.0, bins=30, label=h_label)
    ax[0,1].set_xlabel(f'Rewards, Teacher Training', size=xsize)
    ax[0,1].set_ylabel(f'Frequency', size=ysize)
    ax[0,1].set_title(t_source, size=titlesize)
    min_h = np.min(rew_buf)
    max_h = np.max(rew_buf)

    # Now for the rest of the plots, maybe show distribution throughout training?
    data = [dict(idx=(0,2), start=     0, end= 250000),
            dict(idx=(1,0), start=250000, end= 500000),
            dict(idx=(1,1), start=500000, end= 750000),
            dict(idx=(1,2), start=750000, end=1000000),]

    def subplot(item):
        start = item['start']
        end   = item['end']
        rews = rew_buf[start:end]
        h_label = (f'Rewards. Len: {len(rews)}\nMax: {np.max(rews):0.1f}, '
            f'Min: {np.min(rews):0.1f}, Med: {np.median(rews):0.1f}')
        ax[item['idx']].hist(rews, color='r', alpha=0.8, rwidth=1.0, bins=30, label=h_label)
        ax[item['idx']].set_xlabel(f'Rewards, Teacher Training', size=xsize)
        ax[item['idx']].set_ylabel(f'Frequency', size=ysize)
        ax[item['idx']].set_title(f'Time Steps: {start} --> {end}', size=titlesize)
        ax[item['idx']].set_xlim([min_h, max_h])

    for item in data:
        subplot(item)

    # Bells and whistles.
    for r in range(nrows):
        for c in range(ncols):
            leg = ax[r,c].legend(loc="best", ncol=1, prop={'size':30})
            for legobj in leg.legendHandles:
                legobj.set_linewidth(5.0)
            ax[r,c].tick_params(axis='x', labelsize=ticksize)
            ax[r,c].tick_params(axis='y', labelsize=ticksize)
    plt.tight_layout()
    fig_suffix = 'plot_check_reward_{}-{}.png'.format(args.noise, args.ss)
    if args.data_aug:
        fig_suffix = fig_suffix.replace('.png', '_data-aug.png')
    figname = osp.join(args.head, fig_suffix)
    plt.savefig(figname)
    print("Just saved: {}\n".format(figname))


def load_predictor(args, obs_dim):
    """Loading predictor using PyTorch's recommended procedure."""
    predictor_f = sorted([osp.join(args.head, x) for x in os.listdir(args.head)
            if 'sigma_predictor-' in x and x[-4:] == '.tar' and args.noise in x and ss in x])
    if args.data_aug:
        predictor_f = [x for x in predictor_f if 'data-aug' in x]
    else:
        predictor_f = [x for x in predictor_f if 'data-aug' not in x]
    assert len(predictor_f) == 1, predictor_f
    checkpoint = torch.load(predictor_f[0])
    predictor = MLPNoise(obs_dim, hidden_sizes=(256,256), activation=nn.ReLU,
                         data_type='continuous', n_outputs=1)
    predictor.load_state_dict( checkpoint['model_state_dict'] )
    predictor.eval()
    return predictor


if __name__ == '__main__':
    """We should combine different dataset sizes together and put them into one plot."""
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath',                type=str)
    parser.add_argument('--itr', '-i',          type=int, default=-1)
    parser.add_argument('--noise',              type=str, default='const_0.0')
    parser.add_argument('--train_size', '-ts',  type=int, default=1000000)
    parser.add_argument('--valid_size', '-vs',  type=int, default=200000)
    parser.add_argument('--seed', '-s',         type=int, default=0, help='maybe for ensembles')
    parser.add_argument('--load_policies',      action='store_true',
        help='If enabled, load the policy (or policies) and predict those')
    parser.add_argument('--predict_buffer_raw', action='store_true',
        help='If enabled, predict on the full final buffer and report just f(o)')
    parser.add_argument('--predict_buffer',     action='store_true',
        help='If enabled, predict on the full final buffer and report f(o2)-f(o)')
    parser.add_argument('--check_rew',          action='store_true',
        help='If enabled, check rewards from teacher training run in more detail')
    parser.add_argument('--data_aug',           action='store_true',
        help='If enabled, analyze predictor trained from augmented data.')
    args = parser.parse_args()
    args.ss = ss = f'seed-{str(args.seed).zfill(2)}'
    print()
    print('*'*150)
    print('*'*150)

    # The usual, adding data dir to start.
    if not os.path.exists(args.fpath):
        print(f'{args.fpath} does not exist, pre-pending {DEFAULT_DATA_DIR}')
        args.fpath = osp.join(DEFAULT_DATA_DIR, args.fpath)
        assert osp.exists(args.fpath), args.fpath

    # Logger saved noise predictor training info: /../<exp>/<exp_seed>/experiments.
    if args.fpath[-1] == '/':
        args.fpath = args.fpath[:-1]
    _, args.exp_seed = osp.split(args.fpath)  # put in plot title
    args.head = head = osp.join(args.fpath, 'experiments')
    assert osp.exists(args.head), args.head

    # These are the 'progress-{noise}-{train_size}-{valid_size}.txt' files.
    progress_tables = []
    progress_tables_files = sorted([osp.join(head, x) for x in os.listdir(head)
            if 'progress-' in x and x[-4:] == '.txt' and args.noise in x and ss in x])
    if args.data_aug:
        progress_tables_files = [x for x in progress_tables_files if 'data-aug' in x]
    else:
        progress_tables_files = [x for x in progress_tables_files if 'data-aug' not in x]
    print('Loading progress.txt files:')
    for pf in progress_tables_files:
        print(f'\t{pf}')
        pd_table = pd.read_table(pf)
        progress_tables.append(pd_table)

    # These are the 'preds_labels-{...}.pkl' files with valid labels and predictions.
    print('Loading data files')
    preds_labels = []
    preds_labels_files = sorted([osp.join(head, x) for x in os.listdir(head)
            if 'preds_labels-' in x and x[-4:] == '.pkl' and args.noise in x and ss in x])
    if args.data_aug:
        preds_labels_files = [x for x in preds_labels_files if 'data-aug' in x]
    else:
        preds_labels_files = [x for x in preds_labels_files if 'data-aug' not in x]
    for plf in preds_labels_files:
        with open(plf, 'rb') as fh:
            data = pickle.load(fh)
        l1, l2 = len(data['valid_preds_final']), len(data['valid_labels'])
        print(f'\t{plf} w/lengths {l1} {l2}')
        preds_labels.append(data)
    print()

    # Load the env, as well as the trained time predictor (at the end of training).
    env, _ = load_policy_and_env(args.fpath, itr='last')
    obs_dim = env.observation_space.shape[0]  # don't forget [0]
    predictor = load_predictor(args, obs_dim)
    print(f'Loaded env: {env}, and predictor.')

    # Let's just do one of the following to save compute.
    if args.load_policies:
        itr_to_pol = {}
        for itr in [0, 50, 100, 150, 200, 250]:
            _, policy = load_policy_and_env(args.fpath, itr=itr)
            itr_to_pol[itr] = policy
        load_policies(args, env=env, predictor=predictor, itr_to_pol=itr_to_pol)
    elif args.predict_buffer:
        predict_buffer(args, env=env, predictor=predictor)
    elif args.predict_buffer_raw:
        predict_buffer_raw(args, env=env, predictor=predictor)
    elif args.check_rew:
        check_reward(args, env=env)
    else:
        plot_and_hist(args, preds_labels, progress_tables, progress_tables_files)