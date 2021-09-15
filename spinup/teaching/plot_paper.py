"""Plot Offline RL for the actual paper.

USE THIS FOR OFFICIAL RESULTS! We should be able to run this with a simple
bash script to generate all the possible results we would need to share.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import seaborn as sns  # Daniel: not sure if needed
# https://stackoverflow.com/questions/43080259/
# no-outlines-on-bins-of-matplotlib-histograms-or-seaborn-distplots/43080772
plt.rcParams["patch.force_edgecolor"] = True

import pandas as pd
from copy import deepcopy
import os
import os.path as osp
import numpy as np
import json
from spinup.user_config import DEFAULT_DATA_DIR as DDD
from spinup.teaching.plot_offline_rl import (
    parse_to_get_data_type, smooth, sanity_checks, ragged_array, adjust_fig_suffix,
    get_dirs_S_sorted, COLORS, COLORS_LINE, ENV_NAMES,
)
np.set_printoptions(linewidth=180, suppress=True)

# Matplotlib stuff
titlesize = 33
xsize = 31
ysize = 31
ticksize = 29
legendsize = 21  # adjust as needed
er_alpha = 0.25
lw = 2
NAME_TO_LABEL = {'ant': 'Ant-v3',
                 'halfcheetah': 'HalfCheetah-v3',
                 'hopper': 'Hopper-v3',
                 'walker2d': 'Walker2d-v3',}

# ------------------------------------------------------- #
# Methods to polish up labels, etc., for the actual paper #
# ------------------------------------------------------- #

def get_curriculum_label(args, tail):
    """Returns curriculum labels in a human-readable manner.

    E.g., tail could be:
        td3_offline_curriculum_ep_2500_logged_p_50000000_n_1000000
    A lot depends on what notation we decide to use.
    """
    curr_label = ''

    if 'online_stud_total_25000_' in tail:
        if 'ep_250_logged_scale_1.00t' in tail:
            #curr_label = 'C_scale(t; c=1.00); 2.5%'
            curr_label = '2.5%'
        elif 'ep_250_logged_p_1000000_n_1000000_' in tail:
            #curr_label = 'C_add(t; f=1M); 2.5%'
            curr_label = '2.5%'
        else:
            raise ValueError(tail)
    elif 'online_stud_total_50000_' in tail:
        if 'ep_250_logged_scale_1.00t' in tail:
            #curr_label = 'C_scale(t; c=1.00); 5.0%'
            curr_label = '5.0%'
        elif 'ep_250_logged_p_1000000_n_1000000_' in tail:
            #curr_label = 'C_add(t; f=1M); 5.0%'
            curr_label = '5.0%'
        else:
            raise ValueError(tail)
    elif 'online_stud_total_100000_' in tail:
        if 'ep_250_logged_scale_1.00t' in tail:
            #curr_label = 'C_scale(t; c=1.00); 10.0%'
            curr_label = '10.0%'
        elif 'ep_250_logged_p_1000000_n_1000000_' in tail:
            #curr_label = 'C_add(t; f=1M); 10.0%'
            curr_label = '10.0%'
        else:
            raise ValueError(tail)
    # Now the offline cases:
    elif 'ep_250_logged_p_1000000_n_1000000' in tail:
        curr_label = 'C_add(t; f=1M)'
    elif 'ep_250_logged_p_1000000_n_500000' in tail:
        curr_label = 'C_add(t; f=500K)'
    elif 'ep_250_logged_p_1000000_n_200000' in tail:
        curr_label = 'C_add(t; f=200K)'
    elif 'ep_250_logged_p_1000000_n_100000' in tail:
        curr_label = 'C_add(t; f=100K)'
    elif 'ep_250_logged_p_1000000_n_50000' in tail:
        curr_label = 'C_add(t; f=50K)'
    elif 'ep_250_logged_p_800000_n_0' in tail:
        curr_label = 'C_add(t; p=800K, f=0)'
    elif 'ep_250_logged_scale_0.50t' in tail:
        curr_label = 'C_scale(t; c=0.50)'
    elif 'ep_250_logged_scale_0.75t' in tail:
        curr_label = 'C_scale(t; c=0.75)'
    elif 'ep_250_logged_scale_1.00t' in tail:
        curr_label = 'C_scale(t; c=1.00)'
    elif 'ep_250_logged_scale_1.10t' in tail:
        curr_label = 'C_scale(t; c=1.10)'
    elif 'ep_250_logged_scale_1.25t' in tail:
        curr_label = 'C_scale(t; c=1.25)'
    elif 'ep_2500_logged_p_50000000_n_1000000' in tail:
        curr_label = 'C_add(t; f=1M)'
    elif 'ep_2500_logged_scale_1.00t' in tail:
        curr_label = 'C_scale(t; c=1.00)'
    else:
        raise ValueError(tail)
    return curr_label


def polish_figname(teacher_seed_dir, fig_suffix, args):
    """Special case for ease of use in Overleaf, need to remove these symbols."""
    fig_suffix = adjust_fig_suffix(fig_suffix, args)
    fig_suffix = fig_suffix.replace('[','')
    fig_suffix = fig_suffix.replace(']','')
    fig_suffix = fig_suffix.replace('\'','')
    figname = osp.join(teacher_seed_dir, fig_suffix)
    return figname


def get_teacher_stats(teacher_seed_dir):
    # Derive original online DeepRL teacher results from 'progress.txt'.
    # Let's also do the two stats that we compute for the student as well.
    prog_file = osp.join(teacher_seed_dir, 'progress.txt')
    assert os.path.exists(prog_file), prog_file
    teacher_data = pd.read_table(prog_file)
    teacher_base = os.path.basename(teacher_seed_dir)
    t_perf_np = teacher_data['AverageTestEpRet'].to_numpy()
    t_stat_1  = np.mean(t_perf_np[-10:])
    t_stat_2  = np.mean(t_perf_np[3:])
    t_label   = f'{teacher_base},  M1: {t_stat_1:0.1f},  M2: {t_stat_2:0.1f}'  # To report in table.
    return (teacher_data, teacher_base, t_perf_np, t_label)


def get_student_stats(s_sub_dir, teacher_seed_dir):
    # Similarly, load the student file statistics.
    print(f'\t{s_sub_dir}')
    prog_file = osp.join(s_sub_dir, 'progress.txt')
    config_file = osp.join(s_sub_dir, 'config.json')
    assert os.path.exists(prog_file), prog_file
    assert os.path.exists(config_file), config_file
    with open(config_file, 'rb') as fh:
        config_data = json.load(fh)
    student_data = pd.read_table(prog_file)
    sanity_checks(config=config_data,
                  progress=student_data,
                  teacher_path=teacher_seed_dir)
    return (config_data, student_data)

# ------------------------------------------------------- #
# Plotting methods!                                       #
# ------------------------------------------------------- #

def report_table_plot(args, teacher_seed_dir, student_dirs):
    """This is for reporting a table. See spinup.teaching.plot_offline_rl for details
    and documentation. The input is the same as the plot method.

    NOTE(daniel) As I've reiterated to myself many times, we need to ensure that for
    any runs that used overlap, we actually run them with consistent test-time episode
    statistics. The easiest way is to re-run with 10 fixed test episodes for reporting
    here. Then we do any episodes for overlap. See `spinup.teaching.offline_rl`.

    M1 = FINAL, so it should hopefully get the last 100 episodes.
    M2 = ALL, so it should get all (well, except the first few due to random policy).

    Update: let's just merge the plotting code here, it does the same thing, right?
    And for this we may want to see what happens with no subplots. Just overlay? With
    the long-horizon runs, this will make it easier, right?

    (June 08) What about putting all the 'add' curricula to the left, 'scale' to the right?
    (June 13) Minor edits to make it 'wider' (reduces vertical space :D), etc. Getting close!
    """
    window = args.window
    nrows, ncols = 1, 2
    fig, ax = plt.subplots(nrows, ncols, sharey=True, squeeze=False, figsize=(10*ncols, 6*nrows))
    env_plot = NAME_TO_LABEL[args.name]

    # Load teacher statistics.
    (teacher_data, teacher_base, _, t_label) = get_teacher_stats(teacher_seed_dir)

    # Plot teacher performance, SMOOTHED for readability. Note: no more teacher labels.
    ret_train   = smooth(teacher_data['AverageEpRet'],     window)
    ret_test    = smooth(teacher_data['AverageTestEpRet'], window)
    # Prior code (for debugging only)
    #label_train = f'{teacher_base} (Train)'
    #label_test  = f'{teacher_base} (Test)'
    #ax[0,0].plot(ret_train, ls='--', lw=lw, color='black', label=label_train)
    #ax[0,0].plot(ret_test,  ls='-',  lw=lw, color='black', label=label_test)
    # Newer code (for actual plots)
    label_teach = f'Teacher'
    ax[0,0].plot(ret_test,  ls='--',  lw=lw, color='black', label=label_teach)
    #ax[0,0].axhline(y=np.max(ret_test), color='black', lw=0.5, linestyle='--')
    ax[0,1].plot(ret_test,  ls='--',  lw=lw, color='black', label=label_teach)
    #ax[0,1].axhline(y=np.max(ret_test), color='black', lw=0.5, linestyle='--')

    # Student should be:  <student_exp>/<teacher_base_with_seed>_<seed>/
    # We MUST have `teacher_base` in the directory after <student_exp>.
    s_labels, s_labels_std, s_M1s, s_M2s, s_M1e, s_M2e = [], [], [], [], [], []
    sidx = 0
    for sd in student_dirs:
        student_subdirs = sorted([osp.join(sd,x) for x in os.listdir(sd) if teacher_base in x])
        if len(student_subdirs) == 0:
            continue
        s_stats_1, s_stats_2 = [], []
        student_stats = []
        terminated_early = 0  # number of seeds that terminated early

        # Now `student_subdirs`, for THIS PARTICULAR student, go through its RANDOM SEEDS.
        # Combine all student runs together with random seeds. Note: smoothing applied BEFORE
        # we append to `student_stats`, and before we take the mean / std for the plot. But,
        # compute desired statistics BEFORE smoothing (the last 10 and the average over all
        # evaluations) because we want to report those numbers in tables [see docs above].
        for s_sub_dir in student_subdirs:
            _, student_data = get_student_stats(s_sub_dir, teacher_seed_dir)

            # DO NOT DO SMOOTHING. Compute statistics M1 (stat_1) and M2 (stat_2).
            perf = student_data['AverageTestEpRet'].to_numpy()
            assert (len(perf) <= 250) or (len(perf) in [2500]), len(perf)
            if len(perf) < 250:
                print(f'Note: for {sd}, len(perf): {len(perf)}')
                terminated_early += 1
            stat_1 = np.mean(perf[-10:])
            stat_2 = np.mean(perf[3:])
            s_stats_1.append(stat_1)
            s_stats_2.append(stat_2)

            # Now smooth and add to student_stats for plotting later.
            student_result = smooth(student_data['AverageTestEpRet'], window)
            student_stats.append(student_result)

        # Extract label which is the <student_exp> not the <teacher_base_with_seed> portion.
        _, tail = os.path.split(sd)
        for name in ENV_NAMES:
            if name in tail:
                tail = tail.replace(f'{name}_', '')
        nb_seeds = len(s_stats_1)
        s_label = f'{tail} (x{nb_seeds}, e{terminated_early})'  # newline due to space
        s_label = s_label.replace('_offline_curriculum_', '_curr_')  # use for no newline
        s_label_std = str(s_label)

        # Shape is (num_seeds, num_recordings=250), usually 250 due to (1M steps = 250 epochs).
        # TODO(daniel) Recording of s_stat_{1,2} might not be intended if we have ragged array.
        s_M1     = np.mean(s_stats_1)                       # last 10
        s_M2     = np.mean(s_stats_2)                       # all (except first 3)
        s_M1_err = np.std(s_stats_1) / np.sqrt(nb_seeds)    # last 10
        s_M2_err = np.std(s_stats_2) / np.sqrt(nb_seeds)    # all (except first 3)
        s_label     += f'\n\t{s_M1:0.1f} & {s_M2:0.1f}'
        s_label_std += f'\n\t{s_M1:0.1f} $\pm$ {s_M1_err:0.1f} & {s_M2:0.1f} $\pm$ {s_M2_err:0.1f}'
        s_labels.append(s_label)
        s_labels_std.append(s_label_std)
        s_M1s.append(s_M1)
        s_M2s.append(s_M2)
        s_M1e.append(s_M1_err)
        s_M2e.append(s_M2_err)

        # Actually plot (with error regions if applicable). Standard error of the mean.
        # Also need to update the student label.
        curr_label = get_curriculum_label(args, tail)
        _row = 0 if 'add' in curr_label else 1
        label_curve = f'{curr_label}'  # It's understood this means 'Student'.
        student_ret, student_std, _ = ragged_array(student_stats)
        x_vals = np.arange(len(student_ret))
        ax[0,_row].plot(x_vals, student_ret, lw=lw, color=COLORS[sidx], label=label_curve)
        if nb_seeds > 1:
            ax[0,_row].fill_between(x_vals,
                                    student_ret - (student_std / np.sqrt(nb_seeds)),
                                    student_ret + (student_std / np.sqrt(nb_seeds)),
                                    color=COLORS[sidx],
                                    alpha=0.5)
        sidx += 1

    # Print table! Hopefully later in LaTeX code. Do w/ and w/out st-dev (actually, err).
    M1_max = max(s_M1s)
    M2_max = max(s_M2s)
    M1_ste = s_M1e[ s_M1s.index(M1_max) ]  # Index of the M1_max, get corresponding std-err.
    M2_ste = s_M2e[ s_M2s.index(M2_max) ]  # Index of the M2_max, get corresponding std-err.
    M1_thresh = M1_max - M1_ste
    M2_thresh = M2_max - M2_ste
    print('='*100)
    print(t_label)
    print('\nFor students, thresholds for M1 and M2 (with std err subtracted):')
    print(f'\t{M1_max:0.1f} - {M1_ste:0.1f} = {M1_thresh:0.1f}')
    print(f'\t{M2_max:0.1f} - {M2_ste:0.1f} = {M2_thresh:0.1f}\n')
    # Iterate, check overlapping standard errors.
    for (s_label, s_label_std, s_M1, s_M2, M1err, M2err) in \
            zip(s_labels, s_labels_std, s_M1s, s_M2s, s_M1e, s_M2e):
        print(s_label)
        print(s_label_std)
        if s_M1 == max(s_M1s):
            print(f'\tNOTE: max M1.')
        elif (s_M1+M1err >= M1_thresh):
            print(f'\tNOTE: bold M1. {s_M1:0.1f} + {M1err:0.1f} = {(s_M1+M1err):0.1f}')
        if s_M2 == max(s_M2s):
            print(f'\tNOTE: max M2.')
        elif (s_M2+M2err >= M2_thresh):
            print(f'\tNOTE: bold M2. {s_M2:0.1f} + {M2err:0.1f} = {(s_M2+M2err):0.1f}')
        print()
    print('='*100)

    # Now let's get back to plotting.
    ax[0,0].set_title(f'Performance ({env_plot})', size=titlesize)
    ax[0,1].set_title(f'Performance ({env_plot})', size=titlesize)
    ax[0,0].set_xlabel('Train Epochs', size=xsize)
    ax[0,1].set_xlabel('Train Epochs', size=xsize)
    ax[0,0].set_ylabel('Test Return', size=ysize)
    ax[0,1].set_ylabel('Test Return', size=ysize)
    for r in range(nrows):
        for c in range(ncols):
            leg = ax[r,c].legend(loc="best", ncol=1, prop={'size':legendsize})
            for legobj in leg.legendHandles:
                legobj.set_linewidth(5.0)
            ax[r,c].tick_params(axis='x', labelsize=ticksize)
            ax[r,c].tick_params(axis='y', labelsize=ticksize)
    plt.tight_layout()
    fig_suffix = f'plot_paper_{env_plot}.png'
    figname = polish_figname(teacher_seed_dir, fig_suffix, args)
    plt.savefig(figname)
    print(f'\nSAVED FIGURE: {figname}')


def report_table_plot_2500(args, teacher_seed_dir, student_dirs):
    """Reporting a table and making plots for the 2500 epoch case."""
    window = args.window
    nrows, ncols = 1, 1
    fig, ax = plt.subplots(nrows, ncols, sharey=True, squeeze=False, figsize=(17*ncols, 5*nrows))
    env_plot = NAME_TO_LABEL[args.name]

    # Load teacher statistics.
    (teacher_data, teacher_base, _, t_label) = get_teacher_stats(teacher_seed_dir)

    # Plot teacher performance, SMOOTHED for readability. Note: no more teacher labels.
    ret_train   = smooth(teacher_data['AverageEpRet'],     window)
    ret_test    = smooth(teacher_data['AverageTestEpRet'], window)
    # Prior code (for debugging only)
    #label_train = f'{teacher_base} (Train)'
    #label_test  = f'{teacher_base} (Test)'
    #ax[0,0].plot(ret_train, ls='--', lw=lw, color='black', label=label_train)
    #ax[0,0].plot(ret_test,  ls='-',  lw=lw, color='black', label=label_test)
    # Newer code (for actual plots)
    label_teach = f'Teacher'
    ax[0,0].plot(ret_test,  ls='-',  lw=lw, color='black', label=label_teach)
    ax[0,0].axhline(y=np.max(ret_test), color='black', lw=1.0, linestyle='--')

    # Student should be:  <student_exp>/<teacher_base_with_seed>_<seed>/
    # We MUST have `teacher_base` in the directory after <student_exp>.
    sidx = 0
    s_labels, s_labels_std = [], []
    for sd in student_dirs:
        student_subdirs = sorted([osp.join(sd,x) for x in os.listdir(sd) if teacher_base in x])
        if len(student_subdirs) == 0:
            continue
        s_stats_1, s_stats_2 = [], []
        student_stats = []
        terminated_early = 0  # number of seeds that terminated early

        # Now `student_subdirs`, for THIS PARTICULAR student, go through its RANDOM SEEDS.
        # Combine all student runs together with random seeds. Note: smoothing applied BEFORE
        # we append to `student_stats`, and before we take the mean / std for the plot. But,
        # compute desired statistics BEFORE smoothing (the last 10 and the average over all
        # evaluations) because we want to report those numbers in tables [see docs above].
        for s_sub_dir in student_subdirs:
            _, student_data = get_student_stats(s_sub_dir, teacher_seed_dir)

            # DO NOT DO SMOOTHING. Compute statistics M1 (stat_1) and M2 (stat_2).
            perf = student_data['AverageTestEpRet'].to_numpy()
            assert (len(perf) <= 250) or (len(perf) in [2500]), len(perf)
            if len(perf) < 250:
                print(f'Note: for {sd}, len(perf): {len(perf)}')
                terminated_early += 1
            stat_1 = np.mean(perf[-10:])
            stat_2 = np.mean(perf[3:])
            s_stats_1.append(stat_1)
            s_stats_2.append(stat_2)

            # Now smooth and add to student_stats for plotting later.
            student_result = smooth(student_data['AverageTestEpRet'], window)
            student_stats.append(student_result)

        # Extract label which is the <student_exp> not the <teacher_base_with_seed> portion.
        _, tail = os.path.split(sd)
        for name in ENV_NAMES:
            if name in tail:
                tail = tail.replace(f'{name}_', '')
        nb_seeds = len(s_stats_1)
        s_label = f'{tail} (x{nb_seeds}, e{terminated_early})'  # newline due to space
        s_label = s_label.replace('_offline_curriculum_', '_curr_')  # use for no newline
        s_label_std = str(s_label)

        # Shape is (num_seeds, num_recordings=250), usually 250 due to (1M steps = 250 epochs).
        # TODO(daniel) Recording of s_stat_{1,2} might not be intended if we have ragged array.
        s_M1     = np.mean(s_stats_1)  # last 10
        s_M2     = np.mean(s_stats_2)  # all (except first 3)
        s_M1_err = np.std(s_stats_1) / np.sqrt(nb_seeds)  # last 10
        s_M2_err = np.std(s_stats_2) / np.sqrt(nb_seeds)  # all (except first 3)
        s_label     += f'\n\t{s_M1:0.1f}  &  {s_M2:0.1f}'
        s_label_std += f'\n\t{s_M1:0.1f} $\pm$ {s_M1_err:0.1f} & {s_M2:0.1f} $\pm$ {s_M2_err:0.1f}'
        s_labels.append(s_label)
        s_labels_std.append(s_label_std)

        # Actually plot (with error regions if applicable). Standard error of the mean.
        # Also need to update the student label.
        curr_label = get_curriculum_label(args, tail)
        label_curve = f'Student: {curr_label}'
        student_ret, student_std, _ = ragged_array(student_stats)
        x_vals = np.arange(len(student_ret))
        if np.max(x_vals) > 250:
            ax[0,0].axvline(x=250, color='black', lw=1.0, linestyle='--')
        ax[0,0].plot(x_vals, student_ret, lw=lw, color=COLORS[sidx], label=label_curve)
        if nb_seeds > 1:
            ax[0,0].fill_between(x_vals,
                                 student_ret - (student_std / np.sqrt(nb_seeds)),
                                 student_ret + (student_std / np.sqrt(nb_seeds)),
                                 color=COLORS[sidx],
                                 alpha=0.5)
        sidx += 1

    # Print table! Hopefully later in LaTeX code. Do w/ and w/out st-dev (actually, err).
    print('='*100)
    print(t_label)
    print()
    for s_label, s_label_std in zip(s_labels, s_labels_std):
        print(s_label)
        print(s_label_std)
        print()
    print('='*100)

    # Now let's get back to plotting.
    ax[0,0].set_title(f'Teacher and Student Performance ({env_plot})', size=titlesize)
    ax[0,0].set_xlabel('Train Epochs (Online for Teacher, Offline for Student)', size=xsize)
    ax[0,0].set_ylabel('Test Return', size=ysize)
    for r in range(nrows):
        for c in range(ncols):
            leg = ax[r,c].legend(loc="best", ncol=1, prop={'size':legendsize})
            for legobj in leg.legendHandles:
                legobj.set_linewidth(5.0)
            ax[r,c].tick_params(axis='x', labelsize=ticksize)
            ax[r,c].tick_params(axis='y', labelsize=ticksize)
    plt.tight_layout()
    fig_suffix = f'plot_paper_{env_plot}_2500epochs.png'
    figname = polish_figname(teacher_seed_dir, fig_suffix, args)
    plt.savefig(figname)
    print(f'\nSAVED FIGURE: {figname}')


def report_overlap(args, teacher_seed_dir, student_dirs, c_add_only=True, c_scale_only=False):
    """Reporting overlap. See documentation in other plotting methods.
    We should probably plot teacher, student, then overlap?
    Also, as with the Q-value methods, we can use the c_add_only and c_scale_only args.
    """
    assert (c_add_only or c_scale_only) and not (c_add_only and c_scale_only)
    window = args.window
    nrows, ncols = 1, 3
    fig, ax = plt.subplots(nrows, ncols, sharey=False, squeeze=False, figsize=(10*ncols, 7*nrows))
    env_plot = NAME_TO_LABEL[args.name]

    # Load teacher statistics.
    (teacher_data, teacher_base, _, _) = get_teacher_stats(teacher_seed_dir)

    # Plot teacher performance, SMOOTHED for readability. Note: no more teacher labels.
    ret_test = smooth(teacher_data['AverageTestEpRet'], window)
    label_teach = f'Teacher'
    ax[0,0].plot(ret_test,  ls='--',  lw=lw, color='black', label=label_teach)
    min_v = np.min(ret_test)
    max_v = np.max(ret_test)

    # Student should be:  <student_exp>/<teacher_base_with_seed>_<seed>/
    # We MUST have `teacher_base` in the directory after <student_exp>.
    sidx = 0  # student indices
    s_labels, s_labels_std = [], []
    for sd in student_dirs:
        student_subdirs = sorted([osp.join(sd,x) for x in os.listdir(sd) if teacher_base in x])
        if len(student_subdirs) == 0:
            continue
        s_stats_1, s_stats_2 = [], []
        s_rewards, s_overlap = [], []
        terminated_early = 0  # number of seeds that terminated early

        # NEW: use c_add_only and c_scale_only just for this plot.
        if c_add_only:
            if 'logged_scale_' in sd: continue
        else:
            if 'logged_p_' in sd: continue

        # Now `student_subdirs`, for THIS PARTICULAR student, go through its RANDOM SEEDS.
        # Combine all student runs together with random seeds. Note: smoothing applied BEFORE
        # we append to `student_stats`, and before we take the mean / std for the plot. But,
        # compute desired statistics BEFORE smoothing (the last 10 and the average over all
        # evaluations) because we want to report those numbers in tables [see docs above].
        for s_sub_dir in student_subdirs:
            _, student_data = get_student_stats(s_sub_dir, teacher_seed_dir)

            # DO NOT DO SMOOTHING. Compute statistics M1 (stat_1) and M2 (stat_2).
            perf = student_data['AverageTestEpRet'].to_numpy()
            assert (len(perf) <= 250) or (len(perf) in [2500]), len(perf)
            if len(perf) < 250:
                print(f'Note: for {sd}, len(perf): {len(perf)}')
                terminated_early += 1
            stat_1 = np.mean(perf[-10:])
            stat_2 = np.mean(perf[3:])
            s_stats_1.append(stat_1)
            s_stats_2.append(stat_2)

            # Now smooth and add to student_stats for plotting later.
            _s_rewards = smooth(student_data['AverageTestEpRet'], window)
            _s_overlap = smooth(student_data['O_NoActOverlapV'], window)
            s_rewards.append(_s_rewards)
            s_overlap.append(_s_overlap)

        # Extract label which is the <student_exp> not the <teacher_base_with_seed> portion.
        _, tail = os.path.split(sd)
        for name in ENV_NAMES:
            if name in tail:
                tail = tail.replace(f'{name}_', '')
        nb_seeds = len(s_stats_1)
        s_label = f'{tail} (x{nb_seeds}, e{terminated_early})'  # newline due to space
        s_label = s_label.replace('_offline_curriculum_', '_curr_')  # use for no newline
        s_label_std = str(s_label)

        # Shape is (num_seeds, num_recordings=250), usually 250 due to (1M steps = 250 epochs).
        # TODO(daniel) Recording of s_stat_{1,2} might not be intended if we have ragged array.
        s_M1     = np.mean(s_stats_1)  # last 10
        s_M2     = np.mean(s_stats_2)  # all (except first 3)
        s_M1_err = np.std(s_stats_1) / np.sqrt(nb_seeds)  # last 10
        s_M2_err = np.std(s_stats_2) / np.sqrt(nb_seeds)  # all (except first 3)
        s_label     += f'\n\t{s_M1:0.1f}  &  {s_M2:0.1f}'
        s_label_std += f'\n\t{s_M1:0.1f} $\pm$ {s_M1_err:0.1f} & {s_M2:0.1f} $\pm$ {s_M2_err:0.1f}'
        s_labels.append(s_label)
        s_labels_std.append(s_label_std)

        # Actually plot (with error regions if applicable). Standard error of the mean.
        curr_label = get_curriculum_label(args, tail)  # 'Nicer' student label for plots.
        label_curve = f'{curr_label}'
        s_ret, s_std, _ = ragged_array(s_rewards)
        x_vals = np.arange(len(s_ret))
        ax[0,1].plot(x_vals, s_ret, lw=lw, color=COLORS[sidx], label=label_curve)
        if nb_seeds > 1:
            ax[0,1].fill_between(x_vals,
                                 s_ret - (s_std / np.sqrt(nb_seeds)),
                                 s_ret + (s_std / np.sqrt(nb_seeds)),
                                 color=COLORS[sidx],
                                 alpha=0.5)
        min_v = min(min_v, np.min(s_ret))
        max_v = max(max_v, np.max(s_ret))

        # Plot the overlap predictor -- use same color scheme + legend.
        o_vals, o_stds, _ = ragged_array(s_overlap)
        ax[0,2].plot(x_vals, o_vals, lw=lw, color=COLORS[sidx], label=label_curve)
        if nb_seeds > 1:
            ax[0,2].fill_between(x_vals,
                                 o_vals - (o_stds / np.sqrt(nb_seeds)),
                                 o_vals + (o_stds / np.sqrt(nb_seeds)),
                                 color=COLORS[sidx],
                                 alpha=0.5)
        sidx += 1

    # Overlap-specific axes.
    ax[0,0].set_ylim([min_v - 50, max_v + 50])
    ax[0,1].set_ylim([min_v - 50, max_v + 50])
    ax[0,2].set_ylim([0.0, 0.5])

    # Bells and whistles.
    ax[0,0].set_title(f'Teacher ({env_plot})', size=titlesize)
    ax[0,1].set_title(f'Students', size=titlesize)
    ax[0,2].set_title(f'Student-Teacher Overlap', size=titlesize)
    ax[0,0].set_ylabel('Test Return', size=ysize)
    ax[0,1].set_ylabel('Test Return', size=ysize)
    ax[0,2].set_ylabel('Overlap', size=ysize)
    for r in range(nrows):
        for c in range(ncols):
            leg = ax[r,c].legend(loc="best", ncol=1, prop={'size':legendsize})
            for legobj in leg.legendHandles:
                legobj.set_linewidth(5.0)
            ax[r,c].tick_params(axis='x', labelsize=ticksize)
            ax[r,c].tick_params(axis='y', labelsize=ticksize)
            ax[r,c].set_xlabel('Train Epochs', size=xsize)
    plt.tight_layout()
    fig_suffix = f'plot_paper_{env_plot}_overlap.png'
    figname = polish_figname(teacher_seed_dir, fig_suffix, args)
    plt.savefig(figname)
    print(f'\nSAVED FIGURE: {figname}')


def report_qvalues(args, teacher_seed_dir, student_dirs, c_add_only=True, c_scale_only=False):
    """Reporting Q-values. See documentation in other plotting methods.
    We can show both Q1 and Q2, though the values should be similar.

    For this let's actually use `c_add_only` or `c_scale_only`. This will help focus
    ONLY on additive or ONLY on scaling curricula, making plots more readable.
    Will need to track min/max of axes values to make all of them share the same y range,
    except for the top left, which has the return.
    """
    color_q1 = 'darkblue'
    #color_q2 = 'darkgray'
    assert (c_add_only or c_scale_only) and not (c_add_only and c_scale_only)
    window = args.window
    nrows, ncols = 2, 3
    fig, ax = plt.subplots(nrows, ncols, sharey=False, squeeze=False, figsize=(10*ncols, 5*nrows))
    env_plot = NAME_TO_LABEL[args.name]

    # Load teacher statistics. Note: here we only need the `teacher_base` argument.
    (_, teacher_base, _, _) = get_teacher_stats(teacher_seed_dir)

    # Student should be:  <student_exp>/<teacher_base_with_seed>_<seed>/
    # We MUST have `teacher_base` in the directory after <student_exp>.
    sidx = 0            # Colors per row
    rr, cc = 0, 1       # Indices for row and column. Start at row 0 but column 1.
    min_q, max_q = 0, 0 # For matplotlib y axis ranges
    for sd in student_dirs:
        student_subdirs = sorted([osp.join(sd,x) for x in os.listdir(sd) if teacher_base in x])
        if len(student_subdirs) == 0:
            continue
        s_stats_1, s_stats_2 = [], []
        s_rewards, s_q1_vals, s_q2_vals = [], [], []
        terminated_early = 0  # number of seeds that terminated early

        # NEW: use c_add_only and c_scale_only just for this plot.
        if c_add_only:
            if 'logged_scale_' in sd: continue
        else:
            if 'logged_p_' in sd: continue

        # Now `student_subdirs`, for THIS PARTICULAR student, go through its RANDOM SEEDS.
        # Combine all student runs together with random seeds. Note: smoothing applied BEFORE
        # we append to `student_stats`, and before we take the mean / std for the plot. But,
        # compute desired statistics BEFORE smoothing (the last 10 and the average over all
        # evaluations) because we want to report those numbers in tables [see docs above].
        for s_sub_dir in student_subdirs:
            _, student_data = get_student_stats(s_sub_dir, teacher_seed_dir)

            # DO NOT DO SMOOTHING. Compute statistics.
            perf = student_data['AverageTestEpRet'].to_numpy()
            assert (len(perf) <= 250) or (len(perf) in [2500]), len(perf)
            if len(perf) < 250:
                print(f'Note: for {sd}, len(perf): {len(perf)}')
                terminated_early += 1
            stat_1 = np.mean(perf[-10:])
            stat_2 = np.mean(perf[3:])
            s_stats_1.append(stat_1)
            s_stats_2.append(stat_2)

            # Now smooth and add to student_stats for plotting later.
            _s_rewards = smooth(student_data['AverageTestEpRet'], window)
            _s_q1_vals = smooth(student_data['AverageQ1Vals'], window)
            _s_q2_vals = smooth(student_data['AverageQ2Vals'], window)
            s_rewards.append(_s_rewards)
            s_q1_vals.append(_s_q1_vals)
            s_q2_vals.append(_s_q2_vals)
        nb_seeds = len(s_stats_1)

        # Extract label which is the <student_exp> not the <teacher_base_with_seed> portion.
        _, tail = os.path.split(sd)
        for name in ENV_NAMES:
            if name in tail:
                tail = tail.replace(f'{name}_', '')

        # Actually plot (with error regions if applicable). Standard error of the mean.
        curr_label = get_curriculum_label(args, tail)  # 'Nicer' student label for plots.
        label_curve = f'{curr_label}'
        s_ret, s_std, _ = ragged_array(s_rewards)
        x_vals = np.arange(len(s_ret))
        ax[0,0].plot(x_vals, s_ret, lw=lw, color=COLORS[sidx], label=label_curve)
        if nb_seeds > 1:
            ax[0,0].fill_between(x_vals,
                                 s_ret - (s_std / np.sqrt(nb_seeds)),
                                 s_ret + (s_std / np.sqrt(nb_seeds)),
                                 color=COLORS[sidx],
                                 alpha=0.5)

        # Plot Q-values! (Note: can do just q1, it looks almost identical w/q2)
        q1_vals, q1_stds, _ = ragged_array(s_q1_vals)
        #q2_vals, q2_stds, _ = ragged_array(s_q2_vals)
        min_q = min(min_q, np.min(q1_vals))
        max_q = max(max_q, np.max(q1_vals))
        ax[rr,cc].plot(x_vals, q1_vals, lw=lw, color=color_q1)
        #ax[rr,cc].plot(x_vals, q2_vals, lw=lw, color=color_q2)
        if nb_seeds > 1:
            ax[rr,cc].fill_between(x_vals,
                                   q1_vals - (q1_stds / np.sqrt(nb_seeds)),
                                   q1_vals + (q1_stds / np.sqrt(nb_seeds)),
                                   color=color_q1,
                                   alpha=0.5)
            #ax[rr,cc].fill_between(x_vals,
            #                       q2_vals - (q2_stds / np.sqrt(nb_seeds)),
            #                       q2_vals + (q2_stds / np.sqrt(nb_seeds)),
            #                       color=color_q2,
            #                       alpha=0.5)
        ax[rr,cc].set_title(f'{curr_label}', size=titlesize)
        if rr == 1:
            ax[rr,cc].set_xlabel('Train Epochs', size=xsize)
        ax[rr,cc].set_ylabel('Estim. Q-Values', size=ysize)

        # Move on to the next student!
        cc += 1
        if cc == ncols:
            cc = 0
            rr += 1
        sidx += 1

    # Now let's get back to plotting.
    ax[0,0].set_title(f'Students ({env_plot})', size=titlesize)
    #ax[0,0].set_xlabel('Train Epochs', size=xsize)  # Actually we can skip this.
    ax[0,0].set_ylabel('Test Return', size=ysize)
    for r in range(nrows):
        for c in range(ncols):
            ax[r,c].tick_params(axis='x', labelsize=ticksize)
            ax[r,c].tick_params(axis='y', labelsize=ticksize)
            if r == 0 and c == 0:
                leg = ax[r,c].legend(loc="best", ncol=1, prop={'size':legendsize})
                for legobj in leg.legendHandles:
                    legobj.set_linewidth(5.0)
            else:
                # For all but top left, give a little extra. Also, for log scale, it may help if
                # diverging but (a) we wouldn't show those, and (b) avoid setting y_lim for that.
                ax[r,c].set_ylim([min_q, max_q+100])
                #ax[r,c].set_yscale('log')
    plt.tight_layout()
    fig_suffix = f'plot_paper_{env_plot}_qvalue_addonly_{c_add_only}.png'
    figname = polish_figname(teacher_seed_dir, fig_suffix, args)
    plt.savefig(figname)
    print(f'\nSAVED FIGURE: {figname}')


def report_qvalues_and_overlap(args, teacher_seed_dir, student_dirs):
    """Now the combination of these. I think this will be easier to use.

    Also, I don't think we need the teacher label but we can comment out if needed.
    The legend will be the same across all the subplots (except maybe the teacher),
    can we put it to the bottom somewhere?
    """
    window = args.window
    nrows, ncols = 1, 3
    fig, ax = plt.subplots(nrows, ncols, sharey=False, squeeze=False, figsize=(10*ncols, 7*nrows))
    env_plot = NAME_TO_LABEL[args.name]

    # Load teacher statistics.
    (teacher_data, teacher_base, _, _) = get_teacher_stats(teacher_seed_dir)

    # Plot teacher performance, SMOOTHED for readability. Note: no more teacher labels.
    ret_test = smooth(teacher_data['AverageTestEpRet'], window)
    label_teach = f'Teacher'
    ax[0,0].plot(ret_test,  ls='--',  lw=lw, color='black', label=label_teach)
    min_v = np.min(ret_test)
    max_v = np.max(ret_test)

    # Student should be:  <student_exp>/<teacher_base_with_seed>_<seed>/
    # We MUST have `teacher_base` in the directory after <student_exp>.
    sidx = 0  # student indices
    s_labels, s_labels_std = [], []
    for sd in student_dirs:
        student_subdirs = sorted([osp.join(sd,x) for x in os.listdir(sd) if teacher_base in x])
        if len(student_subdirs) == 0:
            continue
        s_stats_1, s_stats_2 = [], []
        s_rewards, s_overlap, s_q1_vals = [], [], []
        terminated_early = 0  # number of seeds that terminated early

        # Now `student_subdirs`, for THIS PARTICULAR student, go through its RANDOM SEEDS.
        # Combine all student runs together with random seeds. Note: smoothing applied BEFORE
        # we append to `student_stats`, and before we take the mean / std for the plot. But,
        # compute desired statistics BEFORE smoothing (the last 10 and the average over all
        # evaluations) because we want to report those numbers in tables [see docs above].
        for s_sub_dir in student_subdirs:
            _, student_data = get_student_stats(s_sub_dir, teacher_seed_dir)

            # DO NOT DO SMOOTHING. Compute statistics M1 (stat_1) and M2 (stat_2).
            perf = student_data['AverageTestEpRet'].to_numpy()
            assert (len(perf) <= 250) or (len(perf) in [2500]), len(perf)
            if len(perf) < 250:
                print(f'Note: for {sd}, len(perf): {len(perf)}')
                terminated_early += 1
            stat_1 = np.mean(perf[-10:])
            stat_2 = np.mean(perf[3:])
            s_stats_1.append(stat_1)
            s_stats_2.append(stat_2)

            # Now smooth and add to student_stats for plotting later.
            _s_rewards = smooth(student_data['AverageTestEpRet'], window)
            _s_overlap = smooth(student_data['O_NoActOverlapV'], window)
            _s_q1_vals = smooth(student_data['AverageQ1Vals'], window)
            s_rewards.append(_s_rewards)
            s_overlap.append(_s_overlap)
            s_q1_vals.append(_s_q1_vals)

        # Extract label which is the <student_exp> not the <teacher_base_with_seed> portion.
        _, tail = os.path.split(sd)
        for name in ENV_NAMES:
            if name in tail:
                tail = tail.replace(f'{name}_', '')
        nb_seeds = len(s_stats_1)
        s_label = f'{tail} (x{nb_seeds}, e{terminated_early})'  # newline due to space
        s_label = s_label.replace('_offline_curriculum_', '_curr_')  # use for no newline
        s_label_std = str(s_label)

        # Shape is (num_seeds, num_recordings=250), usually 250 due to (1M steps = 250 epochs).
        # TODO(daniel) Recording of s_stat_{1,2} might not be intended if we have ragged array.
        s_M1     = np.mean(s_stats_1)  # last 10
        s_M2     = np.mean(s_stats_2)  # all (except first 3)
        s_M1_err = np.std(s_stats_1) / np.sqrt(nb_seeds)  # last 10
        s_M2_err = np.std(s_stats_2) / np.sqrt(nb_seeds)  # all (except first 3)
        s_label     += f'\n\t{s_M1:0.1f}  &  {s_M2:0.1f}'
        s_label_std += f'\n\t{s_M1:0.1f} $\pm$ {s_M1_err:0.1f} & {s_M2:0.1f} $\pm$ {s_M2_err:0.1f}'
        s_labels.append(s_label)
        s_labels_std.append(s_label_std)

        # [PART 1] Actually plot (with error regions if applicable). Standard error of the mean.
        curr_label = get_curriculum_label(args, tail)  # 'Nicer' student label for plots.
        label_curve = f'{curr_label}'
        s_ret, s_std, _ = ragged_array(s_rewards)
        x_vals = np.arange(len(s_ret))
        ax[0,0].plot(x_vals, s_ret, lw=lw, color=COLORS[sidx], label=label_curve)
        if nb_seeds > 1:
            ax[0,0].fill_between(x_vals,
                                 s_ret - (s_std / np.sqrt(nb_seeds)),
                                 s_ret + (s_std / np.sqrt(nb_seeds)),
                                 color=COLORS[sidx],
                                 alpha=0.5)
        min_v = min(min_v, np.min(s_ret))
        max_v = max(max_v, np.max(s_ret))

        # [PART 2] Plot the overlap predictor -- use same colors. Just for this, y-axis limit?
        o_vals, o_stds, _ = ragged_array(s_overlap)
        ax[0,1].plot(x_vals, o_vals, lw=lw, color=COLORS[sidx], label=label_curve)
        if nb_seeds > 1:
            ax[0,1].fill_between(x_vals,
                                 o_vals - (o_stds / np.sqrt(nb_seeds)),
                                 o_vals + (o_stds / np.sqrt(nb_seeds)),
                                 color=COLORS[sidx],
                                 alpha=0.5)
        ax[0,1].set_ylim([0.0, 0.5])

        # [PART 3] Plot the Q-values. Again, use the same colors. Just Q1, since Q2 is very similar.
        q1_vals, q1_stds, _ = ragged_array(s_q1_vals)
        ax[0,2].plot(x_vals, q1_vals, lw=lw, color=COLORS[sidx], label=label_curve)
        if nb_seeds > 1:
            ax[0,2].fill_between(x_vals,
                                 q1_vals - (q1_stds / np.sqrt(nb_seeds)),
                                 q1_vals + (q1_stds / np.sqrt(nb_seeds)),
                                 color=COLORS[sidx],
                                 alpha=0.5)

        # The next student evaluation run, we just did all seeds of this student. :)
        sidx += 1

    # Bells and whistles.
    ax[0,0].set_title(f'Performance ({env_plot})', size=titlesize)
    ax[0,1].set_title(f'Overlap', size=titlesize)
    ax[0,2].set_title(f'Q-Values', size=titlesize)
    ax[0,0].set_ylabel('Episode Return', size=ysize)
    ax[0,1].set_ylabel('Overlap', size=ysize)
    ax[0,2].set_ylabel('Estim. Q-Values', size=ysize)
    for r in range(nrows):
        for c in range(ncols):
            leg = ax[r,c].legend(loc="best", ncol=1, prop={'size':legendsize})
            for legobj in leg.legendHandles:
                legobj.set_linewidth(5.0)
            ax[r,c].tick_params(axis='x', labelsize=ticksize)
            ax[r,c].tick_params(axis='y', labelsize=ticksize)
            ax[r,c].set_xlabel('Train Epochs', size=xsize)
    plt.tight_layout()
    fig_suffix = f'plot_paper_{env_plot}_overlap_and_qvalues.png'
    figname = polish_figname(teacher_seed_dir, fig_suffix, args)
    plt.savefig(figname)
    print(f'\nSAVED FIGURE: {figname}')


def report_qvalues_onerow(args, teacher_seed_dir, student_dirs, c_add_only=True, c_scale_only=False):
    """Reporting Q-values. See documentation in other plotting methods.
    We can show both Q1 and Q2, though the values should be similar.

    For this let's actually use `c_add_only` or `c_scale_only`. This will help focus
    ONLY on additive or ONLY on scaling curricula, making plots more readable.
    Will need to track min/max of axes values to make all of them share the same y range,
    except for the top left, which has the return.

    UPDATE: ONE ROW only. Assumes we have 5, BTW.
    """
    color_q1 = 'darkblue'
    assert (c_add_only or c_scale_only) and not (c_add_only and c_scale_only)
    window = args.window
    nrows, ncols = 1, 5
    _, ax = plt.subplots(nrows, ncols, sharey=True, squeeze=True, figsize=(6*ncols, 5*nrows))
    env_plot = NAME_TO_LABEL[args.name]

    # Load teacher statistics. Note: here we only need the `teacher_base` argument.
    (_, teacher_base, _, _) = get_teacher_stats(teacher_seed_dir)

    # Student should be:  <student_exp>/<teacher_base_with_seed>_<seed>/
    # We MUST have `teacher_base` in the directory after <student_exp>.
    sidx = 0            # Colors per row
    rr, cc = 0, 1       # Indices for row and column. Start at row 0 but column 1.
    min_q, max_q = 0, 0 # For matplotlib y axis ranges
    for sd in student_dirs:
        student_subdirs = sorted([osp.join(sd,x) for x in os.listdir(sd) if teacher_base in x])
        if len(student_subdirs) == 0:
            continue
        s_stats_1, s_stats_2 = [], []
        s_rewards, s_q1_vals, s_q2_vals = [], [], []
        terminated_early = 0  # number of seeds that terminated early

        # NEW: use c_add_only and c_scale_only just for this plot.
        if c_add_only:
            if 'logged_scale_' in sd: continue
        else:
            if 'logged_p_' in sd: continue

        # Now `student_subdirs`, for THIS PARTICULAR student, go through its RANDOM SEEDS.
        # Combine all student runs together with random seeds. Note: smoothing applied BEFORE
        # we append to `student_stats`, and before we take the mean / std for the plot. But,
        # compute desired statistics BEFORE smoothing (the last 10 and the average over all
        # evaluations) because we want to report those numbers in tables [see docs above].
        for s_sub_dir in student_subdirs:
            _, student_data = get_student_stats(s_sub_dir, teacher_seed_dir)

            # DO NOT DO SMOOTHING. Compute statistics.
            perf = student_data['AverageTestEpRet'].to_numpy()
            assert (len(perf) <= 250) or (len(perf) in [2500]), len(perf)
            if len(perf) < 250:
                print(f'Note: for {sd}, len(perf): {len(perf)}')
                terminated_early += 1
            stat_1 = np.mean(perf[-10:])
            stat_2 = np.mean(perf[3:])
            s_stats_1.append(stat_1)
            s_stats_2.append(stat_2)

            # Now smooth and add to student_stats for plotting later.
            _s_rewards = smooth(student_data['AverageTestEpRet'], window)
            _s_q1_vals = smooth(student_data['AverageQ1Vals'], window)
            _s_q2_vals = smooth(student_data['AverageQ2Vals'], window)
            s_rewards.append(_s_rewards)
            s_q1_vals.append(_s_q1_vals)
            s_q2_vals.append(_s_q2_vals)
        nb_seeds = len(s_stats_1)

        # Extract label which is the <student_exp> not the <teacher_base_with_seed> portion.
        _, tail = os.path.split(sd)
        for name in ENV_NAMES:
            if name in tail:
                tail = tail.replace(f'{name}_', '')

        # Actually plot (with error regions if applicable). Standard error of the mean.
        curr_label = get_curriculum_label(args, tail)  # 'Nicer' student label for plots.
        s_ret, _, _ = ragged_array(s_rewards)
        x_vals = np.arange(len(s_ret))

        # Plot Q-values! (Note: can do just q1, it looks almost identical w/q2)
        q1_vals, q1_stds, _ = ragged_array(s_q1_vals)
        min_q = min(min_q, np.min(q1_vals))
        max_q = max(max_q, np.max(q1_vals))
        ax[sidx].plot(x_vals, q1_vals, lw=lw, color=color_q1)
        if nb_seeds > 1:
            ax[sidx].fill_between(x_vals,
                                  q1_vals - (q1_stds / np.sqrt(nb_seeds)),
                                  q1_vals + (q1_stds / np.sqrt(nb_seeds)),
                                  color=color_q1,
                                  alpha=0.5)
        ax[sidx].set_title(f'{curr_label}', size=titlesize)
        ax[sidx].set_xlabel(f'Train Epochs', size=xsize)
        if sidx == 0:
            ax[sidx].set_ylabel('Estim. Q-Values', size=ysize)

        sidx += 1  # Move on to the next student!

    # Now let's get back to plotting. Remember, just one row, so we iterate over columns.
    for c in range(ncols):
        ax[c].tick_params(axis='x', labelsize=ticksize)
        ax[c].tick_params(axis='y', labelsize=ticksize)
        if c == 0:
            leg = ax[c].legend(loc="best", ncol=1, prop={'size':legendsize})
            for legobj in leg.legendHandles:
                legobj.set_linewidth(5.0)
        else:
            # For all but top left, give a little extra. Also, for log scale, it may help if
            # diverging but (a) we wouldn't show those, and (b) avoid setting y_lim for that.
            ax[c].set_ylim([min_q, max_q+100])
            #ax[c].set_yscale('log')
    plt.tight_layout()
    fig_suffix = f'plot_paper_{env_plot}_qvalue_addonly_{c_add_only}_ONEROW.png'
    figname = polish_figname(teacher_seed_dir, fig_suffix, args)
    plt.savefig(figname)
    print(f'\nSAVED FIGURE: {figname}')


if __name__ == '__main__':
    # Note: `add_only_curr` means adding finalb and concurrent, as those are special
    # cases of it, but ignores any reward shaping which would overly-complicate things.
    # Then with `add_only_curr` we should specify which of the curriculums we want to plot.
    # So technically, `add_only_curr` can be replaced with `len(list_currs) > 0`...
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str)
    parser.add_argument('--window', type=int, default=6)
    parser.add_argument('--longer_2500', action='store_true', help='Only finalb')
    parser.add_argument('--add_only_finalb', action='store_true', help='Only finalb')
    parser.add_argument('--add_only_concur', action='store_true', help='Only concurrent')
    parser.add_argument('--add_only_curr', action='store_true', help='Only curriculums')
    parser.add_argument('--add_logged', action='store_true',
        help='Plot finalb or concurrent, but not any gtreward shaping')
    parser.add_argument('--add_naive', action='store_true', help='Plot runs w/naive TD3')
    parser.add_argument('--add_np', action='store_true', help='Plot runs w/noise predictor')
    parser.add_argument('--algorithm', '-algo', default='td3', help='Choose SAC or TD3 students')
    # 0 or more values expected => creates a list to help filter by distribution.
    parser.add_argument('--list_distrs', nargs="*", type=str, default=[])
    parser.add_argument('--list_currs', nargs="*", type=str, default=[])
    parser.add_argument('--exclude_currs', nargs="*", type=str, default=[])
    parser.add_argument('--f_override', type=str, default=None, help='Override plot name')
    parser.add_argument('--qvalues', action='store_true', help='If plotting Q-values')
    parser.add_argument('--qvalues_row', action='store_true', help='If plotting Q-values')
    parser.add_argument('--overlap', action='store_true', help='If plotting overlap')
    args = parser.parse_args()
    assert args.name in ENV_NAMES, f'Error, please use a name in: {ENV_NAMES}'

    # Scan for relevant directories in the DEFAULT_DATA_DIR w/the env_name.
    if not os.path.exists(DDD):
        print(f'{DDD} does not exist! Please check.')
    directories = [osp.join(DDD,x) for x in os.listdir(DDD) if args.name in x]

    # ------------------------------ FILTER TEACHER/STUDENT DIRECTORIES ------------------------------ #
    # Key assumption: 'offline' is in the directory name for Offline RL!
    # Key assumption: each directory within dirs_T is for a random seed.
    dirs_T = sorted([d for d in directories if 'offline_' not in d and 'online_' not in d])
    dirs_S = sorted([d for d in directories if 'offline_' in d and 'online_' not in d])
    assert args.algorithm in ['td3', 'sac'], "Only supports TD3 and SAC for now"
    algo_str = args.algorithm
    # Mandi: add another filter for algorithm type
    dirs_T = sorted([d for d in dirs_T if algo_str in d])
    dirs_S = sorted([d for d in dirs_S if algo_str in d])

    # Further filter the student directory. It's a bit clumsy.
    new_dirs_S = []
    for d in dirs_S:
        add0, add1, add2, add3, add4, add5 = False, False, False, False, False, False
        if args.add_logged:
            # Add either final buffer or concurrent settings, but prob not both.
            add0 = (algo_str+'_offline_finalb' in d or algo_str+'_offline_concurrent' in d)
            add0 = add0 and ('gtrewshape' not in d)
            assert not (args.add_only_finalb or args.add_only_concur)
        if args.add_np:
            # For noise predictor, we added: `_np_[nalpha]`
            add2 = '_np_' in d
        if args.add_only_finalb:
            add3 = algo_str+'_offline_finalb' in d
            add3 = add3 and ('gtrewshape' not in d)
        if args.add_only_concur:
            add4 = algo_str+'_offline_concurrent' in d
            add4 = add4 and ('gtrewshape' not in d)

        if args.add_only_curr:
            # As of March 09 2021, we use a specific naming convention for curricula.
            # Use: '<env>_td3_offline_curriculum_logged<...>'. Then, add specified
            # curricula on demand, which will override the add5 setting we had earlier.
            #add5 = ('td3_offline_concurrent' in d or 'td3_offline_finalb' in d)  # old way
            #add5 = ('td3_offline_curriculum_logged' in d)  # actually let's not do this

            # Step 1: should we ignore this? If so, no need to adjust add5 (it's False).
            # I'm adding a tiny condition to ensure the last few characters match, this is
            # due to matching _tp_1 with _tp_10 when it wasn't intended.
            exclude = False
            if len(args.exclude_currs) > 0:
                for curriculum in args.exclude_currs:
                    if (curriculum in d) and (curriculum[-3:] == d[-3:]):
                        print(f'EXCLUDE: {curriculum} in {d}')
                        exclude = True

            # Step 2: go through list of currs and see if we want to add.
            if len(args.list_currs) > 0 and (not exclude):
                is_in_distr = False
                for curriculum in args.list_currs:
                    if curriculum in d:
                        is_in_distr = True
                        break
                if is_in_distr:
                    add5 = True

            # Step 3: do a few more checks on `add5`.
            if not exclude:
                add5 = add5 and ('gtrewshape' not in d) and ('_np_' not in d)

        if (add0 or add1 or add2 or add3 or add4 or add5):
            new_dirs_S.append(d)
    dirs_S = new_dirs_S  # override

    # ANOTHER filter, if using `list_distrs`, only consider distributions in this list.
    # Be careful if any distributions are nested within each other.
    if len(args.list_distrs) > 0:
        new_dirs_S = []
        for d in dirs_S:
            is_in_distr = False
            for distr in args.list_distrs:
                if distr in d:
                    is_in_distr = True
                    break
            if is_in_distr:
                new_dirs_S.append(d)
        dirs_S = new_dirs_S  # override

    # Re-sort based on padding zeros, etc.
    dirs_S = get_dirs_S_sorted(dirs_S)

    # BTW we don't filter the teachers this way since their directories are structured differently.
    # dirs_T is usually short, just has `env_td3_act0-1` or something like that. The teachers are
    # buffers within those directories, so we have to filter in a different way, we can do it later.
    # ------------------------------ END OF TEACHER/STUDENT FILTER ------------------------------ #

    # Find random seeds of the TEACHERS. Normally we did one (w/more for students).
    dirs_T_seeds = []
    for dir_t in dirs_T:
        seeds = sorted(
            [osp.join(dir_t,x) for x in os.listdir(dir_t) if 'figures' not in x])
        seeds = [x.split('_')[-1] for x in seeds]  # parse to keep 'sXYZ'
        dirs_T_seeds.append(seeds)

    # Debugging.
    print('\n\n')
    print('*'*120)
    print('*'*120)
    print(f'Calling plot_offline_rl with env: {args.name}. Relevant teachers:')
    for d, ds in zip(dirs_T, dirs_T_seeds):
        print(f'\t{d}, seeds: {ds}')
    print(f'\nRelevant Student directories:')
    for d in dirs_S:
        print(f'\t{d}')
    print('*'*120)
    print('*'*120)

    # Iterate through teachers, and then seeds within the teacher.
    for d, t_seeds in zip(dirs_T, dirs_T_seeds):
        for seed in t_seeds:
            print('\n')
            print('-'*100)
            print(f'PLOTTING TEACHER  {d}  w/seed {seed}')
            print('-'*100)

            # Derive teacher path, `teacher_seed_dir`, for this seed. Then plot.
            teacher_seed_dir = [osp.join(d,x) for x in os.listdir(d) if seed in x]
            assert len(teacher_seed_dir) == 1, teacher_seed_dir
            teacher_seed_dir = teacher_seed_dir[0]

            # I think a table is easier to summarize in a paper. But do a plot as well.
            if args.longer_2500:
                report_table_plot_2500(args, teacher_seed_dir=teacher_seed_dir, student_dirs=dirs_S)
            elif args.qvalues and args.overlap:
                # First let's do this COMBINED one. Then if not both, we can proceed to others.
                report_qvalues_and_overlap(args, teacher_seed_dir=teacher_seed_dir, student_dirs=dirs_S)
            elif args.qvalues_row:
                report_qvalues_onerow(args, teacher_seed_dir=teacher_seed_dir, student_dirs=dirs_S)
            elif args.qvalues:
                report_qvalues(args, teacher_seed_dir=teacher_seed_dir, student_dirs=dirs_S)
            elif args.overlap:
                report_overlap(args, teacher_seed_dir=teacher_seed_dir, student_dirs=dirs_S)
            else:
                report_table_plot(args, teacher_seed_dir=teacher_seed_dir, student_dirs=dirs_S)
