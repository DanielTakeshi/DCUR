"""Plot the ONLINE RL runs.

Actually we might as well combine the overlap and the TP here, I don't want to keep making
a ton of different scripts. Maybe adjust this for the offline plots?
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import pandas as pd
from copy import deepcopy
import os
import os.path as osp
import numpy as np
import json
from spinup.user_config import DEFAULT_DATA_DIR as DDD
from spinup.teaching.plot_offline_rl import (
    get_dirs_S_sorted, smooth, sanity_checks, ragged_array, adjust_fig_suffix,
    COLORS, ENV_NAMES,
)
from spinup.teaching.plot_paper import (
    get_curriculum_label, polish_figname, get_teacher_stats, get_student_stats,
    NAME_TO_LABEL,
)
np.set_printoptions(linewidth=180, suppress=True)

# Matplotlib stuff -- NOTE for `plot_paper_fig` put it in the method.
titlesize = 32
xsize = 30
ysize = 30
ticksize = 28
legendsize = 17
er_alpha = 0.25
lw = 3


def plot(args, teacher_seed_dir, student_dirs):
    """See docs in `plot_offline_rl.py`."""
    window = args.window
    nrows, ncols = 1, 2
    fig, ax = plt.subplots(nrows, ncols, sharey=True, squeeze=False, figsize=(11*ncols, 8*nrows))
    #fig, ax = plt.subplots(nrows, ncols, sharey=True, squeeze=False, figsize=(11*ncols, 8*nrows),
    #        gridspec_kw={'width_ratios': [1, 1.5]})  # Use if we want to scale student plot longer.

    # Load teacher statistics.
    (teacher_data, teacher_base, _, _) = get_teacher_stats(teacher_seed_dir)

    # Teacher performance. Test performance matches spinup's plots, so that's good.
    # Let's also do the two stats that we compute for the student as well.
    t_perf_np   = teacher_data['AverageTestEpRet'].to_numpy()
    t_stat_1    = np.mean(t_perf_np[-10:])
    t_stat_2    = np.mean(t_perf_np[3:])
    ret_train   = smooth(teacher_data['AverageEpRet'],     window)
    ret_test    = smooth(teacher_data['AverageTestEpRet'], window)
    label_train = f'{teacher_base} (Train, {ret_train[-1]:0.1f})'
    label_test  = f'{teacher_base} (Test)\nlast: {ret_test[-1]:0.1f}'
    label_test  = f'{label_test}, M1: {t_stat_1:0.1f}, M2: {t_stat_2:0.1f}'
    ax[0,0].plot(ret_train, lw=lw, color=COLORS[0], label=label_train)
    ax[0,0].plot(ret_test,  lw=lw, color=COLORS[1], label=label_test)

    # NOW deal with students for this particular teacher, or any of its buffers [?].
    # Student should be:  <student_exp>/<teacher_base_with_seed>_<seed>/
    # We MUST have `teacher_base` in the directory after <student_exp>.
    sidx = 0
    for sd in student_dirs:
        student_subdirs = sorted([osp.join(sd,x) for x in os.listdir(sd) if teacher_base in x])
        if len(student_subdirs) == 0:
            continue
        print(f'\nStudent:  {sd}  has seeds:')
        student_stats = []
        student_stats_1 = []
        student_stats_2 = []
        terminated_early = 0  # number of seeds that terminated early

        # Now `student_subdirs`, for THIS PARTICULAR student, go through its RANDOM SEEDS.
        # Combine all student runs together with random seeds. Note: smoothing applied BEFORE
        # we append to `student_stats`, and before we take the mean / std for the plot. But,
        # compute desired statistics BEFORE smoothing (the last 10 and the average over all
        # evaluations) because we want to report those numbers in tables [see docs above].
        for s_sub_dir in student_subdirs:
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

            # BEFORE smoothing (critical!) for plots, compute stats to report [see docs above].
            perf = student_data['AverageTestEpRet'].to_numpy()
            assert (len(perf) <= 250) or (len(perf) in [500,1000,2500]), len(perf)
            if len(perf) < 250:
                print(f'Note: for {sd}, len(perf): {len(perf)}')
                terminated_early += 1
            stat_1 = np.mean(perf[-10:])
            stat_2 = np.mean(perf[3:])
            student_stats_1.append(stat_1)
            student_stats_2.append(stat_2)

            # Now smooth and add to student_stats for plotting later.
            student_result = smooth(student_data['AverageTestEpRet'], window)
            student_stats.append(student_result)

        # Extract label which is the <student_exp> not the <teacher_base_with_seed> portion.
        # However, we probably don't need the env name in `tail` as that's in the title.
        head, tail = os.path.split(sd)
        for name in ENV_NAMES:
            if name in tail:
                tail = tail.replace(f'{name}_', '')
        nb_seeds = len(student_stats)
        s_label = f'{tail}\n(x{nb_seeds}, e{terminated_early}) '  # newline due to space

        # If `tail` has curriculum_ in name, might want to shorten legend names.
        s_label = s_label.replace('_curriculum_', '_curr\n')  # use for newline

        # Shape is (num_seeds, num_recordings=250), usually 250 due to (1M steps = 250 epochs).
        # TODO(daniel) Recording of s_stat_{1,2} might not be intended if we have ragged array.
        student_ret, student_std, ragged = ragged_array(student_stats)
        s_stat_1 = np.mean(student_stats_1)  # last 10
        s_stat_2 = np.mean(student_stats_2)  # all (except first 3)
        s_label += f'last: {student_ret[-1]:0.1f}, stat1: {s_stat_1:0.1f}, stat2: {s_stat_2:0.1f}'
        s_label = s_label.replace('offline_', 'off_')

        # Actually plot (with error regions if applicable). Standard error of mean.
        x_vals = np.arange(len(student_ret))
        if np.max(x_vals) > 250:
            ax[0,1].axvline(x=250, color='black', lw=0.5, linestyle='--')
        ax[0,1].plot(x_vals, student_ret, lw=lw, color=COLORS[sidx], label=s_label)
        if nb_seeds > 1:
            ax[0,1].fill_between(x_vals,
                                 student_ret - (student_std / np.sqrt(nb_seeds)),
                                 student_ret + (student_std / np.sqrt(nb_seeds)),
                                 color=COLORS[sidx],
                                 alpha=0.5)
        sidx += 1

    # Bells and whistles.
    ax[0,0].set_title(f'Teacher {args.name}', size=titlesize)
    ax[0,1].set_title(f'Students {args.name}, w={args.window}', size=titlesize)
    ax[0,0].set_xlabel('Train Epochs', size=xsize)
    ax[0,1].set_xlabel('Train Epochs', size=xsize)
    ax[0,0].set_ylabel('TestEpReturn', size=ysize)
    ax[0,1].set_ylabel('TestEpReturn', size=ysize)
    for r in range(nrows):
        for c in range(ncols):
            leg = ax[r,c].legend(loc="best", ncol=1, prop={'size':legendsize})
            for legobj in leg.legendHandles:
                legobj.set_linewidth(5.0)
            ax[r,c].tick_params(axis='x', labelsize=ticksize)
            ax[r,c].tick_params(axis='y', labelsize=ticksize)
    plt.tight_layout()
    fig_suffix = 'plot_online_rl.png'
    fig_suffix = adjust_fig_suffix(fig_suffix, args)
    figname = osp.join(teacher_seed_dir, fig_suffix)
    plt.savefig(figname)
    print(f'\nSAVED FIGURE: {figname}')


def plot_with_overlap(args, teacher_seed_dir, student_dirs):
    """WITH OVERLAP. Like `plot_offline_rl_with_Overlap.py`."""
    window = args.window
    nrows, ncols = 1, 3
    fig, ax = plt.subplots(nrows, ncols, sharey=False, squeeze=False,
            figsize=(11*ncols, 8*nrows), gridspec_kw={'width_ratios': [1.0, 1.0, 1.0]})

    # Load teacher statistics.
    (teacher_data, teacher_base, _, _) = get_teacher_stats(teacher_seed_dir)

    # Teacher performance. Test performance matches spinup's plots, so that's good.
    # Let's also do the two stats that we compute for the student as well.
    t_perf_np   = teacher_data['AverageTestEpRet'].to_numpy()
    t_stat_1    = np.mean(t_perf_np[-10:])
    t_stat_2    = np.mean(t_perf_np[3:])
    ret_train   = smooth(teacher_data['AverageEpRet'],     window)
    ret_test    = smooth(teacher_data['AverageTestEpRet'], window)
    label_train = f'{teacher_base} (Train, {ret_train[-1]:0.1f})'
    label_test  = f'{teacher_base} (Test)\nlast: {ret_test[-1]:0.1f}'
    label_test  = f'{label_test}, M1: {t_stat_1:0.1f}, M2: {t_stat_2:0.1f}'
    ax[0,0].plot(ret_train, lw=lw, color=COLORS[0], label=label_train)
    ax[0,0].plot(ret_test,  lw=lw, color=COLORS[1], label=label_test)

    # NOW deal with students for this particular teacher, or any of its buffers [?].
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
        student_OLAP = []
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
            assert (len(perf) <= 250) or (len(perf) in [500,1000,2500]), len(perf)
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

            # Get info of overlap on student rollouts.
            student_olap = smooth(student_data['O_NoActOverlapV'], window)
            student_OLAP.append(student_olap)

        # Extract label which is the <student_exp> not the <teacher_base_with_seed> portion.
        # However, we probably don't need the env name in `tail` as that's in the title.
        _, tail = os.path.split(sd)
        for name in ENV_NAMES:
            if name in tail:
                tail = tail.replace(f'{name}_', '')
        nb_seeds = len(s_stats_1)
        s_label = f'{tail} (x{nb_seeds}, e{terminated_early})\n'  # newline due to space
        s_label = s_label.replace('_curriculum_', '_curr')  # use for newline
        s_label_std = str(s_label)
        o_label = str(s_label)  # Overlap

        # Shape is (num_seeds, num_recordings=250), usually 250 due to (1M steps = 250 epochs).
        # TODO(daniel) Recording of s_stat_{1,2} might not be intended if we have ragged array.
        s_M1     = np.mean(s_stats_1)  # last 10
        s_M2     = np.mean(s_stats_2)  # all (except first 3)
        s_M1_err = np.std(s_stats_1) / np.sqrt(nb_seeds)  # last 10
        s_M2_err = np.std(s_stats_2) / np.sqrt(nb_seeds)  # all (except first 3)
        _s_label    = f'{s_label} M1: {s_M1:0.1f}, M2: {s_M2:0.1f}'    # for plot
        s_label     += f'\t{s_M1:0.1f} & {s_M2:0.1f}'                 # for printing
        s_label_std += f'\t{s_M1:0.1f} $\pm$ {s_M1_err:0.1f} & {s_M2:0.1f} $\pm$ {s_M2_err:0.1f}'
        s_labels.append(s_label)
        s_labels_std.append(s_label_std)

        # Actually plot (with error regions if applicable). Standard error of mean.
        student_ret, student_std, _ = ragged_array(student_stats)
        x_vals = np.arange(len(student_ret))
        if np.max(x_vals) > 250:
            ax[0,1].axvline(x=250, color='black', lw=0.5, linestyle='--')
        _s_label = _s_label.replace('curr', 'curr\n')
        ax[0,1].plot(x_vals, student_ret, lw=lw, color=COLORS[sidx], label=_s_label)
        if nb_seeds > 1:
            ax[0,1].fill_between(x_vals,
                                 student_ret - (student_std / np.sqrt(nb_seeds)),
                                 student_ret + (student_std / np.sqrt(nb_seeds)),
                                 color=COLORS[sidx],
                                 alpha=0.5)

        # (Unique here) plot the overlap predictor.
        o_vals, o_std, _ = ragged_array(student_OLAP)
        o_label += f'last OLAP: {o_vals[-1]:0.2f}, max OLAP: {np.max(o_vals):0.2f}'
        o_label = o_label.replace('curr', 'curr\n')
        if np.max(x_vals) > 250:
            ax[0,2].axvline(x=250, color='black', lw=0.5, linestyle='--')
        ax[0,2].plot(x_vals, o_vals, lw=lw, color=COLORS[sidx], label=o_label)
        if nb_seeds > 1:
            ax[0,2].fill_between(x_vals,
                                 o_vals - (o_std / np.sqrt(nb_seeds)),
                                 o_vals + (o_std / np.sqrt(nb_seeds)),
                                 color=COLORS[sidx],
                                 alpha=0.5)
        ax[0,2].set_ylim([-0.05, 1.05])

        sidx += 1  # Move on to the next student.

    # Print table! Hopefully later in LaTeX code. Do w/ and w/out st-dev (actually, err).
    print('='*100)
    for s_label, s_label_std in zip(s_labels, s_labels_std):
        print(s_label)
        print(s_label_std)
        print()
    print('='*100)

    # Bells and whistles.
    ymin0, ymax0 = ax[0,0].get_ylim()
    ymin1, ymax1 = ax[0,1].get_ylim()
    ax[0,0].set_ylim( [min(ymin0,ymin1), max(ymax0,ymax1)] )
    ax[0,1].set_ylim( [min(ymin0,ymin1), max(ymax0,ymax1)] )
    ax[0,0].set_title(f'Teacher {args.name}', size=titlesize)
    ax[0,1].set_title(f'Students {args.name}, w={args.window}', size=titlesize)
    ax[0,2].set_title(f'Overlap Computation', size=titlesize)
    ax[0,0].set_xlabel('Train Epochs', size=xsize)
    ax[0,1].set_xlabel('Train Epochs', size=xsize)
    ax[0,2].set_xlabel('Train Epochs', size=xsize)
    ax[0,0].set_ylabel('TestEpReturn', size=ysize)
    ax[0,1].set_ylabel('TestEpReturn', size=ysize)
    ax[0,2].set_ylabel('Overlap Value', size=ysize)
    for r in range(nrows):
        for c in range(ncols):
            leg = ax[r,c].legend(loc="best", ncol=1, prop={'size':legendsize})
            for legobj in leg.legendHandles:
                legobj.set_linewidth(5.0)
            ax[r,c].tick_params(axis='x', labelsize=ticksize)
            ax[r,c].tick_params(axis='y', labelsize=ticksize)
    plt.tight_layout()
    fig_suffix = 'plot_online_rl_woveralp.png'
    figname = polish_figname(teacher_seed_dir, fig_suffix, args)
    plt.savefig(figname)
    print(f'\nSAVED FIGURE: {figname}')


def plot_paper_fig(args, teacher_seed_dir, student_dirs):
    """Let's try and get a nice figure that's publication-quality."""
    window = args.window

    # Override what we normally have.
    titlesize = 43
    xsize = 38
    ysize = 38
    ticksize = 33
    legendsize = 33
    lw = 3

    # Make the figure.
    nrows, ncols = 1, 2
    fig, ax = plt.subplots(nrows, ncols, sharey=True, squeeze=False,
            figsize=(12*ncols, 8*nrows), gridspec_kw={'width_ratios': [1.0, 1.0]})

    # Teacher performance.
    (teacher_data, teacher_base, _, t_label) = get_teacher_stats(teacher_seed_dir)
    ret_test = smooth(teacher_data['AverageTestEpRet'], window)
    ax[0,0].plot(ret_test, ls='--', lw=lw, color='black', label='Teacher')
    ax[0,1].plot(ret_test, ls='--', lw=lw, color='black', label='Teacher')

    # NOW deal with students for this particular teacher, or any of its buffers [?].
    # Student should be:  <student_exp>/<teacher_base_with_seed>_<seed>/
    # We MUST have `teacher_base` in the directory after <student_exp>.
    sidx = 0
    c1 = 0
    c2 = 0
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
            assert (len(perf) <= 250) or (len(perf) in [500,1000,2500]), len(perf)
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
        # However, we probably don't need the env name in `tail` as that's in the title.
        _, tail = os.path.split(sd)
        for name in ENV_NAMES:
            if name in tail:
                tail = tail.replace(f'{name}_', '')
        nb_seeds = len(s_stats_1)
        assert nb_seeds == 5, nb_seeds
        s_label = f'{tail} (x{nb_seeds}, e{terminated_early})\n'  # newline due to space
        s_label_std = str(s_label)

        # Shape is (num_seeds, num_recordings=250), usually 250 due to (1M steps = 250 epochs).
        # TODO(daniel) Recording of s_stat_{1,2} might not be intended if we have ragged array.
        s_M1     = np.mean(s_stats_1)  # last 10
        s_M2     = np.mean(s_stats_2)  # all (except first 3)
        s_M1_err = np.std(s_stats_1) / np.sqrt(nb_seeds)  # last 10
        s_M2_err = np.std(s_stats_2) / np.sqrt(nb_seeds)  # all (except first 3)
        s_label     += f'\t{s_M1:0.1f} & {s_M2:0.1f}'                 # for printing
        s_label_std += f'\t{s_M1:0.1f} $\pm$ {s_M1_err:0.1f} & {s_M2:0.1f} $\pm$ {s_M2_err:0.1f}'
        s_labels.append(s_label)
        s_labels_std.append(s_label_std)

        # [LEFT] Plot final buffer; [RIGHT] Plot concurrent.
        if 'p_1000000_n_1000000_' in s_label:
            cidx = c1
            c = 0
        else:
            cidx = c2
            c = 1

        # Actually plot.
        s_ret, s_std, _ = ragged_array(student_stats)
        x_vals = np.arange(len(s_ret))
        if 'offline' in tail:
            curr_label = '0.0%'
        else:
            curr_label = get_curriculum_label(args, tail)
        ax[0,c].plot(x_vals, s_ret, lw=lw, color=COLORS[cidx], label=curr_label)
        if nb_seeds > 1:
            ax[0,c].fill_between(x_vals,
                                 s_ret - (s_std / np.sqrt(nb_seeds)),
                                 s_ret + (s_std / np.sqrt(nb_seeds)),
                                 color=COLORS[cidx],
                                 alpha=0.5)

        # Move on to the next student.
        if 'p_1000000_n_1000000_' in s_label:
            c1 += 1
        else:
            c2 += 1
        sidx += 1

    # Print table! Hopefully later in LaTeX code. Do w/ and w/out st-dev (actually, err).
    print('='*100)
    print(f'{t_label}\n')
    for s_label, s_label_std in zip(s_labels, s_labels_std):
        print(s_label)
        print(s_label_std)
        print()
    print('='*100)

    # Bells and whistles.
    _title = NAME_TO_LABEL[args.name]
    ax[0,0].set_title(f'{_title}; C_add(t; f=1M)', size=titlesize)
    ax[0,1].set_title(f'{_title}; C_scale(t; c=1.00)', size=titlesize)
    ax[0,0].set_xlabel('Train Epochs', size=xsize)
    ax[0,1].set_xlabel('Train Epochs', size=xsize)
    ax[0,0].set_ylabel('TestEpReturn', size=ysize)
    for r in range(nrows):
        for c in range(ncols):
            leg = ax[r,c].legend(loc="best", ncol=2, prop={'size':legendsize})
            for legobj in leg.legendHandles:
                legobj.set_linewidth(5.0)
            ax[r,c].tick_params(axis='x', labelsize=ticksize)
            ax[r,c].tick_params(axis='y', labelsize=ticksize)
    plt.tight_layout()
    fig_suffix = f'plot_online_paper_fig_{args.name}.png'
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
    parser.add_argument('--with_overlap', action='store_true', help='Add overlap to subplots')
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
    parser.add_argument('--paper_fig', action='store_true', help='Plot figures for paper')
    args = parser.parse_args()
    assert args.name in ENV_NAMES, f'Error, please use a name in: {ENV_NAMES}'

    # Scan for relevant directories in the DEFAULT_DATA_DIR w/the env_name.
    if not os.path.exists(DDD):
        print(f'{DDD} does not exist! Please check.')
    directories = [osp.join(DDD,x) for x in os.listdir(DDD) if args.name in x]

    # ------------------------------ FILTER TEACHER/STUDENT DIRECTORIES ------------------------------ #
    # Key assumption: 'online_stud' is in the directory name for students (UNLIKE Offline RL).
    # Key assumption: each directory within dirs_T is for a random seed.
    dirs_T = sorted([d for d in directories if 'online_stud_' not in d and 'offline_' not in d])
    dirs_S = sorted([d for d in directories if 'online_stud_' in d or
            'offline_curriculum_ep_250_logged_p_1000000_n_1000000' in d or
            'offline_curriculum_ep_250_logged_scale_1.00t_overlap' in d])
            # I think we want these for a baseline?
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
            if args.paper_fig:
                plot_paper_fig(args, teacher_seed_dir=teacher_seed_dir, student_dirs=dirs_S)
            elif args.with_overlap:
                plot_with_overlap(args, teacher_seed_dir=teacher_seed_dir, student_dirs=dirs_S)
            else:
                plot(args, teacher_seed_dir=teacher_seed_dir, student_dirs=dirs_S)
