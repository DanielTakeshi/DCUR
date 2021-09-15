"""Plot Offline RL."""
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
import itertools
import numpy as np
import sys
import gym
import json
import time
import pickle
from spinup.user_config import DEFAULT_DATA_DIR as DDD

# Matplotlib stuff
titlesize = 32
xsize = 30
ysize = 30
ticksize = 28
legendsize = 23
er_alpha = 0.25
lw = 3
COLORS = ['red', 'blue', 'yellow', 'cyan', 'purple', 'black', 'brown', 'pink',
    'silver', 'green', 'darkblue', 'orange']
COLORS_LINE = list(COLORS)

# Env Names
ENV_NAMES = ['ant', 'halfcheetah', 'hopper', 'walker2d']


def parse_to_get_data_type(pt_file):
    """Parse the .txt file to get the data type."""
    base = os.path.basename(pt_file)
    parsed = base.replace('-dtype-train.txt','')
    parsed = parsed.split('-')
    for idx,item in enumerate(parsed):
        if item == 'noise':
            return parsed[idx+1]  # The NEXT item has noise type.
    print(f'Something went wrong: {pt_file}')
    sys.exit()


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


def sanity_checks(config, progress, teacher_path):
    """Some sanity checks on config.json and progress.txt, etc.

    This will act as an extra layer of protection in case we have scp errors.
    Remember that `config` was saved as we ran Offline RL. And that we have to
    consider different top-level directories. But the first instance of 'data'
    if it is not actually /data? If this is causing problems, just ignore. TBH,
    better to have these checks in the original offline RL running.
    """
    assert config['buffer_path'][0] == '/', config['buffer_path']
    assert teacher_path[0] == '/', teacher_path

    # Remove everything up to the first instance of 'data'.
    def remove_leading_paths(path):
        path_split = path.split('/')  # ['', 'data', ...]
        for i in range(2, len(path_split)):
            if path_split[i] == 'data':
                return '/'.join(path_split[i:])
        print(f'Something wrong happened: {path}')
        sys.exit()

    # The buffer path should be within the teacher path.
    buffer_path = remove_leading_paths(config['buffer_path'])
    teacher_path = remove_leading_paths(teacher_path)
    assert teacher_path in buffer_path, f'{teacher_path} not in {buffer_path}'


def plot(args, teacher_seed_dir, student_dirs):
    """We fix the teacher, and plot performance of it, and its students.

    To cycle through images, look at all teacher directories, or anything that
    does NOT have 'offline' in its name -- those are students. I'm including sanity
    checks on the student directories to catch any potential copy / paste errors.

    For teachers, we first consider a directory which stores teacher informations, with
    subdirs corresponding to seeds. Example: '/data/spinup/data/halfcheetah_td3_act0-1'
    means HalfCheetah TD3 teacher, trained with act_noise=0.1. we then consider a seed
    inside, e.g., 'halfcheetah_td3_act0-1_s10' (for seed 10) and that's `teacher_seed_dir`.
    There may be multiple ways we generated data buffers for that teacher, so plot
    student performance together to compare.

    Args:
        teacher_seed_dir: (str) see description above.
        student_dirs: (list) student directories that potentially should be in the plot.
            These are of the form:  <student_exp>/<teacher_base_with_seed>_<seed>/
            This list contains all with env name matching that of the teacher, then
            we filter to check (a) which teacher we used, then (b) which of these
            students have data buffers from (a).
    """
    window = args.window
    nrows, ncols = 1, 2
    fig, ax = plt.subplots(nrows, ncols, sharey=True, squeeze=False, figsize=(11*ncols, 8*nrows))

    # Derive original Online DeepRL teacher results from 'progress.txt'.
    prog_file = osp.join(teacher_seed_dir, 'progress.txt')
    assert os.path.exists(prog_file), prog_file
    teacher_data = pd.read_table(prog_file)
    teacher_base = os.path.basename(teacher_seed_dir)

    # Teacher performance. Test performance matches spinup's plots, so that's good.
    ret_train   = smooth(teacher_data['AverageEpRet'],     window)
    ret_test    = smooth(teacher_data['AverageTestEpRet'], window)
    label_train = f'{teacher_base} (Train)'
    label_test  = f'{teacher_base} (Test)'
    ax[0,0].plot(ret_train, lw=lw, color=COLORS[0], label=label_train)
    ax[0,0].plot(ret_test,  lw=lw, color=COLORS[1], label=label_test)

    # Next: consider other buffers we ran, these are from the last snapshot, and where
    # we roll out the policies. Ignore any 'valid' files. These .txt files have only
    # one row of statistics, so use `[...].iloc[0]` to get the row, then index by key.
    # NOTE: on Jan 23, I switched to saving buffers in the same 'buffer' directory.
    teacher_buf_dir1 = osp.join(teacher_seed_dir, 'buffer')
    teacher_buf_dir2 = osp.join(teacher_seed_dir, 'rollout_buffer_txts')
    if os.path.exists(teacher_buf_dir1):
        teacher_buffers1 = sorted([osp.join(teacher_buf_dir1, x)
                for x in os.listdir(teacher_buf_dir1) if 'valid' not in x and '.txt' in x])
    else:
        teacher_buffers1 = []
    if os.path.exists(teacher_buf_dir2):
        teacher_buffers2 = sorted([osp.join(teacher_buf_dir2, x)
                for x in os.listdir(teacher_buf_dir2) if 'valid' not in x and '.txt' in x])
    else:
        teacher_buffers2 = []
    teacher_buffers = sorted(teacher_buffers1 + teacher_buffers2)

    if len(teacher_buffers) > 0:
        # Filter teacher buffers so we only get most relevant one to the left subplot.
        if len(args.list_distrs) > 0:
            new_bufs_T = []
            for d in teacher_buffers:
                is_in_distr = False
                for distr in args.list_distrs:
                    if distr in d:
                        is_in_distr = True
                        break
                if is_in_distr:
                    new_bufs_T.append(d)
            teacher_buffers = new_bufs_T  # override
        print('Buffers:')

        # Horizontal dashed line for datasets. Will need to parse the noise description.
        # This is the DATA-GENERATING POLICY performance and good to sanity check.
        for tidx,tb in enumerate(teacher_buffers):
            print(f'\t{tb}')
            data_type = parse_to_get_data_type(tb)
            tb_data = pd.read_table(tb)
            buf_ret = tb_data.iloc[0]['AverageEpRet']
            label = f'{data_type} [{buf_ret:0.0f}]'
            if tidx < len(COLORS_LINE):
                ax[0,0].axhline(buf_ret, ls='dashdot', lw=lw, color=COLORS_LINE[tidx], label=label)
            else:
                print(f'Skipping teacher {tb} due to too many buffers')

    # NOW deal with students for this particular teacher, or any of its buffers [?].
    # Student should be:  <student_exp>/<teacher_base_with_seed>_<seed>/
    # We MUST have `teacher_base` in the directory after <student_exp>.
    sidx = 0
    for sd in student_dirs:
        student_subdirs = sorted([osp.join(sd,x) for x in os.listdir(sd) if teacher_base in x])
        if len(student_subdirs) == 0:
            continue

        # Now `student_subdirs`, for THIS PARTICULAR, student, should only vary w/random seed.
        # Combine all student runs together with random seeds. Note: smoothing applied BEFORE
        # we append to `student_stats`, and before we take the mean / std for the plot.
        print(f'\nStudent:  {sd}  has seeds:')
        student_stats = []
        print(student_subdirs)
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
            student_result = smooth(student_data['AverageTestEpRet'], window)
            student_stats.append(student_result)
#            break

        # Extract label which is the <student_exp> not the <teacher_base_with_seed> portion.
        # However, we probably don't need the env name in `tail` as that's in the title.
        head, tail = os.path.split(sd)
        for name in ENV_NAMES:
            if name in tail:
                tail = tail.replace(f'{name}_', '')
        s_label = f'{tail}\n(x{len(student_stats)})'  # newline due to space

        # Shape is (num_seeds, num_recordings=250), usually 250 due to (1M steps = 250 epochs).
        student_stats = np.array(student_stats)
        student_ret = np.mean(student_stats, axis=0)    
        student_std = np.std(student_stats, axis=0)
        nb_seeds = student_stats.shape[0]
        s_label += f' [{student_ret[-1]:0.1f}]'
        s_label = s_label.replace('offline_', 'off_')

        # Actually plot (with error regions if applicable). Standard error of mean.
        x_vals = np.arange(len(student_ret))
        ax[0,1].plot(x_vals, student_ret, lw=lw, color=COLORS[sidx], label=s_label)
        if len(student_stats) > 1:
            ax[0,1].fill_between(x_vals,
                                 student_ret - (student_std / np.sqrt(nb_seeds)),
                                 student_ret + (student_std / np.sqrt(nb_seeds)),
                                 color=COLORS[sidx],
                                 alpha=0.5)
        sidx += 1

    ax[0,0].set_title(f'Teacher {args.name}', size=titlesize)
    ax[0,1].set_title(f'Students {args.name}', size=titlesize)
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

    fig_suffix = 'plot_offline_BCQ_rl.png'
    if args.add_naive:
        fig_suffix = fig_suffix.replace('.png', '_naive.png')
    if args.add_np:
        fig_suffix = fig_suffix.replace('.png', '_np.png')
    if len(args.list_distrs) > 0:
        fig_suffix = fig_suffix.replace('.png', f'_{args.list_distrs}.png')
    figname = osp.join(teacher_seed_dir, fig_suffix)
    plt.savefig(figname)
    print(f'\nSAVED FIGURE: {figname}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str)
    parser.add_argument('--window', type=int, default=3)
    parser.add_argument('--add_naive', action='store_true', help='Plot runs w/naive BCQ')
    parser.add_argument('--add_np', action='store_true', help='Plot runs w/noise predictor')
    # 0 or more values expected => creates a list to help filter by distribution.
    parser.add_argument('--list_distrs', nargs="*", type=str, default=[])
    args = parser.parse_args()
    assert args.name in ENV_NAMES, f'Error, please use a name in: {ENV_NAMES}'

    # Scan for relevant directories in the DEFAULT_DATA_DIR w/the env_name.
    if not os.path.exists(DDD):
        print(f'{DDD} does not exist! Please check.')
    directories = [osp.join(DDD,x) for x in os.listdir(DDD) if args.name in x]

    # ------------------------------ FILTER TEACHER/STUDENT DIRECTORIES ------------------------------ #
    # Key assumption: 'offline' is in the directory name for Offline RL!
    # Key assumption: each directory within dirs_T is for a random seed.
    dirs_T = sorted([d for d in directories if 'offline_' not in d])
    dirs_S = sorted([d for d in directories if 'offline_' in d])

    # Further filter the student directory. It's a bit clumsy.
    new_dirs_S = []
    for d in dirs_S:
        add1, add2 = False, False
        if args.add_naive:
            # Represents the most naive form of Offline RL.
            # TODO(daniel) real way to fix is to add a keyword like '_naive' at end.
            add1 = ('BCQ_offline_finalb' in d or
                    'BCQ_offline_uniform_0.0_0.25' in d or
                    'BCQ_offline_uniform_0.0_0.50' in d or
                    'BCQ_offline_uniform_0.0_0.75' in d or
                    'BCQ_offline_uniform_0.0_1.00' in d or
                    'BCQ_offline_uniform_0.0_1.25' in d or 
                    'BCQ_offline_uniformeps_0.0_1.25_0.5_nofilt' in d)
            add1 = add1 and ('_np' not in d)  # TODO(daniel) unfortunately need this
        if args.add_np:
            # For noise predictor, we added: `_np_[nalpha]`
            add2 = '_np_' in d
        if (add1 or add2):
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
 #           if seed == 's40':
 #              continue
            print('\n')
            print('-'*100)
            print(f'PLOTTING TEACHER  {d}  w/seed {seed}')
            print('-'*100)

            # Derive teacher path, `teacher_seed_dir`, for this seed. Then plot.
            teacher_seed_dir = [osp.join(d,x) for x in os.listdir(d) if seed in x]
            assert len(teacher_seed_dir) == 1, teacher_seed_dir
            teacher_seed_dir = teacher_seed_dir[0]
            plot(args, teacher_seed_dir=teacher_seed_dir, student_dirs=dirs_S)
