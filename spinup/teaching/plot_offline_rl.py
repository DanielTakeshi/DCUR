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
import numpy as np
import sys
import json
from spinup.user_config import DEFAULT_DATA_DIR as DDD
np.set_printoptions(linewidth=180, suppress=True)
ENV_NAMES = ['ant', 'halfcheetah', 'hopper', 'walker2d']

# Matplotlib stuff
titlesize = 32
xsize = 30
ysize = 30
ticksize = 28
legendsize = 16  # adjust as needed
er_alpha = 0.25
lw = 3
COLORS = ['red', 'blue', 'yellow', 'cyan', 'purple', 'brown', 'orange', 'indigo',
    'gold', 'silver', 'brown']
COLORS_LINE = list(COLORS)


def get_dirs_S_sorted(dirs_S):
    """Finally, we need to make this ordered in a way that is easy to interpret,
    gradually increasing the amount of forward samples. Using my usual trick. :)
    """
    dirs_S_ordered = []
    for s in dirs_S:
        if 'p_1000000_n_50000_overlap' in s:
            s = s.replace('p_1000000_n_50000_overlap', 'p_1000000_n_0050000_overlap')
        elif 'p_1000000_n_100000_overlap' in s:
            s = s.replace('p_1000000_n_100000_overlap', 'p_1000000_n_0100000_overlap')
        elif 'p_1000000_n_200000_overlap' in s:
            s = s.replace('p_1000000_n_200000_overlap', 'p_1000000_n_0200000_overlap')
        elif 'p_1000000_n_500000_overlap' in s:
            s = s.replace('p_1000000_n_500000_overlap', 'p_1000000_n_0500000_overlap')
        elif 'p_800000_n_0_overlap' in s:
            s = s.replace('p_800000_n_0_overlap', 'p_0800000_n_0000000_overlap')
        elif 'online_stud_total_10000_' in s:
            s = s.replace('online_stud_total_10000_', 'online_stud_total_010000_')
        elif 'online_stud_total_25000_' in s:
            s = s.replace('online_stud_total_25000_', 'online_stud_total_025000_')
        elif 'online_stud_total_50000_' in s:
            s = s.replace('online_stud_total_50000_', 'online_stud_total_050000_')
        dirs_S_ordered.append(s)

    # Now, zip the two, then sort, by default it sorts using 1st index.
    dirs_S_zip = zip(dirs_S_ordered, dirs_S)
    dirs_S_zip = sorted(dirs_S_zip)

    # Finally get dirs_S back again, so the code logic proceeds as normal.
    # https://stackoverflow.com/questions/19339/transpose-unzip-function-inverse-of-zip
    _, dirs_S = zip(*dirs_S_zip)
    return dirs_S


def adjust_fig_suffix(fig_suffix, args):
    if args.f_override is not None:
        fig_suffix = fig_suffix.replace('.png', f'_{args.f_override}.png')
    else:
        if args.add_logged:
            fig_suffix = fig_suffix.replace('.png', '_logged.png')
        if args.add_only_finalb:
            fig_suffix = fig_suffix.replace('.png', '_onlyfinalb.png')
        if args.add_only_concur:
            fig_suffix = fig_suffix.replace('.png', '_onlyconcur.png')
        if args.add_only_curr:
            fig_suffix = fig_suffix.replace('.png', f'_onlycurrs_{args.list_currs}.png')
        if args.add_naive:
            fig_suffix = fig_suffix.replace('.png', '_naive.png')
        if args.add_np:
            fig_suffix = fig_suffix.replace('.png', '_np.png')
        if len(args.list_distrs) > 0:
            fig_suffix = fig_suffix.replace('.png', f'_{args.list_distrs}.png')
    return fig_suffix


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

    Daniel note: for machines I use, the directory prefix is one of these two:
        /data/spinup/data/...
        /data/seita/spinup/data/...
    So I remove everything up to the _second_ instance of 'data', since the leading
    directory paths before that second 'data' might not be the same. Apologies for
    inaccurate documentation initially.
    """
    # assert config['buffer_path'][0] == '/', config['buffer_path']
    assert teacher_path[0] == '/', teacher_path

    # Remove everything up to the _second_ instance of 'data'.
    def remove_leading_paths(path):
        path_split = path.split('/')  # ['', 'data', ...]
        #for i in range(0, len(path_split)): # Mandi: changed 2 to 0
        for i in range(2, len(path_split)): # Daniel: see notes above
            if path_split[i] == 'data':
                return '/'.join(path_split[i:])
        print(f'Something wrong happened: {path}')
        sys.exit()

    # The buffer path should be within the teacher path.
    buffer_path = remove_leading_paths(config['buffer_path'])
    teacher_path = remove_leading_paths(teacher_path)
    assert teacher_path in buffer_path, f'{teacher_path} not in {buffer_path}'


def ragged_array(student_stats):
    """As of Feb 18 2021, we terminate if Q-values diverge, hence need to handle this.

    If all items in `student_stats` have the same length, then calling np.array(student_stats)
    should return an array with shape (num_seeds, epochs=250).

    Args:
        student_stats: a list where each item is itself a list. Each of those nested lists
            represents rewards from training that we must plot together, and due to early
            termination conditions, they may not have the same length.
    """
    lengths = [len(x) for x in student_stats]

    if np.std(lengths) > 0:
        print(f'\nDealing with ragged array with lengths: {lengths}')
        # If std is > 0 then we must have some uneven lengths.
        # We can sort by length to make it a bit easier for indexing purposes.
        # Iterate through the (resorted) lists, take mean/std of each part, etc.
        # But TBH error bars will be all over the place with this. :(
        s_stats_sorted = sorted(student_stats, key=len)
        lengths_sorted = [len(x) for x in s_stats_sorted]
        s_ret, s_std = [], []
        prev_idx = 0
        for i in range(len(student_stats)):
            curr_idx = lengths_sorted[i]
            s_array = np.array(
                    [x[prev_idx:curr_idx] for x in student_stats if len(x[prev_idx:curr_idx]) != 0])
            # If true, then we shouldn't have any remaining things to add.
            if s_array.shape == (0,):
                break
            _s_ret  = np.mean(s_array, axis=0)
            _s_std  = np.std(s_array, axis=0)
            s_ret.append(_s_ret)
            s_std.append(_s_std)
            prev_idx = curr_idx
        s_ret = np.concatenate(s_ret)
        s_std = np.concatenate(s_std)
        ragged = True
    else:
        s_ret  = np.mean(student_stats, axis=0)
        s_std  = np.std(student_stats, axis=0)
        ragged = False
    return (s_ret, s_std, ragged)


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

    What to report? We have 250 test-time evaluations, each of which is 10 episodes. We
    may want to report (a) average over last 10 measurements (=100 episodes), and (b) the
    average over the full evaluation history (=2500 episodes), though perhaps ignore the
    very first one just in case there are artifacts of random-ness? In fact, the teacher
    takes random steps for the first 10K steps, hence we could arguably dump the first 3
    evaluations (at 4K, 8K, and 12K steps). It also only starts gradient updates after
    1K steps. So let's just say we dump the first 3 measurements for all teachers/students
    in the plots, and put that number in the plot legends.

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
        s_label = f'{tail} (x{nb_seeds}, e{terminated_early})\n'  # newline due to space

        # If `tail` has curriculum_ in name, might want to shorten legend names.
        #s_label = s_label.replace('_curriculum_', '\n  curr_')  # use for newline
        s_label = s_label.replace('_offline_curriculum_', '_curr_')  # use for no newline

        # Shape is (num_seeds, num_recordings=250), usually 250 due to (1M steps = 250 epochs).
        # TODO(daniel) Recording of s_stat_{1,2} might not be intended if we have ragged array.
        student_ret, student_std, _ = ragged_array(student_stats)
        s_stat_1 = np.mean(student_stats_1)  # last 10
        s_stat_2 = np.mean(student_stats_2)  # all (except first 3)
        s_label += f'last: {student_ret[-1]:0.1f}, M1: {s_stat_1:0.1f}, M2: {s_stat_2:0.1f}'
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

    fig_suffix = 'plot_offline_rl.png'
    fig_suffix = adjust_fig_suffix(fig_suffix, args)
    figname = osp.join(teacher_seed_dir, fig_suffix)
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
            plot(args, teacher_seed_dir=teacher_seed_dir, student_dirs=dirs_S)
