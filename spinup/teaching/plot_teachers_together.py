"""Plot the teacher results from the paper.

See https://gist.github.com/DanielTakeshi/a4a8c431bd3ab30ed578f3b579083c7a
It was surprisingly hard to figure out how to do this ...
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn')

from copy import deepcopy
import os
import os.path as osp
import numpy as np
from spinup.user_config import DEFAULT_DATA_DIR as DDD
from spinup.teaching.plot_offline_rl import smooth
from spinup.teaching.plot_paper import get_teacher_stats
np.set_printoptions(linewidth=180, suppress=True)

# Matplotlib stuff
titlesize = 41
xsize = 35
ysize = 35
ticksize = 31
legendsize = 36  # adjust as needed
lw = 4

# Make sure there's no extra slash at the end.
NAME_TO_TD3 = {
    'Ant-v3':         'ant_td3_act0-1/ant_td3_act0-1_s50',
    'HalfCheetah-v3': 'halfcheetah_td3_act0-1/halfcheetah_td3_act0-1_s40',
    'Hopper-v3':      'hopper_td3_act0-1/hopper_td3_act0-1_s40',
    'Walker2d-v3':    'walker2d_td3_act0-1/walker2d_td3_act0-1_s50',
}
NAME_TO_SAC = {
    'Ant-v3':         'ant_sac_alpha0-2_fix_alpha/ant_sac_alpha0-2_fix_alpha_s50',
    'HalfCheetah-v3': 'halfcheetah_sac_alpha0-2_fix_alpha/halfcheetah_sac_alpha0-2_fix_alpha_s40',
    'Hopper-v3':      'hopper_sac_alpha0-2_fix_alpha/hopper_sac_alpha0-2_fix_alpha_s40',
    'Walker2d-v3':    'walker2d_sac_alpha0-2_fix_alpha/walker2d_sac_alpha0-2_fix_alpha_s50',
}
KEYS = sorted(list(NAME_TO_TD3.keys()))


def teacher_plot(args):
    window = args.window
    nrows, ncols = 1, len(NAME_TO_TD3)
    fig, ax = plt.subplots(nrows, ncols, sharey=False, squeeze=True, figsize=(8*ncols, 8*nrows))
    handles = []

    for i in range(ncols):
        env = KEYS[i]
        print(f'\nPlotting teacher for: {env}')

        # TD3 teacher.
        teacher_seed_dir = osp.join(DDD, NAME_TO_TD3[env])
        (teacher_data, _, _, t_label) = get_teacher_stats(teacher_seed_dir)
        ret_test = smooth(teacher_data['AverageTestEpRet'], window)
        h1, = ax[i].plot(ret_test,  ls='-',  lw=lw, color='blue')
        print(t_label)

        # SAC teacher.
        teacher_seed_dir = osp.join(DDD, NAME_TO_SAC[env])
        (teacher_data, _, _, t_label) = get_teacher_stats(teacher_seed_dir)
        ret_test = smooth(teacher_data['AverageTestEpRet'], window)
        h2, = ax[i].plot(ret_test,  ls='-',  lw=lw, color='red')
        print(t_label)

        # Yes, we need the comma after the `h` in the code above.
        # TODO(daniel) I do not understand why but if I don't have this, then both curves
        # are red (??). However, it seems like adding the h1 for just one (I chose Ant) will
        # cause the TD3 color to be blue. This is very confusing.
        if env == 'Ant-v3':
            handles.append(h1)
        else:
            handles.append(h2)

        # Bells and whistles.
        ax[i].set_xlabel('Train Epochs', size=xsize)
        ax[i].set_ylabel('Test Return', size=ysize)
        ax[i].set_title(f'{env}', size=titlesize)
        ax[i].tick_params(axis='x', labelsize=ticksize)
        ax[i].tick_params(axis='y', labelsize=ticksize)

    # https://stackoverflow.com/questions/37967786/
    fig.legend(
        handles=handles,
        labels=['TD3 Teacher', 'SAC Teacher'],
        loc='lower center',
        prop={'size': legendsize},
        bbox_to_anchor=(0.50, 0.04),
        borderaxespad=0,
        frameon=True,
        ncol=2,
    )

    # https://stackoverflow.com/questions/8248467/ Must happen after tight_layout.
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.30)

    figname = f'plot_paper_teachers.png'
    plt.savefig(figname)
    print(f'\nSAVED FIGURE: {figname}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--window', type=int, default=6)
    args = parser.parse_args()
    if not os.path.exists(DDD):
        print(f'{DDD} does not exist! Please check.')
    teacher_plot(args)