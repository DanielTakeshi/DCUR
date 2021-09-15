"""Plot overlap, specifically the two-stage procedure we use for saved buffers.

Other overlap metrics might need different plotting scripts.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn')
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
from spinup.user_config import DEFAULT_DATA_DIR
np.set_printoptions(precision=4, suppress=True, linewidth=180)
ENV_NAMES = ['ant', 'halfcheetah', 'hopper', 'walker2d']

# Matplotlib stuff
COLORS = ['red', 'blue', 'yellow', 'cyan', 'purple', 'black', 'brown', 'pink',
    'silver', 'green', 'darkblue']
COLORS_LINE = list(COLORS)


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


def get_args(target_dir, args):
    """Get args files, applies to both coarse and fine cases."""
    files = [osp.join(target_dir,x) for x in os.listdir(target_dir)
        if ('args' in x) and ('.json' in x) and ('.swp' not in x) and ('.swo' not in x)]
    if args.add_acts:
        files = [x for x in files if '_acts_True' in x]
    else:
        files = [x for x in files if '_acts_False' in x]
    assert len(files) == 1, files
    with open(files[0]) as f:
        args_from_training = json.load(f)
    return args_from_training


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


def plot_coarse(args, args_coarse, w=3):
    """Plots stuff from spinup.teaching.overlap_coarse runs."""
    titlesize = 32
    xsize = 30
    ysize = 30
    ticksize = 28
    legendsize = 23
    lw = 3
    nrows, ncols = 1, 2
    fig, ax = plt.subplots(nrows, ncols, sharey=False, squeeze=False, figsize=(11*ncols, 7*nrows))

    # The original Online DeepRL teacher results are in 'progress.txt'. These are for overlap.
    prog_txt = 'progress_acts_True.txt' if args.add_acts else 'progress_acts_False.txt'
    prog_file = osp.join(args.coarse_dir, prog_txt)
    assert os.path.exists(prog_file), prog_file
    coarse_prog = pd.read_table(prog_file)

    # Stuff we saved from coarse overlap testing.
    loss_t = coarse_prog['LossT']
    loss_v = coarse_prog['LossV']
    acc_t  = coarse_prog['AccT']
    acc_v  = coarse_prog['AccV']
    epochs = coarse_prog['Epoch']
    loss_t_np = loss_t.to_numpy()
    loss_v_np = loss_v.to_numpy()
    acc_t_np  = acc_t.to_numpy()
    acc_v_np  = acc_v.to_numpy()
    epochs_np = epochs.to_numpy()

    # Ah, if we have panda tables above (which we do) then we only have to pass that in as
    # one argument in the plot() function, instead of plot(x_vals, y_vals, [args...]).
    label1 = f'Train Loss, min {np.min(loss_t_np):0.3f}'
    label2 = f'Valid Loss, min {np.min(loss_v_np):0.3f}'
    label3 = f'Train Acc, max {np.max(acc_t_np):0.3f}'
    label4 = f'Valid Acc, max {np.max(acc_v_np):0.3f}'
    ax[0,0].plot(loss_t, lw=lw, color=COLORS[0], label=label1)
    ax[0,0].plot(loss_v, lw=lw, color=COLORS[1], label=label2)
    ax[0,1].plot(acc_t,  lw=lw, color=COLORS[0], label=label3)
    ax[0,1].plot(acc_v,  lw=lw, color=COLORS[1], label=label4)

    # Bells and whistles.
    ax[0,0].set_title(f'Loss [w/acts: {args.add_acts}]', size=titlesize)
    ax[0,1].set_title(f'Accuracy [w/acts: {args.add_acts}]', size=titlesize)
    ax[0,0].set_xlabel('Train Epochs', size=xsize)
    ax[0,1].set_xlabel('Train Epochs', size=xsize)
    ax[0,1].set_ylim([-0.01, 1.01])

    for r in range(nrows):
        for c in range(ncols):
            leg = ax[r,c].legend(loc="best", ncol=1, prop={'size':legendsize})
            for legobj in leg.legendHandles:
                legobj.set_linewidth(5.0)
            ax[r,c].tick_params(axis='x', labelsize=ticksize)
            ax[r,c].tick_params(axis='y', labelsize=ticksize)

    plt.tight_layout()
    fig_suffix = f'plot_overlap_coarse_acts_{args.add_acts}.png'
    figname = osp.join(args.coarse_dir, fig_suffix)
    plt.savefig(figname)
    print(f'\nSAVED FIGURE: {figname}')


def confusion_coarse(args, args_coarse, w=3):
    """Get a confusion matrix from spinup.teaching.overlap_coarse runs.

    Confusion matrix whose i-th row and j-th column entry indicates the number of samples
    with true label being i-th class and predicted label being j-th class. So, TL;DR, when
    reading the table, for each row, think of those as all the ground truth items in that
    row (i.e., class). For any column in that row, that's the amount of time we predicted
    that column's class when the true label was really the row class (and the diagonal
    indicates correct classification).
    """
    titlesize = 12  # Note: I stack a bunch of things in the title.
    ticksize = 15
    cm_size = 10
    lw = 3

    # Making this slightly larger. Titles can take up a lot of space!
    nrows, ncols = 1, 1
    fig, ax = plt.subplots(nrows, ncols, sharey=False, squeeze=False, figsize=(13*ncols, 10*nrows))

    # Derive original Online DeepRL teacher results from 'progress.txt'.
    prog_txt = 'progress_acts_True.txt' if args.add_acts else 'progress_acts_False.txt'
    prog_file = osp.join(args.coarse_dir, prog_txt)
    data_pkl = 'data_acts_True.pkl' if args.add_acts else 'data_acts_False.pkl'
    data_file = osp.join(args.coarse_dir, data_pkl)
    coarse_prog = pd.read_table(prog_file)
    with open(data_file, 'rb') as fh:
        coarse_data = pickle.load(fh)
    logits_v = coarse_data['logits_v']
    labels_v = coarse_data['labels_v']
    this_v_acc = coarse_data['this_v_acc']
    this_epoch = coarse_data['epoch']

    # Combine logits together, makes: (n_items, n_classes)
    # Similarly for labels, make it:  (n_items,)
    logits_all_v = np.concatenate([x for x in logits_v])
    labels_all_v = np.concatenate([x for x in labels_v])
    del logits_v
    del labels_v
    assert logits_all_v.shape[0] == labels_all_v.shape[0]
    preds_all_v = np.argmax(logits_all_v, axis=1)  # (n_items,)

    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm_raw = confusion_matrix(labels_all_v, preds_all_v)
    cm_norm = confusion_matrix(labels_all_v, preds_all_v, normalize='true')
    #print('\nRaw confusion matrix (rows=true labels, sum across row should be equal):')
    #print(cm_raw)
    #print('\nNormalized confusion matrix by row.')
    #print(cm_norm)

    # Give totals per class. Actually for 'true' the last class in validation will not in
    # general have the same number of true values as the others since I didn't do the last
    # minibatch if it was left over from (num_valid) mod (batch_sizE) during validation.
    num_pred_each = np.sum(cm_raw, axis=0)
    num_true_each = np.sum(cm_raw, axis=1)

    # https://seaborn.pydata.org/generated/seaborn.heatmap.html
    # https://stackoverflow.com/questions/35572000/how-can-i-plot-a-confusion-matrix
    # This is an Axes-level function and will draw the heatmap into the currently-active
    # Axes if none is provided to the ax argument.
    import seaborn as sns
    sns.set(font_scale=1.5)
    sns.heatmap(cm_norm,
                annot=True,
                annot_kws={"size": cm_size})

    # Class to noise name. Add leading zero if needed.
    c2n = args_coarse['class_to_buf_noise']
    new_keys = sorted([str(key).zfill(2) for key in c2n.keys()])
    title_str = f'Confusion Matrix, w/acts: {args.add_acts}'
    for idx,key in enumerate(new_keys):
        key_v2 = int(key)
        assert idx == key_v2, key_v2
        key_v2 = str(key_v2)  # string version of index
        title_str += '\n{}: {},       true #: {},  pred #: {}'.format(key, c2n[key_v2],
                num_true_each[idx], num_pred_each[idx])
    ax[0,0].set_title(title_str, fontsize=titlesize)

    # Bells and whistles.
    ax[0,0].tick_params(axis='both', which='major', labelsize=ticksize)
    plt.tight_layout()
    fig_suffix = f'plot_overlap_confusion_matrix_acts_{args.add_acts}.png'
    figname = osp.join(args.coarse_dir, fig_suffix)
    plt.savefig(figname)
    print(f'\nSAVED FIGURE: {figname}')


def plot_fine(args, dirs_1v1, dirs_1v1_args, dirs_1v1_prog, w=3):
    def get_info(progress):
        # Stuff we saved from fine overlap training.
        data = {}
        data['loss_t']    = progress['LossT']
        data['loss_v']    = progress['LossV']
        data['acc_t']     = progress['AccT']
        data['acc_v']     = progress['AccV']
        data['epochs']    = progress['Epoch']
        data['loss_t_np'] = data['loss_t'].to_numpy()
        data['loss_v_np'] = data['loss_v'].to_numpy()
        data['acc_t_np']  = data['acc_t'].to_numpy()
        data['acc_v_np']  = data['acc_v'].to_numpy()
        data['epochs_np'] = data['epochs'].to_numpy()
        return data

    # Matplotlib
    titlesize = 19
    xsize = 19
    ysize = 19
    ticksize = 19
    legendsize = 16
    lw = 2

    # Get number of models. Should be the same for all of these so index at 0.
    c2n = dirs_1v1_args[0]['class_to_buf_noise']
    n_models = len(c2n)
    nrows, ncols = n_models, n_models
    fig, ax = plt.subplots(nrows, ncols, sharey=True, squeeze=False, figsize=(6*ncols, 5*nrows))

    # Assume c2n ordering aligns w/dirs_1v1 (both sorted, the latter is sorted in this script).
    for r in range(nrows):
        ax[r,0].set_ylabel(c2n[str(r)], size=ysize)
    for c in range(ncols):
        ax[0,c].set_title(c2n[str(c)], size=titlesize)

    # This should iterate through the ordering of dirs_1v1.
    k = 0
    for r in range(nrows):
        for c in range(r+1, ncols):
            dir_name = dirs_1v1[k]
            dir_args = dirs_1v1_args[k]
            _, base_name = os.path.split(dir_name)
            base_name_split = base_name.split('__v__')
            assert len(base_name_split) == 2
            data0, data1 = base_name_split[0], base_name_split[1]
            data = get_info(dirs_1v1_prog[k])
            label_t = 'T Acc, max {:0.2f}'.format(np.max(data['acc_t_np']))
            label_t += f'\n{data0}\n{data1}'
            label_v = 'V Acc, max {:0.2f}'.format(np.max(data['acc_v_np']))
            ax[r,c].plot(data['acc_t'], lw=lw, label=label_t)
            ax[r,c].plot(data['acc_v'], lw=lw, label=label_v)
            k += 1
            ax[r,c].set_ylim([0.45, 1.0])
            ax[r,c].set_xlabel('Train Epochs', size=xsize)
            leg = ax[r,c].legend(loc="best", ncol=1, prop={'size':legendsize})
            for legobj in leg.legendHandles:
                legobj.set_linewidth(5.0)
            ax[r,c].tick_params(axis='x', labelsize=ticksize)
            ax[r,c].tick_params(axis='y', labelsize=ticksize)
    plt.tight_layout()
    fig_suffix = f'plot_overlap_fine_acts_{args.add_acts}.png'
    figname = osp.join(args.fine_dir, fig_suffix)
    plt.savefig(figname)
    print(f'\nSAVED FIGURE: {figname}')


if __name__ == '__main__':
    import argparse
    pr = argparse.ArgumentParser()
    pr.add_argument('fpath', type=str)
    pr.add_argument('--add_acts', action='store_true', help='Overlap consumes (s,a)')
    args = pr.parse_args()
    print()
    print('-'*150)
    print('-'*150)
    print()

    # The usual, adding data dir to start.
    if not osp.exists(args.fpath):
        print(f'{args.fpath} does not exist, pre-pending {DEFAULT_DATA_DIR}')
        args.fpath = osp.join(DEFAULT_DATA_DIR, args.fpath)
        assert osp.exists(args.fpath), args.fpath

    # We must have saved stuff in these dirctories, as of Feb 07, 2021.
    args.coarse_dir = coarse_dir = osp.join(args.fpath, 'overlap_1')
    args.fine_dir   = fine_dir   = osp.join(args.fpath, 'overlap_2')
    assert osp.exists(coarse_dir), coarse_dir
    assert osp.exists(fine_dir),   fine_dir

    # Search for args in the directory. We have two, for act vs no act. Then plot.
    args_coarse = get_args(coarse_dir, args)
    plot_coarse(args, args_coarse)
    confusion_coarse(args, args_coarse)

    # Get all directories for the 'fine' portion, can catch with the `__v__` string.
    dirs_1v1 = sorted([osp.join(fine_dir, x) for x in os.listdir(fine_dir) if '__v__' in x])

    # Get all arguments, etc.
    dirs_1v1_args = []
    dirs_1v1_prog = []
    for directory in dirs_1v1:
        args_fine = get_args(directory, args)
        dirs_1v1_args.append(args_fine)
        prog_txt = 'progress_acts_True.txt' if args.add_acts else 'progress_acts_False.txt'
        prog_file = osp.join(directory, prog_txt)
        assert os.path.exists(prog_file), prog_file
        prog = pd.read_table(prog_file)
        dirs_1v1_prog.append(prog)
    print(f'length dirs_1v1: {len(dirs_1v1)}')

    # Plot all of them together.
    plot_fine(args, dirs_1v1, dirs_1v1_args, dirs_1v1_prog)