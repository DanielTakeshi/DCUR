"""Utilities for overlap code."""
import os
import os.path as osp
import sys
import json
import pickle
import datetime
import numpy as np
import torch
import torch.nn as nn
import spinup.algos.pytorch.td3.core as core
torch.set_printoptions(edgeitems=10, linewidth=180)
np.set_printoptions(precision=4, edgeitems=20, linewidth=180, suppress=True)


class Classifier(nn.Module):
    """Following our older `distances` code, except w/a different architecture."""

    def __init__(self, obs_dim, act_dim, hidden_sizes, n_outputs, use_act=False,
            activation=nn.ReLU):
        super().__init__()
        self.use_act = use_act
        if use_act:
            layers = [obs_dim + act_dim] + list(hidden_sizes) + [n_outputs]
        else:
            layers = [obs_dim] + list(hidden_sizes) + [n_outputs]
        self.v = core.mlp(layers, activation)

    def forward(self, obs, act=None):
        if self.use_act:
            assert act is not None
            obs_act = torch.cat([obs, act], dim=-1)
            return self.v(obs_act)
        else:
            assert act is None
            return self.v(obs)


class Data:
    """For now our data is small enough that we can use a single class here."""

    def __init__(self, obs_buf, act_buf, labels, args):
        self.obs_buf = obs_buf
        self.act_buf = act_buf
        self.y_buf = labels
        self.use_gpu = args.use_gpu
        self.unique, self.counts = np.unique(self.y_buf, return_counts=True)
        self.n_classes = len(self.unique)
        if self.use_gpu:
            self.obs_buf = torch.as_tensor(self.obs_buf, dtype=torch.float32).to('cuda')
            self.act_buf = torch.as_tensor(self.act_buf, dtype=torch.float32).to('cuda')
            self.y_buf = torch.as_tensor(self.y_buf, dtype=torch.long).to('cuda')
        self.size = self.obs_buf.shape[0]
        self.batch_size = args.batch_size
        self.add_acts = args.add_acts

    def get_label_counts(self):
        return dict(zip(self.unique, self.counts))

    def sample_batch(self, start_end=None):
        """For training, `start_end` should be none.

        obs: (batch_size, obs_dim), label: (batch_size), not (batch_size, 1).
        We also return the action but we may ignore it in some cases.
        """
        if start_end is None:
            idxs = np.random.randint(0, self.size, size=self.batch_size)
        else:
            idxs = np.arange(start_end[0], start_end[1])
        if self.use_gpu:
            batch = {'obs': self.obs_buf[idxs],
                     'label': self.y_buf[idxs],
                     'act': self.act_buf[idxs],}
        else:
            batch = {'obs': torch.as_tensor(self.obs_buf[idxs], dtype=torch.float32),
                     'label': torch.as_tensor(self.y_buf[idxs], dtype=torch.long),
                     'act': torch.as_tensor(self.act_buf[idxs], dtype=torch.float32),}
        return batch


def get_buffers_class_info(args):
    """Modies the args to provide buffer info, if doing overlap with multiple buffers.

    Get buffers within a directory. All rollout buffers have pairing valid buffers.
    Note that sorting must be consistent among train/valid to properly index classes.
    Also provides `class_to_buf_noise` to the `args`, which is used in outer code.
    """
    buf_dir = osp.join(args.fpath, 'buffer')
    assert osp.exists(buf_dir), buf_dir
    buf_final  = [osp.join(buf_dir, x) for x in os.listdir(buf_dir) if 'final' in x and '.p' in x]
    bufs_t     = sorted([osp.join(buf_dir, x) for x in os.listdir(buf_dir) if 'train' in x and '.p' in x])
    bufs_v     = sorted([osp.join(buf_dir, x) for x in os.listdir(buf_dir) if 'valid' in x and '.p' in x])
    bufs_t_txt = sorted([osp.join(buf_dir, x) for x in os.listdir(buf_dir) if 'train' in x and '.txt' in x])
    bufs_v_txt = sorted([osp.join(buf_dir, x) for x in os.listdir(buf_dir) if 'valid' in x and '.txt' in x])
    assert len(buf_final) == 1
    buf_final = buf_final[0]
    assert len(bufs_t) == len(bufs_v)
    assert len(bufs_t_txt) == len(bufs_v_txt)
    args.bufs_t     = bufs_t
    args.bufs_v     = bufs_v
    args.bufs_t_txt = bufs_t_txt
    args.bufs_v_txt = bufs_v_txt
    args.buf_final  = buf_final

    # Let's map from the class to noise dist, follows the bufs_{t,v} sorting convention.
    # bufs_{t,v} gives us the list which maps from index to the full path.
    class_to_buf_noise = {}
    for c, bpath in enumerate(bufs_t):
        _, tail = osp.split(bpath)
        tailsplit = tail.split('-')
        assert 'noise' == tailsplit[5], tailsplit
        buf_noise = tailsplit[6]
        class_to_buf_noise[c] = buf_noise
    args.class_to_buf_noise = class_to_buf_noise
