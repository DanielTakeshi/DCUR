"""Utility mehods for:
spinup.teaching.offline_rl
spinup.teaching.online_rl
"""
from copy import deepcopy
from collections import defaultdict
import itertools
import numpy as np
import sys
import os
import os.path as osp
import joblib
import time
import gym
import pickle
import torch
import torch.nn as nn
from torch.optim import Adam
from spinup.utils.logx import EpochLogger
from spinup.teaching.noise_predictor import MLPNoise
np.set_printoptions(precision=4, suppress=True, linewidth=180)
torch.set_printoptions(precision=4, linewidth=180)


class TestTimePredictor:
    """Use to test time predictor on student data (has to be from eval rollouts)."""

    def __init__(self, net, log_dir):
        self.net = net
        self.net.eval()
        self.student_log_dir = osp.join(log_dir, 'student_rollouts')
        self.n_student_samples = 10000
        if not os.path.exists(self.student_log_dir):
            os.makedirs(self.student_log_dir, exist_ok=True)

    def clear_student_data(self):
        self.x = []

    def store_student_data(self, o):
        self.x.append(o)

    def save_data_disk(self, epoch):
        """Would be better to get a consistent student data to predict later."""
        base_name = f'data_ep_{str(epoch).zfill(4)}_len_{len(self.x)}.pkl'
        target_dir = osp.join(self.student_log_dir, base_name)
        print(f'Saving student data to: {target_dir}')
        with open(target_dir, 'wb') as fh:
            pickle.dump(self.x, fh)

    def predict_student_data(self):
        x = np.array(self.x)
        print(f'Now TimePredictor predicting on STUDENT rollout data: {x.shape}')
        x = torch.as_tensor(x, dtype=torch.float32)
        vals = self.net(x)
        vals = vals.detach().numpy()
        res = {
            'TP_Mean': np.mean(vals),
            'TP_Medi': np.median(vals),
            'TP_Std': np.std(vals),
            'TP_Neg': np.sum(vals < 0.0) / len(vals),
        }
        return res


def should_we_end_early(logger):
    """Test if we should terminate early.

    If we see average Q-values exceeding some range, say outside (-10K, 10K), that means
    performance has collapsed, so we should exit training to avoid wasting compute. In
    healty online training of TD3, we should see average Q-values within (0, 1000).
    """
    terminate = False

    # NOTE(daniel) On this branch (large-scale-runs), we're disabling this.
    #Q1Vals_avg, Q1Vals_std = logger.get_stats('Q1Vals')
    #terminate = (Q1Vals_avg < -5000) or (Q1Vals_avg > 10000)

    return terminate


def load_noise_predictor(obs_dim, model_path):
    """Load noise predictor. See `noise_predictor.py` for how we saved.

    We also load the optimizer, in case we want to do further training,
    but we can also just ignore that.
    """
    o_dim = obs_dim[0]  # annoying
    hidden_sizes = (256, 256)
    model = MLPNoise(o_dim, hidden_sizes, activation=nn.ReLU,
                     data_type='continuous', n_outputs=1)
    optim = Adam(model.parameters(), lr=1e-4)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optim.load_state_dict(checkpoint['optim_state_dict'])
    return model


def get_revised_reward(model, n_alpha, r_baseline, o, a, r, o2, d, time=False,
        sparse=False):
    """Get revised shaped reward.

    Might have to tune this carefully, maybe zero centering or doing another scaling.
    Also, we should track statistics about this extra metric. Also, consider that if d=True,
    then we don't have a successor state, so we probably should not apply reward shaping.

    Args:
        n_alpha: scalar we use to multiply the reward shaping term, to adjust strength.
        r_baseline: if we use a 'baselined' version of the reward shaping: f(s')-f(s).
        time: False if doing a sigma or xi/vareps predictor. True if a time predictor,
            which reverses the terms we want to use, since high values are now better.
        o.shape: [batch_size, obs_dim], observations, (torch tensor).
        r.shape: [batch_size],          rewards (torch.float32).
        time: True if we're using the time predictor, usually the case as of April 2021.
        sparse: True if we're using sparse env rewards, so that means IGNORING the normal
            MuJoCo env reward, in favor of using just whatever we have from shaping. We
            still EVALUATE with the normal env reward, of course -- just that for training
            here, we will not use the MuJoCo hand-crafted reward.

    Returns:
        (reward, reward_stats), the former for minibatches, the latter for logging.
    """
    val_1 = model(o)
    val_2 = model(o2)

    # NOTE(daniel) As of 28 April 2021, John suggests we might try the other way.
    #assert r_baseline == 1, 'For now please set baseline 1'

    if time:
        # Higher values mean more desirable (advanced) data tuples.
        # Also we should probably clamp values as the true labels should be in [0, 1].
        # Option 0: na * f(o2)
        # Option 1: na * (f(o2) - f(o1))
        val_1 = torch.clamp(val_1, min=0.0, max=1.0)
        val_2 = torch.clamp(val_2, min=0.0, max=1.0)
        if r_baseline == 0:
            r_extra = n_alpha * val_2
        elif r_baseline == 1:
            r_extra = n_alpha * (val_2 - val_1)
        stats = dict(Time1=val_1.detach().numpy(),
                     Time2=val_2.detach().numpy(),
                     RExtra=r_extra.detach().numpy())
    else:
        # Remember that if predicting varepsilon/xi, then 'sigma' is really that value.
        # However both have the same interpretation: higher values mean more noise.
        # Option 0: -na * f(o2)
        # Option 1: na * (f(o1) - f(o2))
        if r_baseline == 0:
            r_extra = -n_alpha * val_2
        elif r_baseline == 1:
            r_extra = n_alpha * (val_1 - val_2)
        stats = dict(Sigma1=val_1.detach().numpy(),
                     Sigma2=val_2.detach().numpy(),
                     RExtra=r_extra.detach().numpy())

    # If certain tuples in the batch have done=True, do not assign reward shaping.
    r_extra[d == 1] = 0.0

    # Reward shaping. This produces a torch float tensor. Account for sparse rewards.
    #print(f'\ntime: {time}, sparse: {sparse}, r: {r.dtype}, r_extra: {r_extra.dtype}')
    #print(f'reward `r`: \n{r}')
    if sparse:
        r = r_extra
    else:
        r = r + r_extra
    r = r.detach()  # Ensure no grad_fn, but pretty sure it isn't needed.
    #print(f'reward `r`: \n{r}')
    return r, stats


def get_gt_shaping_logged(n_alpha, data, replay_buffer, r):
    """Another way to get reward shaping, for logged data in a ground truth fashion.

    `data['idxs']` and `fraction_B` have shape (batch_size,). `r` has similar shape
    but is a torch tensor. We get the extra reward, then convert to a torch tensor
    and add it to the existing reward to override later. Empirically I'm seeing
    `r` have rewards around 5-10 for HalfCheetah.
    """
    assert np.max(data['idxs']) <= replay_buffer.max_size, replay_buffer.max_size
    fraction_B = data['idxs'] / replay_buffer.max_size
    r_extra_B = n_alpha * fraction_B
    stats = dict(RExtra=r_extra_B)
    r = r + torch.as_tensor(r_extra_B, dtype=torch.float32)
    return r, stats


class StudentOnlineSchedule:
    """Use to form the online student schedule for getting samples.
    Only applies to `spinup.teaching.online_rl`.

    If we only want the student data size to be, say, 20% of the teacher's size (so that'd
    be 200K total student samples) we could do this by collecting at the end of every epoch
    and getting 4000 / 5 = 800 samples. However, episodes are 1K in length so we probably
    want to get full episodes. Another way (the one we're doing it) will be to keep the
    `steps_per_epoch` at 4000, but to only get student data after every 5th epoch.

    This still creates some issues with end-of-trajectory handling and such, since we will
    be running for 4000 steps, then stopping, then running again later where we presumably
    do a full reset of the env. We could avoid the reset but I think that's problematic
    when doing one episode with a LOT of gradient updates cut in between one point, moreso
    than in online training (since we're not running the student every epoch).

    UPDATE: actually never mind, maybe this isn't the way to go. If we're only doing 5 gradient
    updates in between actions, maybe this won't change the policy too much? Maybe it's better
    to do this: in normal online training, we take an env step EACH TIME, and then update after
    everh 50 steps (and do 50 grad updates). The process for NORMAL ONLINE TRAINING is:

    t=001 env.step()
      ...
    t=049 env.step()
    t=050 env.step(), now do update() 50 times
    t=051 env.step()
      ...
    t=099 env.step()
    t=100 env.step(), now do update() 50 times
    t=101 env.step()

    but now what we'd do with using only 20% online samples in OFFLINE is this:

    t=001 update() 1 time
    t=002 update() 1 time
    t=003 update() 1 time
    t=004 update() 1 time
    t=005 update() 1 time, env.step()
    t=006 update() 1 time
    t=007 update() 1 time
    t=008 update() 1 time
    t=009 update() 1 time
    t=010 update() 1 time, env.step()
    t=011 update() 1 time
    t=012 update() 1 time
    t=013 update() 1 time
    t=014 update() 1 time
    t=015 update() 1 time, env.step()
    t=016 update() 1 time
      ...

    Recall that this is because I changed it so we do updates() every step, there wasn't
    a reason to avoid that. Actually it might be better to switch to the earlier updating
    strategy of calling update() 50 times -- need to check that it doens't chage the
    curriculum learning numbers, so that would be like:

    t=001
      ...
    t=004
    t=005 env.step()
    t=006
      ...
    t=009
    t=010 env.step()
    t=011
      ...
    t=049
    t=050 env.step(), now do update() 50 times
    t=051
      ...
    t=099
    t=100 env.step(), now do update() 50 times
    t=101
      ...

    This assumes the student wants ~200K samples, so it steps 1/5-th as often.
    We can do something similar for 100K samples, so it steps 1/10-th as often.
    """

    def __init__(self, args):
        self.args = args
        self.steps_per_epoch = 4000  # We've never changed this.
        self.epochs = args.epochs
        self.T = self.epochs * self.steps_per_epoch
        self.size_S = args.student_size
        self.size_T = args.buffer_size

        # Equal if we're doing 250 epochs, if more, then self.T is larger.
        assert self.T >= self.size_T

        # This `collect_interval` is really what matters.
        self.collect_interval = int(self.T / self.size_S)

    def __str__(self):
        """
        Note: grad steps is approximate because we may skip a few early time steps to
        simulate the random exploration without updates that happens in usual training.
        """
        ss = '\nStudentOnlineSchedule:\n'
        ss += f'\tTime steps (i.e., number of grad updates): {self.T} (approx)\n'
        ss += f'\tTeacher Buf: {self.size_T} (fixed offline data to sample from)\n'
        ss += f'\tStudent Buf: {self.size_S} (student gets this many samples throughout training)\n'
        ss += f'\tStudent takes one online (w/noise) step every {self.collect_interval} time steps\n'
        return ss

    def assign_buffer(self, student_buffer):
        self.student_buffer = student_buffer
