"""Noise predictor for MuJoCo.

Importing the same ReplayBuffer from load_policy.py to make the buffer consistent.

[Feb 10 2021] Supports predicting the "vareps/xi" parameter, which is the fraction
of actions that are to be fully uniformly random from the action bounds. That is
learnable. BTW, this will only include filtered samples, so we'll expect to see
skewed targets towards lower values of this parameter even for HalfCheetah (which
has 1000 steps in each episode, but not all will be saved with 'nonaddunif_' data).

[Feb 10 2021] Using a torch/numpy random seed, and then using that in file names
to save, in case it helps with ensembling later, where we'd just load everything
and take the average of the predictions.

[Feb 11 2021] Supports time prediction, so predict the fraction of time in training.

[March 10 2021] Now trying data augmentation in case that helps? I can confirm that
    if we set np.random.seed(...) the same, then the valid indices (for logged data)
    will be the same, hence we can compare predictions on the validation data. You
    can also double check by looking at the naive predictor performance if you only
    consider the non-data augmented versions.

[March 17 2021] Fixing this so we actually use 'more random' data.
"""
from copy import deepcopy
import os
import os.path as osp
import itertools
import numpy as np
import sys
import gym
import time
import pickle
import torch
import torch.nn as nn
from torch.optim import Adam
import spinup.algos.pytorch.td3.core as core
from spinup.utils.logx import EpochLogger
from spinup.algos.pytorch.td3.td3 import ReplayBuffer
from spinup.teaching.load_policy import (load_policy_and_env, get_buffer_base_name)
from spinup.user_config import DEFAULT_DATA_DIR
np.set_printoptions(suppress=True, precision=4, linewidth=160, edgeitems=10)
ENVS = ['ant', 'halfcheetah', 'hopper', 'walker2d']


class MLPNoise(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation, data_type, n_outputs):
        super().__init__()
        self.data_type = data_type
        self.v = core.mlp([obs_dim] + list(hidden_sizes) + [n_outputs], activation)

    def forward(self, obs):
        v = self.v(obs)
        return torch.squeeze(v, -1)  # [batch_size, 1] --> [batch_size]


class NoisePredictor:

    def __init__(self, args, env, get_action):
        self.args = args
        self.get_action = get_action
        self.train_frac = 0.8

        # Environment info needed for network and buffer.
        self.env = env
        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape[0]
        obs_dim = self.obs_dim[0]  # annoying

        # Noise Predictor.
        self.n_outputs = 1
        self.net = MLPNoise(obs_dim, hidden_sizes=(256,256), activation=nn.ReLU,
                            data_type='continuous', n_outputs=self.n_outputs)
        self.criterion = nn.MSELoss()
        self.optim = Adam(self.net.parameters(), lr=1e-4)
        print(self.net)

        # Train/valid datasets and get baseline predictors.
        self.v_idx_copy = None
        self.bl_preds = {}
        self.load_buffer()
        if args.data_aug:
            self.augment_data()

        # Set up the logger to save in `experiments/` subdirectory.
        head = osp.join(args.fpath, 'experiments')
        os.makedirs(head, exist_ok=True)
        ts = str(args.train_size).zfill(7)
        vs = str(args.valid_size).zfill(7)
        ss = str(args.seed).zfill(2)

        # File base names.
        if args.time_prediction:
            base1 = f'progress-{args.noise}-tot-{ts}-seed-{ss}.txt'
            base2 = f'preds_labels-{args.noise}-tot-{ts}-seed-{ss}.pkl'
            base3 = f'sigma_predictor-{args.noise}-tot-{ts}-seed-{ss}.tar'
            base4 = f'valid_indices-{args.noise}-tot-{ts}-seed-{ss}.txt'
        else:
            base1 = f'progress-{args.noise}-t-{ts}-v-{vs}-seed-{ss}.txt'
            base2 = f'preds_labels-{args.noise}-t-{ts}-v-{vs}-seed-{ss}.pkl'
            base3 = f'sigma_predictor-{args.noise}-t-{ts}-v-{vs}-seed-{ss}.tar'
            base4 = f'valid_indices-{args.noise}-t-{ts}-v-{vs}-seed-{ss}.txt'
        if args.data_aug:
            base1 = base1.replace('.txt', '_data-aug.txt')
            base2 = base2.replace('.pkl', '_data-aug.pkl')
            base3 = base3.replace('.tar', '_data-aug.tar')
            base4 = base4.replace('.txt', '_data-aug.txt')

        # Save logger, predictions, data, neural network.
        self.logger = EpochLogger(output_dir=head, output_fname=base1)
        self.preds_labels_dir  = osp.join(head, base2)
        self.sigma_predict_dir = osp.join(head, base3)
        self.valid_indices_dir = osp.join(head, base4)
        if self.v_idx_copy is not None:
            np.savetxt(fname=self.valid_indices_dir, X=self.v_idx_copy, fmt='%i')

    def load_buffer(self):
        """Essentially recreate the init, call the built-in loading, then inspect.

        Now using the params_desired argument to double check loading.
        In the case of time prediction, we need to construct train and valid data.
        It may be OK to do it just by randomizing each tuple, (s,a,r,d,s'), since
        's' gives us all the information we need and the network is only going to
        look at 's'. There is no overlapping info, unlike the case with Atari.
        We also assign to `self.v_idx_copy,` so we can inspect the valid indices.
        """
        args = self.args
        env_name = None
        for item in ENVS:
            if item in args.train_path:
                env_name = item
                break
        if env_name is None:
            print(f'Something bad happened: {args} vs {ENVS}')
            sys.exit()
        # Daniel (02/26/2021) adding case to handle Mandi's saving of walker in buffer.
        if env_name == 'walker2d': env_name = 'walker'

        # Make buffer classes.
        params_train = dict(buffer_path=args.train_path, np_path=None, env_arg=env_name)
        params_valid = dict(buffer_path=args.valid_path, np_path=None, env_arg=env_name)
        self.t_data = ReplayBuffer(self.obs_dim, self.act_dim, size=args.train_size)
        self.v_data = ReplayBuffer(self.obs_dim, self.act_dim, size=args.valid_size)

        # Load buffers. If predicting the "xi" or "vareps" variable, that's stored in "std".
        if args.time_prediction:
            buffer_size = int(1e6)
            assert args.train_path == args.valid_path  # Special case.

            # Does not use std, uses fixed size of 1e6 for now.
            self.t_data.load_from_disk(path=args.train_path,
                                       buffer_size=buffer_size,
                                       with_std=False,
                                       params_desired=params_train)
            self.v_data.load_from_disk(path=args.valid_path,
                                       buffer_size=buffer_size,
                                       with_std=False,
                                       params_desired=params_valid)

            # If time prediction, t_data and v_data are identical. Filter to train/valid.
            # Not necessary to sort them but can make it easier to check.
            N = len(self.v_data.act_buf)
            n_train = int(N * self.train_frac)
            indices = np.random.permutation(N)
            t_idx = np.sort(indices[:n_train])
            v_idx = np.sort(indices[n_train:])
            print(f'Predicting time, {N} items to: {len(t_idx)} train, {len(v_idx)} valid.')
            self.v_idx_copy = np.copy(v_idx).astype(int)  # Assign this.

            # Now fix buffers so we only keep items from t_idx or v_idx as desired.
            self.t_data.subsample_data(idxs=t_idx, combo_size=buffer_size)
            self.v_data.subsample_data(idxs=v_idx, combo_size=buffer_size)
        else:
            self.t_data.load_from_disk(path=args.train_path,
                                       buffer_size=args.train_size,
                                       with_std=True,
                                       params_desired=params_train)
            self.v_data.load_from_disk(path=args.valid_path,
                                       buffer_size=args.valid_size,
                                       with_std=True,
                                       params_desired=params_valid)

        # Inspect the data and get baselines. Ideally the network is better than baselines!
        t_x = self.t_data.obs_buf
        v_x = self.v_data.obs_buf
        t_y, v_y = self._get_labels()
        print(f'T buf ptr/size/maxsize: {self.t_data.ptr}, {self.t_data.size}, {self.t_data.max_size}')
        print(f'V buf ptr/size/maxsize: {self.v_data.ptr}, {self.v_data.size}, {self.v_data.max_size}')
        print(f'Train, X: {t_x.shape}, Y: {t_y.shape}')
        print(f'Valid, X: {v_x.shape}, Y: {v_y.shape}')
        print(f'mean(train labels), medi(train labels): {np.mean(t_y):0.4f}, {np.median(t_y):0.4f}')
        print(f'mean(valid labels), medi(valid labels): {np.mean(v_y):0.4f}, {np.median(v_y):0.4f}')
        self.bl_preds['mean(train)'] = np.mean(t_y)
        self.bl_preds['medi(train)'] = np.median(t_y)

        # If only predicting the train data mean/median, then what is valid performance?
        # The np.mean() that wraps around these averages over full valid data.
        # The perf_MSE can be directly compared with the validation nn.MSELoss() value.
        # (Edit) unless the valid also includes the random data augmentation... in which
        # case THIS computation here will NOT consider it (good!).
        perf_MSE = np.mean(       (self.bl_preds['mean(train)'] - v_y) ** 2 )
        perf_MAE = np.mean( np.abs(self.bl_preds['medi(train)'] - v_y)      )
        print(f'Baseline Mean Estimator, full valid data MSE:   {perf_MSE:0.4f}')
        print(f'Baseline Median Estimator, full valid data MAE: {perf_MAE:0.4f}\n')

    def train(self):
        """Structured in a similar manner as my Atari runs, only with MUCH simpler code."""
        args = self.args

        # Get batch WITH standard deviation. :D
        def get_batch(buffer, start_end=None):
            if args.time_prediction:
                data = buffer.sample_batch(args.batch_size, with_timebuf=True, start_end=start_end)
                x, y = data['obs'], data['time']
            else:
                data = buffer.sample_batch(args.batch_size, with_std=True, start_end=start_end)
                x, y = data['obs'], data['std']
            assert len(y.shape) == 1, f'{y.shape} and {len(y.shape)}'
            return (x, y)

        # Note: max_size will have been potentially updated if we do data augmentation.
        lg = self.logger
        mb_t = int((self.t_data).max_size / args.batch_size)
        mb_v = int((self.v_data).max_size / args.batch_size)
        self.v_preds = []
        self.net.train()

        for e in range(args.epochs):
            lg.store(Epoch=e)

            # Train. Draw minibatches instead of going in order (order=problematic).
            for _ in range(mb_t):
                x, y = get_batch(self.t_data)
                self.optim.zero_grad()
                outputs = self.net(x)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optim.step()
                lg.store(LossT=loss)

            # Valid. (Get each item once, modulo the very last minibatch.)
            with torch.no_grad():
                for i in range(mb_v):
                    start_end = (i * args.batch_size, (i+1) * args.batch_size)
                    x, y = get_batch(self.v_data, start_end)
                    outputs = self.net(x)
                    loss = self.criterion(outputs, y)
                    lg.store(LossV=loss)

                    # If final epoch, let's save predictions in a list.
                    if e == args.epochs - 1:
                        self.v_preds.extend( list(outputs.numpy()) )

                    # Sanity checks. Can remove later.
                    naive_p = self.bl_preds['mean(train)']
                    naive   = np.mean( ((naive_p - y).numpy()) ** 2 )
                    actual  = np.mean( ((outputs - y).numpy()) ** 2 )
                    lg.store(LossVNaive=naive)
                    lg.store(LossVActual=actual)

            # Note: train/valid means we train, THEN report valid.
            lg.log_tabular('Epoch', average_only=True)
            lg.log_tabular('LossT', average_only=True)
            lg.log_tabular('LossV', average_only=True)  # nn MSE computation as done w/PyTorch
            lg.log_tabular('LossVActual', average_only=True)  # sanity, checks nn MSE computation
            lg.log_tabular('LossVNaive',  average_only=True)  # sanity, checks baseline predictor
            lg.dump_tabular()  # prevents leakage of statistics into next logging

    def save(self):
        """Some other stuff to save. Careful if changing other parts of the code!

        For saving PyTorch models, I think we may want the most flexible case later
        when we load the model but may want to resume training.
        https://pytorch.org/tutorials/beginner/saving_loading_models.html

        Load with something like this:
            model = MLPNoise(*args, **kwargs)
            optim = Adam(*args, **kwargs)
            checkpoint = torch.load(PATH)
            model.load_state_dict(checkpoint['model_state_dict'])
            optim.load_state_dict(checkpoint['optim_state_dict'])
            model.train()
        """
        _, v_y = self._get_labels()
        preds_labels = {
            'valid_preds_final': self.v_preds,
            'valid_labels': v_y,
        }
        with open(self.preds_labels_dir, 'wb') as fh:
            pickle.dump(preds_labels, fh)

        # Save PyTorch model.
        save_info = {
            'model_state_dict': self.net.state_dict(),
            'optim_state_dict': self.optim.state_dict(),
        }
        torch.save(save_info, self.sigma_predict_dir)

    def _get_labels(self):
        """Just to make it more easy to be consistent.

        Note: only called two places. One during buffer loading where we load this to check
        the training labels, then later during save(). If we're doing data augmentation for
        time prediction, then we can actually leave it as is; the first call happens before
        the time_buf gets modified, the second call is after, and it's fine to include it when
        saving."""
        if self.args.time_prediction:
            t_y = self.t_data.time_buf
            v_y = self.v_data.time_buf
        else:
            t_y = self.t_data.std_buf
            v_y = self.v_data.std_buf
        return (t_y, v_y)

    def augment_data(self):
        """Augments data via random Gaussian noise.

        Most values are within [-1,1] for each component.
        """
        env = self.env
        policy = self.get_action
        max_ep_len = 1000
        assert args.train_size == args.valid_size
        n_train = int(args.train_size * self.train_frac)
        n_valid = int(args.train_size - n_train)

        # Current observation buffers, of type float32.
        t_x = self.t_data.obs_buf
        v_x = self.v_data.obs_buf
        assert t_x.shape[0] == n_train and v_x.shape[0] == n_valid
        assert t_x.dtype == np.float32 and v_x.dtype == np.float32

        # New buffers start with `size=0` and `max_size` as `n_train` or `n_valid`.
        t_data_rand = ReplayBuffer(self.obs_dim, self.act_dim, size=n_train)
        v_data_rand = ReplayBuffer(self.obs_dim, self.act_dim, size=n_valid)

        # https://numpy.org/doc/stable/reference/random/generated/numpy.random.randn.html
        obs_t_rand = 2.0 * np.random.randn(n_train, self.obs_dim[0])
        obs_v_rand = 2.0 * np.random.randn(n_valid, self.obs_dim[0])
        print('Randomly drawing data-augmented states from a multivariate Gaussian:')
        print(f'  obs_t_rand: {obs_t_rand.shape}')
        print(f'  obs_v_rand: {obs_v_rand.shape}')
        print(f'       mean {np.mean(obs_t_rand):0.2f}, std {np.std(obs_t_rand):0.2f}, max/min: {np.max(obs_t_rand):0.2f} {np.min(obs_t_rand):0.2f}')
        print(f'       mean {np.mean(obs_v_rand):0.2f}, std {np.std(obs_v_rand):0.2f}, max/min: {np.max(obs_v_rand):0.2f} {np.min(obs_v_rand):0.2f}')
        print(f'  t_x: mean {np.mean(t_x):0.2f}, std {np.std(t_x):0.2f}, max/min: {np.max(t_x):0.2f} {np.min(t_x):0.2f}')
        print(f'  t_y: mean {np.mean(v_x):0.2f}, std {np.std(v_x):0.2f}, max/min: {np.max(v_x):0.2f} {np.min(v_x):0.2f}')
        print()

        # We store (o,a,r,d,o') for completeness but we really ONLY need the `o` here.
        # Actually to make this simpler let's just make the a, r, o2, d on the fly.
        def add_buffer(o, newbuf):
            a = np.random.randn(self.act_dim)
            o2 = np.copy(o)
            r, d = -1, -1
            newbuf.store(o, a, r, o2, d)

        # Sequentially add items from train and valid into the appropriate replay buffer.
        for i in range(n_train):
            o = obs_t_rand[i]
            add_buffer(o, t_data_rand)
        for i in range(n_valid):
            o = obs_v_rand[i]
            add_buffer(o, v_data_rand)

        # Combine/merge with old buffer, e.g., obs bufs: cat((N,odim),(N,odim)) -> (N*2,odim).
        t_combo_x = np.concatenate((self.t_data.obs_buf,  t_data_rand.obs_buf), axis=0)
        v_combo_x = np.concatenate((self.v_data.obs_buf,  v_data_rand.obs_buf), axis=0)
        t_combo_y = np.concatenate((self.t_data.time_buf, -1*np.ones(n_train)), axis=0) # labels, -1
        v_combo_y = np.concatenate((self.v_data.time_buf, -1*np.ones(n_valid)), axis=0) # labels, -1
        self.t_data.obs_buf  = t_combo_x
        self.v_data.obs_buf  = v_combo_x
        self.t_data.time_buf = t_combo_y
        self.v_data.time_buf = v_combo_y

        # If we're doubling the size we have to put in fake stuff in the other parts.
        N  = self.t_data.max_size
        NV = self.v_data.max_size
        assert len(self.t_data.obs_buf.shape) == 2
        odim = self.t_data.obs_buf.shape[1]
        assert odim == self.obs_dim[0], f'{odim} vs {self.obs_dim}'  # self.obs_dim is (N,)
        adim = self.act_dim
        self.t_data.obs2_buf = np.concatenate((self.t_data.obs2_buf, np.zeros((N,odim))), axis=0)
        self.v_data.obs2_buf = np.concatenate((self.v_data.obs2_buf, np.zeros((NV,odim))), axis=0)
        self.t_data.act_buf  = np.concatenate((self.t_data.act_buf,  np.zeros((N,adim))), axis=0)
        self.v_data.act_buf  = np.concatenate((self.v_data.act_buf,  np.zeros((NV,adim))), axis=0)
        self.t_data.rew_buf  = np.concatenate((self.t_data.rew_buf,  np.zeros(N)), axis=0)
        self.v_data.rew_buf  = np.concatenate((self.v_data.rew_buf,  np.zeros(NV)), axis=0)
        self.t_data.done_buf = np.concatenate((self.t_data.done_buf, np.zeros(N)), axis=0)
        self.v_data.done_buf = np.concatenate((self.v_data.done_buf, np.zeros(NV)), axis=0)

        # And don't forget this!
        self.t_data.size *= 2
        self.t_data.max_size *= 2
        self.v_data.size *= 2
        self.v_data.max_size *= 2
        print('\nAfter data augmentation, we have:')
        print(f'  t_data.obs:  {self.t_data.obs_buf.shape}')
        print(f'  v_data.obs:  {self.v_data.obs_buf.shape}')
        print(f'  t_data.time: {self.t_data.time_buf.shape}')
        print(f'  v_data.time: {self.v_data.time_buf.shape}')
        print(f'  t_data.obs2: {self.t_data.obs2_buf.shape} (only for allowing sampling)')
        print(f'  v_data.obs2: {self.v_data.obs2_buf.shape} (only for allowing sampling)')
        print(f'  t_data.size/max: {self.t_data.size}, {self.t_data.max_size}')
        print(f'  v_data.size/max: {self.v_data.size}, {self.v_data.max_size}')

    # NOTE(Daniel): deprecated.
    def augment_data_deprecated(self):
        """Augments the data with fake data, right after the `load_buffer()` method.

        It's not clear the best way to generate 'random' data, and sampling 'uniformly' is
        problematic given that the observation ranges from (-inf, inf) in each component.
        Since we already have the env and the saved teacher policy (with itr=-1 by default
        meaning the last teacher policy) then we might as well generate some samples on the
        fly. Then add to the original buffer, but be careful.

        Right now the noise comes from the `vareps` term that we use, so some probability
        of the time we will take an action purely at random from the environment. NOW, for the
        noise predictor, we are currently ONLY doing state --> time. Now this might still be
        problematic for the states at the beginning of each episode (as they'll also have the
        labels of -1 since they'll be generated with some probability this way) but otherwise
        we'll have to resort to ad-hoc methods to filter states.

        At the end, we have 4 replay buffers (2 train/valid with original data, 2 train/valid
        with the newer 'more random' data). Then we stack the train/valid labels for these.
        We _only_ have to worry about the obs_buf and the time_buf, I believe. And stacking
        them means old (original) data first, random data second. It's fine to do this without
        shuffling as we randomly draw minibatches anyway. However we have to adjust the
        max_size of this AND need to be careful to only use the `obs_buf` and `time_buf`, the
        other components (`act_buf`, etc.) could be undefined / garbage / zero.
        """
        env = self.env
        policy = self.get_action
        max_ep_len = 1000
        assert args.train_size == args.valid_size
        n_train = int(args.train_size * self.train_frac)
        n_valid = int(args.train_size - n_train)

        # Current observation buffers, of type float32.
        t_x = self.t_data.obs_buf
        v_x = self.v_data.obs_buf
        assert t_x.shape[0] == n_train and v_x.shape[0] == n_valid
        assert t_x.dtype == np.float32 and v_x.dtype == np.float32

        # New buffers start with `size=0` and `max_size` as `n_train` or `n_valid`.
        t_data_rand = ReplayBuffer(self.obs_dim, self.act_dim, size=n_train)
        v_data_rand = ReplayBuffer(self.obs_dim, self.act_dim, size=n_valid)

        # Add to buffer if sufficiently different from existing data?
        # Edit: no, this takes too long. Let's just keep adding while having noise injected.
        # We store (o,a,r,d,o') for completeness but we really ONLY need the `o` here.
        def add_buffer(o, a, r, o2, d, oldbuf, newbuf):
            #diffs    = o[None] - oldbuf.obs_buf                 # (N, obs_dim)
            #diffs_L2 = np.linalg.norm(diffs, axis=1)            # (N,)
            ##diffs_L2 = np.sqrt( np.sum((diffs**2), axis=1) )    # same as prior line
            #min_L2   = np.min(diffs_L2)
            #print(f'min L2: {min_L2:0.5f}')
            newbuf.store(o, a, r, o2, d)

        # Because we need a different state distribution than the teacher.
        def sample_vareps():
            return np.random.uniform(low=0.20, high=0.50)

        # Train. ep_ret/len are not really needed but good to track.
        o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
        vareps = sample_vareps()
        while t_data_rand.size < t_data_rand.max_size:
            if np.random.rand() < vareps:
                a = env.action_space.sample()
            else:
                a = policy(o)
            o2, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1
            add_buffer(o, a, r, o2, d, oldbuf=self.t_data, newbuf=t_data_rand)
            o = o2
            if d or (ep_len == max_ep_len):
                print('Epis(T) %d \t EpRet %.3f \t EpLen %d \t VarEps %.3f \t Size %d' % (
                        n, ep_ret, ep_len, vareps, t_data_rand.size))
                o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
                n += 1
                vareps = sample_vareps()

        # Valid. ep_ret/len are not really needed but good to track.
        o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
        vareps = sample_vareps()
        while v_data_rand.size < v_data_rand.max_size:
            if np.random.rand() < vareps:
                a = env.action_space.sample()
            else:
                a = policy(o)
            o2, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1
            add_buffer(o, a, r, o2, d, oldbuf=self.v_data, newbuf=v_data_rand)
            o = o2
            if d or (ep_len == max_ep_len):
                print('Epis(V) %d \t EpRet %.3f \t EpLen %d \t VarEps %.3f \t Size %d' % (
                        n, ep_ret, ep_len, vareps, v_data_rand.size))
                o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
                n += 1
                vareps = sample_vareps()

        # Combine/merge with old buffer, e.g., obs bufs: cat((N,odim),(N,odim)) -> (N*2,odim).
        t_combo_x = np.concatenate((self.t_data.obs_buf,  t_data_rand.obs_buf), axis=0)
        v_combo_x = np.concatenate((self.v_data.obs_buf,  v_data_rand.obs_buf), axis=0)
        t_combo_y = np.concatenate((self.t_data.time_buf, -1*np.ones(n_train)), axis=0) # labels, -1
        v_combo_y = np.concatenate((self.v_data.time_buf, -1*np.ones(n_valid)), axis=0) # labels, -1
        self.t_data.obs_buf  = t_combo_x
        self.v_data.obs_buf  = v_combo_x
        self.t_data.time_buf = t_combo_y
        self.v_data.time_buf = v_combo_y

        # If we're doubling the size we have to put in fake stuff in the other parts.
        N  = self.t_data.max_size
        NV = self.v_data.max_size
        assert len(self.t_data.obs_buf.shape) == 2
        odim = self.t_data.obs_buf.shape[1]
        assert odim == self.obs_dim[0], f'{odim} vs {self.obs_dim}'  # self.obs_dim is (N,)
        adim = self.act_dim
        self.t_data.obs2_buf = np.concatenate((self.t_data.obs2_buf, np.zeros((N,odim))), axis=0)
        self.v_data.obs2_buf = np.concatenate((self.v_data.obs2_buf, np.zeros((NV,odim))), axis=0)
        self.t_data.act_buf  = np.concatenate((self.t_data.act_buf,  np.zeros((N,adim))), axis=0)
        self.v_data.act_buf  = np.concatenate((self.v_data.act_buf,  np.zeros((NV,adim))), axis=0)
        self.t_data.rew_buf  = np.concatenate((self.t_data.rew_buf,  np.zeros(N)), axis=0)
        self.v_data.rew_buf  = np.concatenate((self.v_data.rew_buf,  np.zeros(NV)), axis=0)
        self.t_data.done_buf = np.concatenate((self.t_data.done_buf, np.zeros(N)), axis=0)
        self.v_data.done_buf = np.concatenate((self.v_data.done_buf, np.zeros(NV)), axis=0)

        # And don't forget this!
        self.t_data.size *= 2
        self.t_data.max_size *= 2
        self.v_data.size *= 2
        self.v_data.max_size *= 2
        print('\nAfter data augmentation, we have:')
        print(f'  t_data.obs:  {self.t_data.obs_buf.shape}')
        print(f'  v_data.obs:  {self.v_data.obs_buf.shape}')
        print(f'  t_data.time: {self.t_data.time_buf.shape}')
        print(f'  v_data.time: {self.v_data.time_buf.shape}')
        print(f'  t_data.obs2: {self.t_data.obs2_buf.shape} (only for allowing sampling)')
        print(f'  v_data.obs2: {self.v_data.obs2_buf.shape} (only for allowing sampling)')
        print(f'  t_data.size/max: {self.t_data.size}, {self.t_data.max_size}')
        print(f'  v_data.size/max: {self.v_data.size}, {self.v_data.max_size}')


if __name__ == '__main__':
    """
    Note: if train_size and valid_size are smaller than the datasets we have stored,
    then we should only take a subset of the data files. Hence we should not have more
    than one set of data files for a given agent, which simplifies things considerably.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath',                type=str)
    parser.add_argument('--itr', '-i',          type=int, default=-1)
    parser.add_argument('--noise',              type=str, default='const_0.0')
    parser.add_argument('--train_size', '-ts',  type=int, default=1000000)
    parser.add_argument('--valid_size', '-vs',  type=int, default=200000)
    parser.add_argument('--epochs', '-e',       type=int, default=50)
    parser.add_argument('--batch_size', '-bs',  type=int, default=64)
    parser.add_argument('--seed', '-s',         type=int, default=0, help='maybe for ensembles')
    parser.add_argument('--time_prediction',    action='store_true',
        help='If enabled, ignores the `noise` and runs time prediction on logged data')
    parser.add_argument('--data_aug',           action='store_true',
        help='If enabled, augment data by sampling states and assigning them with label -1')
    args = parser.parse_args()
    if args.data_aug:
        assert args.time_prediction, 'Only supports data_aug on time_prediction.'

    # Set seeds. Perhaps we'll use ensembles later, could be helpful for time prediction.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # If using time prediction, let's ignore the noise and assign it time_prediction.
    # Also we shoudd ignore the train_size and valid_size since those are hard-coded later.
    if args.time_prediction:
        args.noise = 'time_prediction'
        args.train_size = int(1e6)
        args.valid_size = int(1e6)

    # The usual, adding data dir to start.
    if not os.path.exists(args.fpath):
        print(f'{args.fpath} does not exist, pre-pending {DEFAULT_DATA_DIR}')
        args.fpath = osp.join(DEFAULT_DATA_DIR, args.fpath)
        assert osp.exists(args.fpath), args.fpath

    # Path to buffers. Using the same methods as from the load policy script which names
    # the files. TODO(daniel) in Offline RL scripts, we pass this full name as argument,
    # we should instead use this method as we do here. [Feb 10] Time pred has special case.
    if args.time_prediction:
        tmp = osp.join(args.fpath, 'buffer')
        files = [osp.join(tmp, x) for x in os.listdir(tmp) if 'final_buffer' in x]
        assert len(files) == 1, files
        args.train_path = files[0]
        args.valid_path = files[0]  # NOTE: load `files[0]` twice, then later filter the data.
    else:
        base_train = get_buffer_base_name(noise=args.noise,
                                          size=int(args.train_size),
                                          data_type='train',
                                          ending='.p')
        base_valid = get_buffer_base_name(noise=args.noise,
                                          size=int(args.valid_size),
                                          data_type='valid',
                                          ending='.p')
        args.train_path = osp.join(args.fpath, 'buffer', base_train)
        args.valid_path = osp.join(args.fpath, 'buffer', base_valid)

    # Load the env, following utils/test_policy.py.
    env, get_action = load_policy_and_env(args.fpath, args.itr if args.itr >=0 else 'last')

    # Build noise predictor.
    predictor = NoisePredictor(args, env, get_action)
    predictor.train()
    predictor.save()
