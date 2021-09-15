"""Overlap for MuJoCo, using a COARSE network, with lots of outputs.
We can then use a second layer after this to make it more fine-grained.

Currently finds all buffers within a teacher's buffer/ directory.

Prior overlap here: https://github.com/CannyLab/distances/tree/master/distances
which I worked on for the NeurIPS 2019 workshop paper, though we never used it.
Should think about the architecture, it should take in either (s) or (s,a), and
maybe it follows the same architecture as policy or Q-networks.

Helpful references:
https://pytorch.org/docs/stable/notes/cuda.html
https://pytorch.org/tutorials/beginner/saving_loading_models.html
https://discuss.pytorch.org/t/how-to-load-all-data-into-gpu-for-training/27609/28
https://discuss.pytorch.org/t/what-is-the-difference-between-doing-net-cuda-vs-net-to-device/69278/9

We're currently combining multiple options (opt) here. They are:

opt 1: 'coarse' overlap.
opt 2: 'fine' overlap.
opt 3: binary overlap but where we just drag and drop two buffers.
opt 4: where we take an arbitrary data and feed it through the net, ideally can be
    used for the student. Maybe even support adding in an extra output to an existing
    overlap network for the student, then running the training.

Update mid-March 2021: let's also define a new class here for overlap analysis within
our Offline RL runs.
"""
import os
import os.path as osp
import sys
import gym
import copy
import json
import time
import pickle
import datetime
import itertools
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import spinup.algos.pytorch.td3.core as core
from spinup.utils.logx import EpochLogger
from spinup.algos.pytorch.td3.td3 import ReplayBuffer
from spinup.teaching.load_policy import (load_policy_and_env, get_buffer_base_name)
from spinup.teaching.overlap_utils import (Classifier, Data)
import spinup.teaching.overlap_utils as OU
from spinup.user_config import DEFAULT_DATA_DIR
torch.set_printoptions(edgeitems=10, linewidth=180)
np.set_printoptions(precision=4, edgeitems=20, linewidth=180, suppress=True)
ENVS = ['ant', 'halfcheetah', 'hopper', 'walker2d']


class OverlapOfflineRL:
    """Specifically for analysis within OfflineRL runs.

    Basic usage: student learns in an offline fashion, but we want to find overlap between
    its state distribution and the current teacher data (which may differ throughout training
    due to the effect of curricula). Since the student is offline, the only way we can get
    data is during test-time evaluation. So repeatedly do overlap computation throughout.
    """

    def __init__(self, env, curriculum, t_buffer):
        self.curriculum = curriculum
        self.t_buffer = t_buffer

        # Env stuff.
        self.env = env
        self.obs_dim = self.env.observation_space.shape
        self.act_dim = act_dim = self.env.action_space.shape[0]

        # For classifiers. Get `n_student_samples` each time to compare w/teacher.
        self.use_gpu = False
        self.n_classes = 2
        self.lrate = 1e-4
        self.l2reg = 1e-6
        self.epochs = 20  # should tune
        self.n_student_samples = 50000
        self.batch_size = 100
        self.train_frac = 0.8

        # Store different classes here; may make it more scalable to multiple teachers.
        # For int `idx`, c2data[idx] = dict(obs, act, y=idx), so maybe `idx` is redundant?
        self.c2data = {}  # Student data in class 0, teachers in 1 (and maybe beyond).
        self.s_label = int(0)
        self.t_label = int(1)
        self.c2data[self.s_label] = dict(obs=[], act=[], y=[])
        self._store_teacher_data(label=self.t_label)

        # Debugging.
        print(f'\nOverlap analysis for env: {self.env}')
        #self._inspect_data()

    def _inspect_data(self):
        print(f'For class {self.t_label} (teacher):')
        data = self.c2data[self.t_label]
        print('obs: {}'.format(data['obs'].shape))
        print('act: {}'.format(data['act'].shape))
        print('y:   {} (label: {})'.format(data['y'].shape, data['y'][0]))

    def _store_teacher_data(self, label):
        """Store the full teacher data as a dict, then put that in the c2data dict."""
        obs = self.t_buffer.obs_buf
        act = self.t_buffer.act_buf
        assert len(obs.shape) == 2 and len(act.shape) == 2
        y = np.ones( (obs.shape[0]) )
        y = (y * label).astype(np.uint8)
        self.c2data[label] = dict(obs=obs, act=act, y=y)

    def clear_student_data(self):
        """Clear the student data so we start overlap computation w/only fresh data."""
        self.c2data[self.s_label] = dict(obs=[], act=[], y=[])

    def store_student_data(self, o, a):
        """Public method for adding student data, we'll later turn to np.array for overlap."""
        self.c2data[self.s_label]['obs'].append(o)
        self.c2data[self.s_label]['act'].append(a)
        self.c2data[self.s_label]['y'].append(self.s_label)

    def compute_overlap(self, teacher_data):
        """Public method for usage in Offline RL.

        Passing in `teacher_data` only for the purposes of getting curriculum values if
        they are set. We should have already stored teacher data during initialization.
        Note: here we should do the train/valid split, might as well get it done before
        we call the training later. To make it easier, let's just keep the data fixed,
        but we'll make separate train and valid _indices_, and then sample from those.

        It is simpler to create the overlap net here since we need a new network for each
        new data batch from the student.
        """
        sl = self.s_label
        tl = self.t_label
        self.c2_train_idx = {}
        self.c2_valid_idx = {}

        # For student data, we need to make the lists into numpy arrays first.
        self.c2data[sl]['obs'] = np.array(self.c2data[sl]['obs'])
        self.c2data[sl]['act'] = np.array(self.c2data[sl]['act'])
        self.c2data[sl]['y']   = np.array(self.c2data[sl]['y'])
        Ns = len(self.c2data[sl]['y'])
        print(f'\nForm train/valid split, for STUDENT data, nsamples: {Ns}.')
        assert Ns >= self.n_student_samples, Ns
        s_indices = np.random.permutation(Ns)
        s_limit = int(Ns * self.train_frac)
        self.c2_train_idx[sl] = s_indices[:s_limit]
        self.c2_valid_idx[sl] = s_indices[s_limit:]
        s_intersect = np.intersect1d(self.c2_train_idx[sl], self.c2_valid_idx[sl])
        assert len(s_intersect) == 0

        # Teacher samples are pre-loaded, and we only want INDICES in (min_idx, max_idx).
        min_idx = teacher_data.curr_min
        max_idx = teacher_data.curr_max
        print(f'Form train/valid split, for TEACHER data, min/max: {min_idx}/{max_idx}.')
        t_range = np.arange(min_idx, max_idx)  # from min (inclusive) to max (exclusive)
        Nt = len(t_range)
        # Daniel: actually sometimes we have 1950 if the scale param is 0.50t. Let's ignore.
        #assert Nt >= 3950, Nt  # Hm I thought it was 4000 but off by 50... not a big deal.
        t_indices = np.random.permutation(Nt)  # indices for t_range
        t_limit = int(Nt * self.train_frac)
        self.c2_train_idx[tl] = t_range[ t_indices[:t_limit] ]
        self.c2_valid_idx[tl] = t_range[ t_indices[t_limit:] ]
        t_intersect = np.intersect1d(self.c2_train_idx[tl], self.c2_valid_idx[tl])
        assert len(t_intersect) == 0

        # Next, let's make sure we're using the same amount of data for each. The indices
        # for teachers were already randomized (for students they're from the same policy)
        # so removing any items after a certain index should be fine.
        min_t = min(s_limit, t_limit)
        min_v = min(Ns - s_limit, Nt - t_limit)

        # Crop the indices within these dicts.
        self.c2_train_idx[sl] = self.c2_train_idx[sl][:min_t]
        self.c2_train_idx[tl] = self.c2_train_idx[tl][:min_t]
        self.c2_valid_idx[sl] = self.c2_valid_idx[sl][:min_v]
        self.c2_valid_idx[tl] = self.c2_valid_idx[tl][:min_v]

        # Debugging.
        print(f'Ns: {Ns}, Nt: {Nt}')
        print(f'min_t: {min_t}, min_v: {min_v}')
        print('student (c={}), obs {}'.format(sl, self.c2data[sl]['obs'].shape))
        print('               act {}'.format(self.c2data[sl]['act'].shape))
        print('               y   {}'.format(self.c2data[sl]['y'].shape))
        print('student, len train/valid INDICES: {}, {}'.format(
                len(self.c2_train_idx[sl]), len(self.c2_valid_idx[sl])))
        print('teacher (c={}), obs {}'.format(tl, self.c2data[tl]['obs'].shape))
        print('               act {}'.format(self.c2data[tl]['act'].shape))
        print('               y   {}'.format(self.c2data[tl]['y'].shape))
        print('teacher, len train/valid INDICES: {}, {}'.format(
                len(self.c2_train_idx[tl]), len(self.c2_valid_idx[tl])))

        # Classifiers. We can do the obs and obs+act cases separately.
        self.net_1 = Classifier(self.obs_dim[0],
                                self.act_dim,
                                hidden_sizes=(256,256),
                                n_outputs=self.n_classes,
                                use_act=False)
        self.net_2 = Classifier(self.obs_dim[0],
                                self.act_dim,
                                hidden_sizes=(256,256),
                                n_outputs=self.n_classes,
                                use_act=True)
        if self.use_gpu:
            self.net_1 = self.net_1.to('cuda')
            self.net_2 = self.net_2.to('cuda')

        # Criterion and Optimizer. The criterion doesn't have to be separate but w/e.
        self.criterion_1 = nn.CrossEntropyLoss()
        self.criterion_2 = nn.CrossEntropyLoss()
        self.optim_1 = Adam(self.net_1.parameters(), lr=self.lrate, weight_decay=self.l2reg)
        self.optim_2 = Adam(self.net_2.parameters(), lr=self.lrate, weight_decay=self.l2reg)

        # Overlap! (Update April 28: let's not do the actions one, taking a while.)
        print('\nTraining overlap, withOUT actions...')
        res_1 = self._train(self.net_1, self.optim_1, self.criterion_1, add_acts=False)
        #print('\nTraining overlap, WITH actions...')
        #res_2 = self._train(self.net_2, self.optim_2, self.criterion_2, add_acts=True)
        #print('Done, proceed with Offline RL training...\n')
        #return (res_1, res_2)
        print('Done, proceed with Offline RL training...\n')
        return (res_1, None)

    def _sample_batch(self, dtype, start_end=None):
        """Sample a minibatch.

        For now assume that we want equal numbers of student vs teacher in each minibatch.
        For training this just means sampling from the INDICES, again we can't sample from
        the raw data since the data might contain more items than we have indices.

        For valid we need to be a bit careful ... we're dividing the start and end indices,
        then sampling those from the valid indices. It assumes that those are equal in size.
        """
        assert dtype in ['train', 'valid']
        sl = self.s_label
        tl = self.t_label
        num = int(self.batch_size / 2)
        if dtype == 'train':
            s_indices = self.c2_train_idx[sl]
            t_indices = self.c2_train_idx[tl]
            s_idxs = np.random.choice(s_indices, size=num)
            t_idxs = np.random.choice(t_indices, size=num)
        else:
            s_indices = self.c2_valid_idx[sl]
            t_indices = self.c2_valid_idx[tl]
            start, end = start_end
            assert end - start == self.batch_size, f'{end} - {start}'
            s_idxs = s_indices[int(start/2) : int(end/2)]
            t_idxs = t_indices[int(start/2) : int(end/2)]
        assert len(s_indices) == len(t_indices), 'We want to sample equally.'

        # Student data at indices `s_idxs`.
        s_o = self.c2data[sl]['obs'][s_idxs]
        s_a = self.c2data[sl]['act'][s_idxs]
        s_y = self.c2data[sl]['y']  [s_idxs]

        # Teacher data at indices `t_idxs`.
        t_o = self.c2data[tl]['obs'][t_idxs]
        t_a = self.c2data[tl]['act'][t_idxs]
        t_y = self.c2data[tl]['y']  [t_idxs]

        # Combine + return dict.
        b_o = np.concatenate((s_o,t_o), axis=0)  # (batch_size, obs_dim)
        b_a = np.concatenate((s_a,t_a), axis=0)  # (batch_size, act_dim)
        b_y = np.concatenate((s_y,t_y), axis=0)  # (batch_size,)
        batch = {
            'obs': torch.as_tensor(b_o, dtype=torch.float32),
            'act': torch.as_tensor(b_a, dtype=torch.float32),
            'label': torch.as_tensor(b_y, dtype=torch.long),
        }
        return batch

    def _train(self, net, optim, criterion, add_acts, debug=False):
        """Structured in a similar manner as my Atari runs, only with MUCH simpler code.

        For each epoch, do train first, THEN valid. The valid computes overlap statistics.

        When training, the size of teacher data may vary, while student data should be
        fixed. Generally the student data will be smaller than the teacher data (exception
        is at the beginning of training if we use a curriculum that only provides a small
        number of teacher samples). Maybe it's best to ensure before we call this that we
        will be using the same number of data in each class? Thus, the requirement we have
        is: given `self.c2_{train,valid}_idx`, the `tl` and `sl` labels will have the same
        number of indices to sample from. Again, the data is fixed but we only sample from
        a specified range of indices.
        """
        #lg = self.logger
        n_train = len(self.c2_train_idx[self.s_label]) * self.n_classes
        n_valid = len(self.c2_valid_idx[self.s_label]) * self.n_classes
        mb_t = int(n_train / self.batch_size)
        mb_v = int(n_valid / self.batch_size)
        print(f'Training with t/v minibatches: {mb_t}, {mb_v}')
        start_time = time.time()
        best_v_acc = 0
        result = {}

        for e in range(self.epochs):
            #logits_v, labels_v = [], []

            # Train. Draw minibatches instead of going in order (order=problematic).
            loss_t, total_t, correct_t = 0, 0, 0
            net.train()
            for _ in range(mb_t):
                batch = self._sample_batch('train')
                o, y, a = batch['obs'], batch['label'], batch['act']
                optim.zero_grad()
                if add_acts:
                    outputs = net(o, a)
                else:
                    outputs = net(o)
                loss = criterion(outputs, y)
                loss.backward()
                optim.step()
                _, predicted = torch.max(outputs, 1)
                total_t += y.size(0)
                correct_t += (predicted == y).sum().item()
                loss_t += loss.item()
            LossT = loss_t / mb_t
            AccT = correct_t / total_t

            # Valid. Get each item once, except for last "valid_size mod batch_size" items.
            # Edit: actually it's a bit more complex than this because the valid data size
            # is variable, especially for the teacher data with curricula. Update: let's
            # make the datasets equal among student/teacher, then we can use start_end and
            # divide both numbers by 2, to find indices in the student/teacher index lists.
            loss_v, total_v, correct_v = 0, 0, 0
            net.eval()
            with torch.no_grad():
                for i in range(mb_v):
                    start_end = (i * self.batch_size, (i+1) * self.batch_size)
                    batch = self._sample_batch('valid', start_end=start_end)
                    o, y, a = batch['obs'], batch['label'], batch['act']
                    if add_acts:
                        outputs = net(o, a)
                    else:
                        outputs = net(o)
                    loss = criterion(outputs, y)
                    _, predicted = torch.max(outputs, 1)
                    total_v += y.size(0)
                    correct_v += (predicted == y).sum().item()
                    loss_v += loss.item()
                    #logits_v.append(outputs.detach().cpu().numpy())  # appending (batch, n_classes)
                    #labels_v.append(y.detach().cpu().numpy())        # appending (batch)
                LossV = loss_v / mb_v
                AccV = correct_v / total_v
                OverlapV = 2.0 * (1.0 - AccV)

            # Compute stats each minibatch, and report the average, so it's the epoch average.
            elapsed_time = time.time() - start_time
            if debug:
                print('  Epoch/Time:    {}, {:0.3f}'.format(e, elapsed_time))
                print('  LossT/LossV:   {:0.3f}, {:0.3f}'.format(LossT, LossV))
                print('  ActT/AccV:     {:0.3f}, {:0.3f}'.format(AccT, AccV))
                print('  OverlapV:      {:0.3f}'.format(OverlapV))

            # If this is the best epoch, save the statistics.
            if AccV > best_v_acc:
                best_v_acc = AccV
                result['OverlapV'] = OverlapV
                result['AccT'] = AccT
                result['AccV'] = AccV
                result['LossT'] = LossT
                result['LossV'] = LossV
                result['Epoch'] = e
        return result


class Overlap:

    def __init__(self, args, env, class_i=None, class_j=None, pretrained_dict=None):
        """

        Args:
            class_i: used for opt=2, i-th class to test.
            class_j: used for opt=2, j-th class to test.
            pretrained_dict: used for opt=2 to load a pre-trained model from opt=1, then
                based on class_{i,j}, we'll extract the corresponding parts of the params.
        """
        self.args = args
        self.add_acts = args.add_acts

        # Environment info needed for network and buffer.
        self.env = env
        self.obs_dim = self.env.observation_space.shape
        act_dim = self.env.action_space.shape[0]
        obs_dim = self.obs_dim[0]  # annoying

        # Form the train/valid data, get number of classes. Might have to change later.
        if args.opt in [1, 2]:
            self.data_train, self.data_valid = self._make_data(obs_dim, act_dim)
        else:
            raise NotImplementedError
        self.n_classes = self.data_train.n_classes
        self.measure_overlap = (self.n_classes == 2)

        # Classifier, potentially with pre-trained dict.
        self.net = Classifier(obs_dim, act_dim, hidden_sizes=(256,256),
                n_outputs=self.n_classes, use_act=self.add_acts)
        if pretrained_dict is not None:
            # Standard `OrderedDict.update()`, adjust weights wrt pretrained_dict.
            w_keys = sorted([x for x in pretrained_dict.keys() if 'bias' not in x])
            b_keys = sorted([x for x in pretrained_dict.keys() if 'weight' not in x])
            w_matrix = pretrained_dict[w_keys[-1]]  # (n_models, 256) --> (2, 256)
            b_vector = pretrained_dict[b_keys[-1]]  # (n_models,)     --> (2,)
            pretrained_dict[w_keys[-1]] = w_matrix[[class_i,class_j],:]
            pretrained_dict[b_keys[-1]] = b_vector[[class_i,class_j]]
            model_dict = self.net.state_dict()
            model_dict.update(pretrained_dict)
            self.net.load_state_dict(model_dict)

        # Criterion and Optimizer.
        self.criterion = nn.CrossEntropyLoss()
        self.optim = Adam(self.net.parameters(), lr=args.lrate, weight_decay=args.l2reg)
        if args.use_gpu:
            self.net = self.net.to('cuda')
        print(self.net)

        # Save logger, predictions/data, neural network.
        base = f'progress_acts_{self.add_acts}.txt'
        self.logger = EpochLogger(output_dir=args.overlap_dir, output_fname=base)
        self.overlap_dir = args.overlap_dir

    def _make_data(self, obs_dim, act_dim):
        """Constructs the data. For simplcity we can merge a bunch of buffers.

        No need for PyTorch data loaders, we're just loading from arrays.
        THIS DEPENDS ON THE BUF PATHS BEING CONSISTENTLY STORED ACROSS TRAIN/VALID!
        """
        args = self.args

        # Form the data but keep track of the label (class index) as uint8.
        def get_data(bufs_paths):
            all_obs = []
            all_act = []
            labels = []
            for idx,path in enumerate(bufs_paths):
                buf = torch.load(path)
                obs = buf['obs']
                act = buf['act']
                assert len(obs.shape) == 2 and obs.shape[1] == obs_dim, obs.shape
                assert len(act.shape) == 2 and act.shape[1] == act_dim, act.shape
                y = np.ones( (obs.shape[0]) )
                y = (y * idx).astype(np.uint8)
                all_obs.append( obs )
                all_act.append( act )
                labels.append( y)
            all_obs = np.concatenate(all_obs)  # (N, obs_dim)
            all_act = np.concatenate(all_act)  # (N, act_dim)
            all_y   = np.concatenate(labels)   # (N,)
            all_y = all_y.astype(np.uint8)
            return all_obs, all_act, all_y

        # Construct the class (a lightweight wrapper, really).
        obs_t, act_t, labels_t = get_data(bufs_paths=args.bufs_t)
        obs_v, act_v, labels_v = get_data(bufs_paths=args.bufs_v)
        data_train = Data(obs_t, act_t, labels_t, args)
        data_valid = Data(obs_v, act_v, labels_v, args)

        print('\nConstructed train/valid data.')
        print(f'  Train: {data_train.obs_buf.shape}')
        print(f'         {data_train.y_buf.shape}')
        print(f'  Valid: {data_valid.obs_buf.shape}')
        print(f'         {data_valid.y_buf.shape}')
        if args.use_gpu:
            print(f'obs device: {data_train.obs_buf.device}')
            print(f'y device:   {data_train.y_buf.device}')
        y_counts_t = data_train.get_label_counts()
        y_counts_v = data_valid.get_label_counts()
        print(f'Train counts: {y_counts_t}')
        print(f'Valid counts: {y_counts_v}')
        assert data_train.n_classes == data_valid.n_classes
        print(f'Number of classes: {data_train.n_classes}\n')
        return (data_train, data_valid)

    def train(self):
        """Structured in a similar manner as my Atari runs, only with MUCH simpler code.

        BTW, for each epoch, we do train first, THEN the valid set.
        """
        args = self.args
        lg = self.logger
        mb_t = int(self.data_train.size / args.batch_size)
        mb_v = int(self.data_valid.size / args.batch_size)
        print(f'\nNow training with t/v minibatches: {mb_t}, {mb_v}')
        start_time = time.time()
        best_v_acc = 0

        for e in range(args.epochs):
            lg.store(Epoch=e)
            logits_v = []
            labels_v = []

            # Train. Draw minibatches instead of going in order (order=problematic).
            loss_t, total_t, correct_t = 0, 0, 0
            self.net.train()
            for _ in range(mb_t):
                batch = self.data_train.sample_batch()
                x, y, a = batch['obs'], batch['label'], batch['act']
                self.optim.zero_grad()
                if self.add_acts:
                    outputs = self.net(x, a)
                else:
                    outputs = self.net(x)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optim.step()
                _, predicted = torch.max(outputs, 1)
                total_t += y.size(0)
                correct_t += (predicted == y).sum().item()
                loss_t += loss.item()
            lg.store(LossT=loss_t / mb_t)
            lg.store(AccT=correct_t / total_t)

            # Valid. Get each item once, except for last "valid_size mod batch_size" items.
            loss_v, total_v, correct_v = 0, 0, 0
            self.net.eval()
            with torch.no_grad():
                for i in range(mb_v):
                    start_end = (i * args.batch_size, (i+1) * args.batch_size)
                    batch = self.data_valid.sample_batch(start_end=start_end)
                    x, y, a = batch['obs'], batch['label'], batch['act']
                    if self.add_acts:
                        outputs = self.net(x, a)
                    else:
                        outputs = self.net(x)
                    loss = self.criterion(outputs, y)
                    _, predicted = torch.max(outputs, 1)
                    total_v += y.size(0)
                    correct_v += (predicted == y).sum().item()
                    loss_v += loss.item()
                    logits_v.append(outputs.detach().cpu().numpy())  # appending (batch, n_classes)
                    labels_v.append(y.detach().cpu().numpy())        # appending (batch)
                this_v_acc = correct_v / total_v
                lg.store(LossV=loss_v / mb_v)
                lg.store(AccV=this_v_acc)
                if self.measure_overlap:
                    lg.store(OverlapV=2.0 * (1.0 - this_v_acc))

            # Compute stats each minibatch, and report the average, so it's the epoch average.
            elapsed_time = time.time() - start_time
            lg.store(Time=elapsed_time)
            lg.log_tabular('Epoch', average_only=True)
            lg.log_tabular('LossT', average_only=True)
            lg.log_tabular('LossV', average_only=True)
            lg.log_tabular('AccT', average_only=True)
            lg.log_tabular('AccV', average_only=True)
            lg.log_tabular('Time', average_only=True)
            if self.measure_overlap:
                lg.log_tabular('OverlapV', average_only=True)
            lg.dump_tabular()  # prevents leakage of statistics into next logging

            # If valid is better than before, save at this epoch.
            if this_v_acc > best_v_acc:
                best_v_acc = this_v_acc
                self.save(logits_v, labels_v, this_v_acc, epoch=e)

    def save(self, logits_v, labels_v, this_v_acc, epoch):
        """Save PyTorch model using the most flexible case, plus the predictions.

        I think we may also want to save the full logits as well, in `data_to_store`.

        logits_v: a LIST of (batch, n_classes) outputs, for each valid minibatch.
        labels_v: a LIST of (batch) labels, for each valid minibatch.
        """
        data_pth = osp.join(self.overlap_dir, f'data_acts_{self.add_acts}.pkl')
        data_to_store = {
            'logits_v': logits_v,
            'labels_v': labels_v,
            'this_v_acc': this_v_acc,
            'epoch': epoch,
        }
        with open(data_pth, 'wb') as fh:
            pickle.dump(data_to_store, fh)
        net_pth = osp.join(self.overlap_dir, f'net_acts_{self.add_acts}.pth')
        save_info = {
            'model_state_dict': self.net.state_dict(),
            'optim_state_dict': self.optim.state_dict(),
        }
        torch.save(save_info, net_pth)


if __name__ == '__main__':
    pr = argparse.ArgumentParser()
    pr.add_argument('fpath', type=str)
    pr.add_argument('--itr', '-i', type=int, default=-1, help='Only for loading env')
    # Neural network training
    pr.add_argument('--epochs', '-e', type=int, default=30)
    pr.add_argument('--batch_size', '-bs', type=int, default=512)
    pr.add_argument('--lrate', type=float, default=1e-4)
    pr.add_argument('--l2reg', type=float, default=1e-5)
    pr.add_argument('--use_gpu', action='store_true', help='Use GPU or not')
    pr.add_argument('--add_acts', action='store_true', help='Overlap consumes (s,a)')
    # Special things we might add.
    pr.add_argument('--opt', type=int, default=None, help='See documentation above')
    pr.add_argument('--buf1', type=str, default=None, help='drag-and-drop buffer 1')
    pr.add_argument('--buf2', type=str, default=None, help='drag-and-drop buffer 2')
    args = pr.parse_args()
    assert args.opt in [1, 2, 3, 4], args.opt

    # The usual, adding data dir to start.
    if not os.path.exists(args.fpath):
        print(f'{args.fpath} does not exist, pre-pending {DEFAULT_DATA_DIR}')
        args.fpath = osp.join(DEFAULT_DATA_DIR, args.fpath)
        assert osp.exists(args.fpath), args.fpath

    # We'll store results in a subdir based on the option.
    overlap_dir = osp.join(args.fpath, f'overlap_{args.opt}')
    os.makedirs(overlap_dir, exist_ok=True)
    args.overlap_dir = overlap_dir

    # Load the env, following utils/test_policy.py. Only to get the `env` portion.
    env, _ = load_policy_and_env(args.fpath, args.itr if args.itr >=0 else 'last')

    if args.opt == 1:
        OU.get_buffers_class_info(args)

        # Dump + save args.
        date = '{}'.format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))
        jstr = f'args_multiway_coarse_acts_{args.add_acts}-{date}.json'
        fpath = osp.join(overlap_dir, jstr)
        with open(fpath, 'w') as fh:
            json.dump(vars(args), fh, indent=4, sort_keys=True)
        print("\nsaved revised args at:\n\t{}\n".format(fpath))
        print(f'Loaded {len(args.bufs_t)} train/valid rollout buffer pairs.')

        overlap = Overlap(args, env)
        overlap.train()

    elif args.opt == 2:
        OU.get_buffers_class_info(args)
        classes = sorted(args.class_to_buf_noise.keys())
        n_classes = len(classes)

        # Load the pre-trained dict.
        overlap_coarse_dir = osp.join(args.fpath, f'overlap_1')
        args.overlap_coarse_dir = overlap_coarse_dir
        pretrained_pth = osp.join(overlap_coarse_dir, f'net_acts_{args.add_acts}.pth')
        pretrained_dict = torch.load(pretrained_pth)
        pretrained_dict = pretrained_dict['model_state_dict']

        # Go through and iterate upon the buffers. We just need to change a few args
        # values and then we can use essentially the same code as the coarse version.
        full_bufs_t = list(args.bufs_t)
        full_bufs_v = list(args.bufs_v)

        for i in range(n_classes):
            for j in range(i+1, n_classes):
                # First, make one of the N-choose-2 directories (do this on demand).
                name_i = args.class_to_buf_noise[i]
                name_j = args.class_to_buf_noise[j]
                assert name_i != name_j, name_i
                sub_dir = f'{name_i}__v__{name_j}'
                os.makedirs(osp.join(overlap_dir, sub_dir), exist_ok=True)

                # Second, adjust the buffers lists and `args.overlap_dir`.
                args.bufs_t = [full_bufs_t[i], full_bufs_t[j]]  # use this for train
                args.bufs_v = [full_bufs_v[i], full_bufs_v[j]]  # use this for valid
                args.overlap_dir = osp.join(overlap_dir, sub_dir)
                print('\n\n')
                print('-'*120)
                print(f'Now training {name_i} vs {name_j}')
                print(f'Train: {args.bufs_t}\nValid: {args.bufs_v}')
                print('-'*120)

                # Dump + save args. Note, args.overlap_dir updated w/subdir.
                date = '{}'.format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))
                jstr = f'args_multiway_fine_acts_{args.add_acts}-{date}.json'
                fpath = osp.join(args.overlap_dir, jstr)
                with open(fpath, 'w') as fh:
                    json.dump(vars(args), fh, indent=4, sort_keys=True)

                # Build nets and train! Note: make a copy of pretrained_dict.
                pretrained_copy = copy.deepcopy(pretrained_dict)
                overlap = Overlap(args, env, class_i=i, class_j=j,
                                  pretrained_dict=pretrained_copy)
                overlap.train()

    elif args.opt == 3:
        assert osp.exists(args.buf1), args.buf1
        assert osp.exists(args.buf2), args.buf2
        # TODO(daniel)

    elif args.opt == 4:
        pass
        # TODO(daniel)

    else:
        raise ValueError(args.opt)