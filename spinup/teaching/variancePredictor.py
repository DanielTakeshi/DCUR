from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import spinup.algos.pytorch.td3.core as core
from spinup.utils.logx import EpochLogger
import torch.nn as nn
import torch.optim as optim

class DataBuffer:
    def __init__ (self, path):
        self.replay_samples, self.variance_labels = torch.load(path)
    def sample_batch(self, batch_size = 32):
        idxs = np.random.randint(0, len(self.replay_samples), size=batch_size)
        data = np.array([self.replay_samples[i] for i in idxs])
        return torch.from_numpy(data), torch.from_numpy(np.take(self.variance_labels, idxs))

class VarianceNet(nn.Module):
    def __init__(self, obs_space, dropout, nb_classes=1):
        """ Variance predictor network, taken from Daniel's rlpyt EpsilonNet. 
            Currently, it is used only in non-discretized settings.
        """
        super(VarianceNet, self).__init__()
        self.obs_space = obs_space
        self.dropout = nn.Dropout(dropout)
        self.hidden1 = nn.Linear(11, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, nb_classes)

    def forward(self, x):
        """Forward function to compute net output given `x`.
        In normal DQN, just call `net(states)`. The first dimension of `x` is
        the batch size, and the remaining should be the `obs_space`. ASSUMES
        WE DIVIDE BY 255 HERE! If we do it on the whole dataset earlier, due to
        the large size we can get close to or exceed 64G RAM.
        """

        _batch_size = x.shape[0]
        x = x.float()
        x = self.hidden1(x)
        x = nn.functional.relu(x)
        x = nn.functional.relu(self.fc4(x))
        x = self.dropout(x)
        x = self.fc5(x)
        x = nn.functional.tanh(x)
        return x

class SampleStreamVariance:
    def __init__(self, net, log_dir):
        self.log_dir = log_dir
        self.net = net
        self.opt = optim.Adam(self.net.parameters(), lr=0.0001, weight_decay=0.00005)
        self.criterion = nn.MSELoss()
        self.errors = []
    def train(self, batch_size, train_buffer, test_buffer, n_cycles):
        for i in range(n_cycles):
            self.optimize(batch_size, train_buffer)
            evaluate_error = self.evaluate(batch_size, test_buffer)
            print('mse in iteration ', i, evaluate_error)
            self.errors.append(evaluate_error)
    def optimize(self, batch_size, data_buffer):
        inputs, labels = data_buffer.sample_batch(batch_size)
        self.net.train()
        self.opt.zero_grad()
        outputs = self.net(inputs)
        with torch.set_grad_enabled(True):
            labels = labels.float()
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.opt.step()
    def evaluate(self, batch_size, data_buffer):
        inputs, labels = data_buffer.sample_batch(batch_size)
        predictions = self.net(inputs)
        mse = np.mean(np.square(predictions.detach().numpy() - labels.detach().numpy()))
        labels = labels.detach().numpy()
        return mse

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str)
    parser.add_argument('--test_data', type=str)
    args = parser.parse_args()
    dataBuffer = DataBuffer(args.train_data)
    testBuffer = DataBuffer(args.test_data)
    net = VarianceNet(dataBuffer.sample_batch()[0].shape[1:], 0)
    sampler = SampleStreamVariance(net, '/home/abhinavg/spinningup/data/variancePredictor')
    sampler.train(32, dataBuffer, testBuffer, 100000)
    print('FINISHED TRAINING\n\n\n')
    print(sampler.evaluate(500, testBuffer))
#     with open('errorfile.txt', 'wb') as f:
#         np.save(f, np.array(sampler.errors))

