import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import os
import itertools
import random
import numpy as np
from pswe import PSWE
from settransformer import PMA, SAB, ISAB
from fspool import FSPool
from collections import defaultdict
from tqdm import tqdm
from ModelNet40_data import ModelNet40
from torch.multiprocessing import Pool, Process, set_start_method
torch.multiprocessing.set_start_method('spawn', force=True)
import time

batch_size = 32
base_random_seed = 2054
num_epochs = 200
output_dim = 256
lr = 1e-3
num_classes = 40

def layer(layer_type, input_dim, output_dim, non_linearity=None):
    if layer_type == 'MLP':
        if non_linearity is None:
            return nn.Sequential(nn.Linear(input_dim, output_dim))
        else:
            return nn.Sequential(nn.Linear(input_dim, output_dim), non_linearity)
    elif layer_type == 'ISAB':
        return ISAB(dim_in=input_dim, dim_out=output_dim, num_heads=4, num_inds=16)

class Backbone(nn.Module):
    def __init__(self, backbone_type, input_dim):
        super(Backbone, self).__init__()
        self.layers = nn.ModuleList()

        if backbone_type == 'MLP':
            hidden_layers = [256] * 2
            output_dim = 256
        elif backbone_type == 'ISAB':
            hidden_layers = [256]
            output_dim = 256

        self.layers.append(layer(backbone_type, input_dim, hidden_layers[0], non_linearity=nn.ReLU()))
        for i in range(1, len(hidden_layers)):
            self.layers.append(layer(backbone_type, hidden_layers[i-1], hidden_layers[i], non_linearity=nn.ReLU()))
        self.layers.append(layer(backbone_type, hidden_layers[-1], output_dim))
        self.backbone = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.backbone(x)

class Pooling(nn.Module):
    def __init__(self, pooling, d_in=1, num_projections=1, num_ref_points=1):

        super(Pooling, self).__init__()

        self.pooling = pooling
        # pooling mechanism
        if 'PSWE' in pooling:
            self.pool = PSWE(d_in, num_ref_points, num_projections)
            self.num_outputs = num_projections
        elif 'PMA' in pooling:
            self.pool = PMA(d_in, num_seeds=num_ref_points, num_heads=1)
            self.num_outputs = d_in
        elif 'FSPool' in pooling:
            self.pool = FSPool(d_in, num_ref_points)
            self.num_outputs = d_in
        elif pooling == 'GAP':
            self.num_outputs = d_in
        else:
            raise ValueError('Pooling type {} not implemented!'.format(pooling))

    def forward(self, P):
        """
        Input size: B x N x d_in
        B: batch size, N: # elements per set, d_in: # features per element

        Output size: B x self.num_outputs
        """

        # apply pooling
        if self.pooling == 'GAP':
            U = torch.mean(P, dim=1)
        else:
            U = self.pool(P).view(-1, self.num_outputs)

        return U

def train_test(b, e, n, experiment_ID, num_points_per_set, gpu_index):

    device = torch.device('cuda:' + str(gpu_index) if torch.cuda.is_available() else 'cpu')

    # create results directory if it doesn't exist
    results_dir = './results/modelnet40/{}/{}_{}_{}/'.format(num_points_per_set, b, e, n)
    os.makedirs(results_dir, exist_ok=True)

    random_seed = int(base_random_seed + experiment_ID)

    res_path = results_dir + '{}.json'.format(random_seed)

    print("params", b, e, n, experiment_ID, num_points_per_set, gpu_index)

    # Set the random seed
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # get the datasets
    phases = ['train', 'test']
    dataset = {}
    for phase in phases:
        dataset[phase] = ModelNet40(num_points_per_set, partition=phase)

    # create the dataloaders
    loader = {}
    for phase in phases:
        if phase == 'train':
            shuffle = True
        else:
            shuffle = False
        loader[phase] = DataLoader(dataset[phase], batch_size=batch_size, shuffle=shuffle)

    # create the modules
    backbone = Backbone(b, input_dim=3)
    pooling = Pooling(e, d_in=output_dim, num_projections=n, num_ref_points=num_points_per_set)
    classifier = nn.Linear(pooling.num_outputs, num_classes)

    backbone.to(device)
    pooling.to(device)
    classifier.to(device)

    # start training
    criterion = nn.CrossEntropyLoss()

    params = list(backbone.parameters()) + \
             list(pooling.parameters()) + \
             list(classifier.parameters())
    optim = Adam(params, lr=lr)
    scheduler = StepLR(optim, step_size=50, gamma=0.5)

    epochMetrics = defaultdict(list)
    for epoch in tqdm(range(num_epochs)):

        for phase in phases:

            if phase == 'train':
                backbone.train()
                pooling.train()
                classifier.train()
            else:
                backbone.eval()
                pooling.eval()
                classifier.eval()

            loss_ = []
            acc_ = []

            for i, data in enumerate(loader[phase]):

                # zero the parameter gradients
                optim.zero_grad()

                x, y = data

                x = x.to(device).to(torch.float)
                y = y.to(device).squeeze()

                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

                    # pass the sets through the backbone and pooling
                    z = backbone(x)
                    v = pooling(z)
                    logits = classifier(v)
                    loss = criterion(logits, y)

                    acc = (1. * (torch.argmax(logits, dim=1) == y)).mean().item()

                    # backpropogation only in training phase
                    if phase == 'train':
                        # Backward pass
                        loss.backward()
                        # 1-step gradient descent
                        optim.step()

                # save losses and accuracies
                loss_.append(loss.item())
                acc_.append(acc)

            epochMetrics[phase, 'loss'].append(np.mean(loss_))
            epochMetrics[phase, 'acc'].append(np.mean(acc_))

        scheduler.step()

        # save intermediate results so far
        torch.save(epochMetrics, results_dir + '{}.json'.format(random_seed))

    return epochMetrics


def main():

    backbones = ['MLP', 'ISAB']
    embeddings = ['GAP', 'PMA', 'FSPool', 'PSWE']
    num_projections_range = [1, 4, 16, 64, 256, 1024]
    experiment_IDs = list(1e4 * (1 + np.arange(10))) # 10 random seeds
    num_points_per_set_range = [1024] # 1024 points per set

    params_all = list(itertools.product(backbones,
                                        embeddings,
                                        num_projections_range,
                                        experiment_IDs,
                                        num_points_per_set_range))

    # remove redundant parameters (other pooling methods with different # of slices)
    params = []
    i = 0
    for p in params_all:
        if (p[2] == num_projections_range[0]) or ('SWE' in p[1] and p[2] > num_projections_range[0]):
            params.append(p + (0, ))
            i += 1

    print(params)
    print('Number of parameter/random seed combinations:', len(params))

    num_processes = 10
    print('Now starting the code using {} parallel processes...'.format(min(num_processes, len(params))))

    pool = Pool(num_processes)
    all_results = pool.starmap(train_test, params)
    pool.close()
    pool.join()

if __name__ == '__main__':
    main()
