import thop
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import random
import optuna
import numpy as np
import os
import pickle
import time
import argparse
from functools import partial
DEVICE = torch.device("cpu")
DIR = ".."
BATCHSIZE = 128
N_TRAIN_EXAMPLES = BATCHSIZE * 30
N_VALID_EXAMPLES = BATCHSIZE * 10


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def define_model(configuration):
    n_layers = configuration["n_layers"]
    layers = []
    in_features = 28 * 28
    for i in range(n_layers):
        out_features = configuration["n_units_l{}".format(i)]
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        p = configuration["dropout_{}".format(i)]
        layers.append(nn.Dropout(p))
        in_features = out_features
    layers.append(nn.Linear(in_features, 10))
    layers.append(nn.LogSoftmax(dim=1))
    return nn.Sequential(*layers)


def train_model(model, optimizer, train_loader):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.view(-1, 28 * 28).to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        F.nll_loss(model(data), target).backward()
        optimizer.step()


def eval_model(model, valid_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(valid_loader):
            data, target = data.view(-1, 28 * 28).to(DEVICE), target.to(DEVICE)
            pred = model(data).argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = correct / N_VALID_EXAMPLES
    flops, _ = thop.profile(model, inputs=(torch.randn(1, 28 * 28).to(DEVICE),), verbose=False)
    return np.log2(flops), 1 - accuracy,


def objective(configuration):
    result = {}
    train_dataset = torchvision.datasets.FashionMNIST(
        DIR, train=True, download=True, transform=torchvision.transforms.ToTensor()
    )
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(train_dataset, list(range(N_TRAIN_EXAMPLES))),
        batch_size=BATCHSIZE,
        shuffle=True,
    )
    val_dataset = torchvision.datasets.FashionMNIST(
        DIR, train=False, transform=torchvision.transforms.ToTensor()
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(val_dataset, list(range(N_VALID_EXAMPLES))),
        batch_size=BATCHSIZE,
        shuffle=True,
    )
    model = define_model(configuration).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(), configuration["lr"]
    )
    train_time_start = time.time()
    n_epoch = configuration["n_epoch"]
    for epoch in range(n_epoch):
        train_model(model, optimizer, train_loader)
    train_time = time.time() - train_time_start
    flops, error_rate = eval_model(model, val_loader)
    result["flops"] = flops
    result["train_time"] = train_time
    result["error_rate"] = error_rate
    return result


def test(seed, lexico_info):
    from flaml import tune, CFO
    set_seed(seed)

    # search space
    search_space = {
        "n_layers": tune.randint(lower=1, upper=3),
        "n_epoch": tune.randint(lower=1, upper=20),
        "n_units_l0": tune.randint(lower=4, upper=128),
        "n_units_l1": tune.randint(lower=4, upper=128),
        "n_units_l2": tune.randint(lower=4, upper=128),
        "dropout_0": tune.uniform(lower=0.2, upper=0.5),
        "dropout_1": tune.uniform(lower=0.2, upper=0.5),
        "dropout_2": tune.uniform(lower=0.2, upper=0.5),
        "lr": tune.uniform(lower=1e-5, upper=1e-1),
    }

    algo = CFO(
        space=search_space,
        metric=lexico_info["metric_priority"][0],
        mode="min",
        seed=seed,
        lexico_info=lexico_info,
    )

    analysis = tune.run(
        objective,
        config=search_space,
        low_cost_partial_config={
            "n_layers": 1,
        },
        local_dir="logs/",
        num_samples=100000000,
        time_budget_s=30,
        search_alg=algo,
        use_ray=False,
    )
    result = analysis.results
    f = open("/home/svz5418/shaokun/FLAML/test_imp/nn_test.pckl", "wb")
    pickle.dump(result, f)
    f.close()

lexico_info={}
lexico_info["metric_priority"]=["error_rate", "flops"]
lexico_info["tolerance"]={"error_rate": 0.02, "flops": 0}
lexico_info["target"]={"error_rate": 0, "flops": 0}
test(0, lexico_info)
