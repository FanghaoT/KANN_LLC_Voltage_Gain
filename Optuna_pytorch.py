"""
Optuna example that optimizes multi-layer perceptrons using PyTorch.

In this example, we optimize the validation accuracy of fashion product recognition using
PyTorch and FashionMNIST. We optimize the neural network architecture as well as the optimizer
configuration. As it is too time consuming to use the whole FashionMNIST dataset,
we here use a small subset of it.

"""

import os

import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms
import pandas as pd
import numpy as np


DEVICE = torch.device("cuda:0")
BATCH_SIZE = 128
CLASSES = 1
DIR = os.getcwd()
EPOCHS = 1000
# N_TRAIN_EXAMPLES = BATCH_SIZE * 30
# N_VALID_EXAMPLES = BATCH_SIZE * 10


def define_model(trial):
    # We optimize the number of layers, hidden units and dropout ratio in each layer.
    n_layers = trial.suggest_int("n_layers", 1, 3)
    layers = []

    in_features = 4
    for i in range(n_layers):
        out_features = trial.suggest_int("n_units_l{}".format(i), 4, 128)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        p = trial.suggest_float("dropout_l{}".format(i), 0, 0)
        layers.append(nn.Dropout(p))

        in_features = out_features
    layers.append(nn.Linear(in_features, CLASSES))
    # layers.append(nn.LogSoftmax(dim=1))

    return nn.Sequential(*layers)
def get_dataset():
    # Load .json Files
    DATA = pd.read_csv('normalized_onlyPO.csv')

    f = DATA.loc[:, 'f'].values
    Cr = DATA.loc[:, 'Cr'].values
    Ln = DATA.loc[:, 'Ln'].values
    Q = DATA.loc[:, 'Q'].values
    M = DATA.loc[:, 'M'].values
    # Cr = DATA.loc[:,'Cr'].values
    # Lm = DATA.loc[:,'Lm'].values
    # fr = DATA.loc[:,'fr'].values
    # R = DATA.loc[:,'R'].values
    # Vo = DATA.loc[:,'Vo'].values

    # Compute labels
    f = f.reshape((-1, 1))
    Cr = Cr.reshape((-1, 1))
    Ln = Ln.reshape((-1, 1))
    Q = Q.reshape((-1, 1))
    M = M.reshape((-1, 1))
    # Cr = Cr.reshape((-1,1))
    # Lm = Lm.reshape((-1,1))
    # # fr = fr.reshape((-1,1))
    # R = R.reshape((-1,1))
    # Vo = Vo.reshape((-1,1))

    temp_input = np.concatenate((f, Cr, Ln, Q), axis=1)
    temp_output = np.concatenate(M)

    in_tensors = torch.from_numpy(temp_input).view(-1, 4)
    out_tensors = torch.from_numpy(temp_output).view(-1, 1)

    # # Save dataset for future use
    np.save("dataset.fc.in.npy", in_tensors.numpy())
    np.save("dataset.fc.out.npy", out_tensors.numpy())

    return torch.utils.data.TensorDataset(in_tensors, out_tensors)


def get_dataloader():
    # Load dataset
    dataset = get_dataset()

    # Split the dataset
    train_size = int(0.8 * len(dataset))
    valid_size = int(0.1 * len(dataset))

    # train_size = N_TRAIN_EXAMPLES
    # valid_size = N_VALID_EXAMPLES


    test_size = len(dataset) - train_size - valid_size
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size, test_size])
    kwargs = {'num_workers': 0, 'pin_memory': True}
    # kwargs = {'num_workers': 0, 'pin_memory': True, 'pin_memory_device': "cuda:0"}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)

    return train_loader, valid_loader, train_size, valid_size



def objective(trial):


    # Generate the model.
    model = define_model(trial).to(DEVICE)

    # Generate the optimizers.
    # optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam"])
    lr = trial.suggest_float("lr", 1e-8, 1e-1, log=True)

    DECAY_RATIO = 0.5
    # optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr*(DECAY_RATIO ** (0+ epoch_i // DECAY_EPOCH)))

    # Get the FashionMNIST dataset.
    train_loader, valid_loader, train_size, valid_size = get_dataloader()

    # Training of the model.
    for epoch in range(EPOCHS):
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr*(DECAY_RATIO ** (0+ epoch // 100)))
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # Limiting training data for faster epochs.
            if batch_idx * BATCH_SIZE >= train_size:
                break

            data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()
            output = model(data.float())
            criterion = nn.MSELoss()
            loss = criterion(output, target.float())
            loss.backward()
            optimizer.step()

        # Validation of the model.
        model.eval()
        correct = 0
        epoch_valid_loss=0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(valid_loader):
                # Limiting validation data.
                if batch_idx * BATCH_SIZE >= valid_size:
                    break
                data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)
                output = model(data.float())
                criterion = nn.MSELoss()
                loss = criterion(output, target.float())

                epoch_valid_loss += loss.item()


        accuracy = epoch_valid_loss / len(valid_loader)

        trial.report(accuracy, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=1000, timeout=60000)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
