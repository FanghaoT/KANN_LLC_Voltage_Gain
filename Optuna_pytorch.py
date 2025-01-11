"""
Optuna example that optimizes multi-layer perceptrons using PyTorch.

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
BATCH_SIZE = 64
CLASSES = 1
DIR = os.getcwd()
EPOCHS = 1000



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

def get_dataset(datapath):
    # Load .json Files
    # DATA = pd.read_csv(os.path.join(inputfolder,'normalized_Small.csv'))
    DATA = pd.read_csv(datapath)
    fn= DATA.loc[:,'fn'].values
    Ln = DATA.loc[:,'Ln'].values
    Q = DATA.loc[:,'Q'].values
    Vout_unit=DATA.loc[:,'Vout_unit'].values
    V_FHA = DATA.loc[:,'V_FHA'].values
    
    # Compute labels
    fn = fn.reshape((-1,1))
    Ln = Ln.reshape((-1,1))
    Q=Q.reshape((-1,1))
    Vout_unit=Vout_unit.reshape((-1,1))
    V_FHA = V_FHA.reshape((-1,1))

    temp_input = np.concatenate((fn,Ln,Q,V_FHA),axis=1)
    temp_output = np.concatenate(Vout_unit)
    
    in_tensors = torch.from_numpy(temp_input).view(-1, 4)
    out_tensors = torch.from_numpy(temp_output).view(-1, 1)


    # # # Save dataset for future use
    # np.save(os.path.join(outputfolder,"dataset.fc.in.npy"), in_tensors.numpy())
    # np.save(os.path.join(outputfolder,"dataset.fc.out.npy"), out_tensors.numpy())

    return torch.utils.data.TensorDataset(in_tensors, out_tensors)

def get_dataloader():

    inputfolder = 'OutputData'
    outputfolder = 'OutputANN'
    trainval_data_path = os.path.join(inputfolder,"normalized_big.csv")
    test_data_path = os.path.join(inputfolder,"normalized_test_big.csv")
    # Load dataset
    trainval_dataset = get_dataset(trainval_data_path)
    test_dataset = get_dataset(test_data_path)
    # Split the dataset
    train_size = int(0.8 * len(trainval_dataset))
    valid_size = len(trainval_dataset) - train_size
    # test_size = test_dataset
    train_dataset, valid_dataset= torch.utils.data.random_split(trainval_dataset, [train_size, valid_size])
    kwargs = {'num_workers': 0, 'pin_memory': True}
    # kwargs = {'num_workers': 0, 'pin_memory': True, 'pin_memory_device': "cuda:0"}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
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
