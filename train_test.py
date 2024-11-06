#Yuliia Frund

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import model_conv
from subset_maker import My_DataSet
import test


def training(labels, train, epochs, asr_model, batch_size, save_as, device):
    """
    :param labels: list
    :param train: list of tuples [(input, true label)]
    :param epochs: integer
    :param asr_model: model to train
    :param batch_size: integer
    :param save_as: name of trained model - string
    :param device: cuda or cpu, string
    """
    asr_model.train()

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(asr_model.parameters())
    # optimizer = optim.SGD(asr_model.parameters(), lr=0.01, momentum=0.9)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    # scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, epochs=epochs, steps_per_epoch=int(len(train_loader)), anneal_strategy='linear')


    for epoch in range(epochs):
        epoch_losses = []
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{epochs}', unit='batch') as pbar:
            for inputs, targets in train_loader:

                optimizer.zero_grad()
                preds = asr_model(inputs).squeeze()
                targets = torch.tensor([labels.index(targ) for targ in targets]).to(device)
                loss = loss_function(preds, targets)
                epoch_losses.append(loss.item())
                loss.backward()
                optimizer.step()
                # scheduler.step()
                pbar.set_postfix({'loss': np.mean(epoch_losses)})
                pbar.update(1)


    torch.save(asr_model.state_dict(), save_as)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def move_to_device(data, device):
        return [(sample[0].to(device), sample[1]) for sample in data]

    with open('labels.txt', 'r', encoding='utf-8') as f:
        labels = f.read().splitlines()

    train_synth = torch.load('train_test_datasets/train_dataset_synth.pth')
    train_synth = move_to_device(train_synth, device)
    test_synth = torch.load('train_test_datasets/test_dataset_synth.pth')
    test_synth = move_to_device(test_synth, device)
    train_nat = torch.load('train_test_datasets/train_dataset_nat.pth')
    train_nat = move_to_device(train_nat, device)
    test_nat = torch.load('train_test_datasets/test_dataset_nat.pth')
    test_nat = move_to_device(test_nat, device)

    # mod = model_conv.ASR_Model(len(labels), device).to('cuda')
    # training(labels, train_synth, 400, mod, 124, 'models/asr_conv_synth_400.pth', device)
    #
    # mod = model_conv.ASR_Model(len(labels), device).to('cuda')
    # training(labels, train_nat, 400, mod, 124, 'models/asr_conv_nat_400.pth', device)

    mod_nat = model_conv.ASR_Model(len(labels), device)
    mod_nat.load_state_dict(torch.load('models/asr_conv_nat_200.pth', map_location=torch.device('cpu')))
    nat_score = test.roc_auc(test_synth, labels, mod_nat)
    print(f'Natural speech data, convolutional layers, roc-auc score: {nat_score[0]}, accuracy: {nat_score[1]}')

    mod_synth = model_conv.ASR_Model(len(labels), device)
    mod_synth.load_state_dict(torch.load('models/asr_conv_synth_200.pth', map_location=torch.device('cpu')))
    synth_score = test.roc_auc(test_nat, labels, mod_synth)
    print(f'Synthetic data, convolutional layers, roc-auc score: {synth_score[0]}, accuracy: {synth_score[1]}')



if __name__ == '__main__':
    main()




