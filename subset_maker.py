#Yuliia Frund

import torch
from torch.utils.data import Dataset
import dataset_maker
from pathlib import Path

class My_DataSet(Dataset):
  def __init__(self, data, device='cpu'):
    self.data = data

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    return self.data[idx]


def split_data(file, train, device='cpu'):
  """
  :param file: data to split
  :param train: percentage of train data, e.g. 0.8
  :param device: can be set to cuda
  :return: train dataset, test dataset lists of tuples
  """
  data = My_DataSet(file, device=device)

  train_size = int(train * len(data))
  test_size = len(data) - train_size
  train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])

  return train_dataset, test_dataset


def main():

  device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  synth_files = Path('synthetic_speech_commands/')
  nat_files = Path('natural_speech_commands/')
  maximum = min(dataset_maker.balance_data(synth_files), dataset_maker.balance_data(nat_files))

  data_synth = dataset_maker.get_features(synth_files, maximum)
  data_nat = dataset_maker.get_features(nat_files, maximum)

  train_dataset, test_dataset = split_data(data_synth, 0.8, device=device)
  torch.save(train_dataset, 'train_test_datasets/train_dataset_synth.pth')
  torch.save(test_dataset, 'train_test_datasets/test_dataset_synth.pth')

  train_dataset, test_dataset = split_data(data_nat, 0.8, device=device)
  torch.save(train_dataset, 'train_test_datasets/train_dataset_nat.pth')
  torch.save(test_dataset, 'train_test_datasets/test_dataset_nat.pth')


if __name__ == '__main__':
    main()

