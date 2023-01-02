import pickle
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torch
import numpy as np
import pickle
import config_task
import os
import os.path

def unpickle(file):
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding = 'bytes')
    return d[b'labels'], d[b'data']

class ImageFolder(data.Dataset):
    def __init__(self, roots, transform = None, loader = unpickle):
        self.transform = transform
        self.roots = roots
        self.loader = loader

    def __getitem__(self, index):
      X = []
      y = []
      for root in self.roots:
        X.append(self.loader(root)[1])
        y.append(self.loader(root)[0])
      X = np.concatenate(X, axis = 0)
      X = torch.FloatTensor(X)
      y = np.concatenate(y)
      data = X[index].view(3, 32, 32)
      if self.transform is not None:
        data = self.transform(data)
      target = y[index] 
      #print('data!', data.shape)
      return data, target
    
    def __len__(self):
        labels = []
        for root in self.roots:
            label = self.loader(root)[0]
            labels.append(label)
        targets = np.concatenate(labels)
        return len(targets)

def prepare_data_loaders(data_roots, shuffle_train = True):
    train_loaders = []
    val_loaders = []
    num_classes = [10] 
    imdb_dirs_train = [data_roots + '/' + 'data_batch_' + str(i) for i in [1, 2, 3, 4]]
    imdb_dirs_val = [data_roots + '/' + 'data_batch_5']
    means = [125.3, 123.0, 113.9]
    stds = [63.0, 62.1, 66.7]
    transform = transforms.Compose([
        #transforms.Resize(72),
        #transforms.RandomCrop(64),
        #transforms.ToTensor(),
        transforms.Normalize(means, stds)
    ])
    trainloader = torch.utils.data.DataLoader(ImageFolder(imdb_dirs_val, transform, loader = unpickle), batch_size = 128, shuffle = shuffle_train, num_workers = 2, pin_memory = True)
    valloader = torch.utils.data.DataLoader(ImageFolder(imdb_dirs_val, transform, loader = unpickle), batch_size=100, shuffle = False, num_workers = 2, pin_memory = True)
    train_loaders.append(trainloader)
    val_loaders.append(valloader)
    return train_loaders, val_loaders, num_classes