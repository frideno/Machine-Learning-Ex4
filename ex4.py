# omri fridental, gal politzer
# 323869545, 212257729

"""Data loading code:
"""
import os
import os.path

import soundfile as sf
import librosa
import numpy as np
import torch
import torch.utils.data as data

AUDIO_EXTENSIONS = [
    '.wav', '.WAV',
]


def is_audio_file(filename):
    return any(filename.endswith(extension) for extension in AUDIO_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    spects = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_audio_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    spects.append(item)
    return spects


def spect_loader(path, window_size, window_stride, window, normalize, max_len=101):
    y, sr = sf.read(path)
    # n_fft = 4096
    n_fft = int(sr * window_size)
    win_length = n_fft
    hop_length = int(sr * window_stride)

    # STFT
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length, window=window)
    spect, phase = librosa.magphase(D)

    # S = log(S+1)
    spect = np.log1p(spect)

    # make all spects with the same dims
    # TODO: change that in the future
    if spect.shape[1] < max_len:
        pad = np.zeros((spect.shape[0], max_len - spect.shape[1]))
        spect = np.hstack((spect, pad))
    elif spect.shape[1] > max_len:
        spect = spect[:, :max_len]
    spect = np.resize(spect, (1, spect.shape[0], spect.shape[1]))
    spect = torch.FloatTensor(spect)

    # z-score normalization
    if normalize:
        mean = spect.mean()
        std = spect.std()
        if std != 0:
            spect.add_(-mean)
            spect.div_(std)

    return spect


class GCommandLoader(data.Dataset):
    """A google command data set loader where the wavs are arranged in this way: ::
        root/one/xxx.wav
        root/one/xxy.wav
        root/one/xxz.wav
        root/head/123.wav
        root/head/nsdf3.wav
        root/head/asd932_.wav
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        window_size: window size for the stft, default value is .02
        window_stride: window stride for the stft, default value is .01
        window_type: typye of window to extract the stft, default value is 'hamming'
        normalize: boolean, whether or not to normalize the spect to have zero mean and one std
        max_len: the maximum length of frames to use
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        spects (list): List of (spects path, class_index) tuples
        STFT parameter: window_size, window_stride, window_type, normalize
    """

    def __init__(self, root, transform=None, target_transform=None, window_size=.02,
                 window_stride=.01, window_type='hamming', normalize=True, max_len=101):
        classes, class_to_idx = find_classes(root)
        spects = make_dataset(root, class_to_idx)
        if len(spects) == 0:
            raise (RuntimeError("Found 0 sound files in subfolders of: " + root + "Supported audio file extensions are: " + ",".join(AUDIO_EXTENSIONS)))

        self.root = root
        self.spects = spects
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = spect_loader
        self.window_size = window_size
        self.window_stride = window_stride
        self.window_type = window_type
        self.normalize = normalize
        self.max_len = max_len

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (spect, target) where target is class_index of the target class.
        """
        path, target = self.spects[index]
        spect = self.loader(path, self.window_size, self.window_stride, self.window_type, self.normalize, self.max_len)
        if self.transform is not None:
            spect = self.transform(spect)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return spect, target

    def __len__(self):
        return len(self.spects)



"""
Network class:
contains 2 convolution layers, followed by 2 linear layers.
"""

import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        
        self.fc1 = nn.Linear(in_features=12 * 37 * 22, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=30)

    def forward(self, t):

      # (2) hidden conv layer
      t = self.conv1(t)
      t = F.relu(t)
      t = F.max_pool2d(t, kernel_size=2, stride=2)

      # (3) hidden conv layer
      t = self.conv2(t)
      t = F.relu(t)
      t = F.max_pool2d(t, kernel_size=2, stride=2)

      # (4) hidden linear layer
      t = t.reshape(-1, 12 * 37 * 22)
      t = self.fc1(t)
      t = F.relu(t)

      # (5) hidden linear layer
      t = self.fc2(t)
      t = F.relu(t)
      
      # (6) output layer
      t = self.out(t)
      #t = F.log_softmax(t, dim=1)

      return t



"""Train data:
each example:
- zeroes grad
- feed forward, and compute loss
- backprop
- return delta

overall, Trainer class train for num_epoches epoches with learning rate eta.
"""

import torch.nn as nn
import torch.nn.functional as F
import torch
from os import listdir
from os.path import isfile, join
from torchvision import transforms
        
def get_num_correct(preds, labels):
  return preds.argmax(dim=1).eq(labels).sum().item()

class Trainer:
  def __init__(self, network, eta):
    self.net = network
    self.optimizer = torch.optim.SGD(self.net.parameters(), lr=eta)
    self.criterion = nn.CrossEntropyLoss()
    
  def predict(self, x):
    n, i = x.max(0)
    return i.item()


  def train(self, training_set, num_epoches, validation_set=None):

    
    for epoch in range(num_epoches):
      total = 0
      pct = 0
      for i, (inputs, lables) in enumerate(training_set):

        # convert to gpu:

        inputs = inputs.to(device)
        lables = lables.to(device)

        # forward, backward, optimize:
        self.optimizer.zero_grad()
        outputs = self.net(inputs)
        loss = self.criterion(outputs, lables)
        loss.backward()

        self.optimizer.step()
        
        pct += get_num_correct(outputs, lables);
        total += len(outputs)

      print('{0} (training): correct at = {1} out of {2}'.format(epoch, pct, total))
      if validation_set: self.validate(validation_set)

  def validate(self, validation_set):
    total = 0
    pct = 0
    for i, (inputs, lables) in enumerate(validation_set):

      # convert to gpu:
      inputs = inputs.to(device)
      lables = lables.to(device)
      outputs = self.net(inputs)

      pct += get_num_correct(outputs, lables);
      total += len(outputs)

    print('validation: correct at = {0} out of {1}'.format(pct, total))


  def test(self, test_set, file_names):

  
    for file, (inputs, lables) in zip(file_names, test_set):

      # name of file:
      filename = file[0].split("/")[-1]
      
      # convert to gpu:
      inputs = inputs.to(device)
      outputs = self.net(inputs)
      
      y_hat = self.predict(outputs[0])
      print(str(filename) + ', ' + str(y_hat))
        
    




# main:
# open train, validation, test sets.

# add normalization:
normalize = transforms.Compose([
    transforms.Normalize(mean=[0], std=[1])
])

trainset = GCommandLoader('./data/train', transform=normalize)
validset = GCommandLoader('./data/valid', transform=normalize)
testset = GCommandLoader('./data/test', transform=normalize)

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
network = Network().to(device)

train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=100, shuffle=True,
        num_workers=20, pin_memory=True, sampler=None)

validation_loader = torch.utils.data.DataLoader(
        validset, batch_size=100, shuffle=True,
        num_workers=20, pin_memory=True, sampler=None)

test_loader = torch.utils.data.DataLoader(
        testset, batch_size=1, shuffle=None,
        num_workers=0, pin_memory=True, sampler=None)

# train and validate:
trainer = Trainer(network, eta=0.05)
trainer.train(train_loader, 25, validation_loader)
trainer.validate(validation_loader)

# test:
trainer.test(test_loader, testset.spects)

!for d in $(ls ./data/train); do bash aug.sh "$d"; done
