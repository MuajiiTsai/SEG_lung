import unet
import torch
import torch.nn as nn
import os
import argparse
import matplotlib.pyplot as plt
import time
from dataset import *
from utils import *
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='./npy', type=str)
parser.add_argument('--epoch', default=1, type=int)
parser.add_argument('--seed', default=1027, type=int)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--learning_rate', default=1e-5, type=float)
args = parser.parse_args()

timer = time.strftime('%Y_%m_%d_%H_%M', time.localtime())
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'seed': args.seed,      # Your seed number, you can pick your lucky number. :) #1027
    'valid_ratio': 0.2,   # validation_size = train_size * valid_ratio
    'test_ratio': 0.2,   # test_size = train_size * test_ratio
    'n_epochs': args.epoch,     # Number of epochs.            
    'batch_size': args.batch_size, 
    'learning_rate': args.learning_rate,              
    'weight_decay': 1e-4,
    'path': args.path,
    'early_stop': 10,    # If model has not improved for this many consecutive epochs, stop training.
    'time': timer,     
    'save_path': f'./models/model_{timer}.ckpt'  # Your model will be saved here.
}

same_seed(config['seed'])

whole_data = os.listdir(f"{config['path']}/MRI")
whole_dataset = brain_dataset(config['path'], whole_data, train=False)
whole_loader =  DataLoader(whole_dataset, batch_size=1, shuffle=False)


train_data, valid_data, test_data = dataset_split(whole_data, config['valid_ratio'], config['test_ratio'])


train_dataset, valid_dataset, test_dataset = brain_dataset(config['path'], train_data), brain_dataset(config['path'], valid_data), brain_dataset(config['path'], test_data, train=False)

train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model = unet.UNet()
model = model.cuda()

best_loss = trainer(train_loader, valid_loader, model, config, device)

avg = tester(test_loader, model, config, device)
print(avg)

"""
TODO: 
"""
