import torch
import torch.nn.functional as F
import argparse
import time
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from unet import UNet
from utils import *
from dataset import *

#prediction

parser = argparse.ArgumentParser(description='Train UNet')
parser.add_argument('--path', default='./npy/', type=str, help='test data directory')
parser.add_argument('--csv_filepath', default='./test_data.csv', type=str, help='default: test_data.csv')
parser.add_argument('--model_path', default='./models/model_2022_05_06_19_41.ckpt',type=str)

args = parser.parse_args()

timer = time.strftime('%Y_%m_%d_%H_%M', time.localtime())
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'path': args.path,
    'csv_path': args.csv_filepath,
    'model_path': args.model_path,
    'time': timer
}

test_pd = pd.read_csv(config['csv_path'])
test_data = test_pd['test'].to_list()
# print(test_data)

test_dataset = brain_dataset(config['path'], test_data, train=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model = UNet()
save_dict = torch.load(config['model_path'])
model.load_state_dict(save_dict)
model = model.cuda()

avg = tester(test_loader, model, config, device)
print(avg)
