# --- Imports --- #
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
# from model.Kdehazing import Net
from model.mitnet import Net
from datasets.datasets import DehazingDataset
from os.path import exists, join, basename
from torchvision import transforms
from utils import to_psnr, validation
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Testing hyper-parameters for neural network')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--threads', type=int, default=15, help='number of threads for data loader to use')
parser.add_argument('--net', default='', help="path to net_Dehazing (to continue training)")
parser.add_argument('--category', help='Set image category (indoor or outdoor?)', default='UD', type=str)
# parser.add_argument("--n_GPUs", help='list of GPUs for training neural network', default=[5], type=list)
opt = parser.parse_args()

print(opt)

# ---  hyper-parameters for testing the neural network --- #
val_batch_size = opt.testBatchSize
data_threads: object = opt.threads
net_path = opt.net
category = opt.category
# GPUs_list = opt.n_GPUs

# device_ids = GPUs_list
# --- Set category-specific hyper-parameters  --- #
if category == 'UD':
    # val_data_dir = '/media/happy507gpu/DataSpace/XiaoMenglin/SOTS_remix/indoor/'
    # val_data_dir = '/media/happy507gpu/DataSpace/XiaoMenglin/dataset/I-HAZE/'
    # val_data_dir = '/media/happy507gpu/DataSpace/XiaoMenglin/dataset/O-HAZE/'
    # val_data_dir = '/media/happy507gpu/DataSpace/XiaoMenglin/sisterest/ourselves/test77/'
    # val_data_dir ='/media/happy507gpu/DataSpace/XiaoMenglin/sisterest/1/i-haze/1/'
    # val_data_dir = '/media/happy507gpu/DataSpace/XiaoMenglin/dataset/HSTS/'
    #val_data_dir = '/media/happy507/DataSpace1/XiaoMenglin/sisterest/real/'
        val_data_dir = '/media/happy507/DataSpace1/zhangzijiao/zijianshu512/val/'
    # val_data_dir = '/media/happy507gpu/DataSpace/zhangzijiao/ceshi1/'
elif category == 'outdoor':
    # val_data_dir = '/media/happy507gpu/DataSpace/XiaoMenglin/dataset/NH-HAZE/'
    val_data_dir = '/media/happy507/DataSpace1/XiaoMenglin/SOTS_remix/outdoor/'
    # val_data_dir = '/media/happy507gpu/DataSpace/XiaoMenglin/dataset/O-HAZE/'
    # val_data_dir = '/media/happy507gpu/DataSpace/zhangzijiao/O-HAZE/train/'
else:
    raise Exception('Wrong image category. Set it to indoor or outdoor for RESIDE dateset.')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# --- Validation data loader --- #ce
test_dataset = DehazingDataset(root_dir=val_data_dir, transform=transforms.Compose([transforms.ToTensor()]),
                               train=False)

test_dataloader = DataLoader(test_dataset, batch_size=val_batch_size, num_workers=data_threads, shuffle=False)

# --- Define the network --- #
model = Net()
# print(model)
# --- Multi-GPU --- #
model = model.to(device)
model = nn.DataParallel(model, device_ids=[0])
torch.backends.cudnn.enabled = False

# --- Load the network weight --- #
# model.load_state_dict(torch.load('/media/happy507/DataSpace1/zhangzijiao/weight/weight256/zijianshu/our/checkpoints2/86_point.checkpoint.pth'))
model.load_state_dict(
    torch.load('/media/happy507/DataSpace1/liuyuxin/lyx2/zijian/our/46_point.checkpoint.pth'))

# --- Use the evaluation model in testing --- #
model.eval()
print('--- Testing starts! ---')
val_psnr, val_ssim = validation(model, test_dataloader, device, category, save_tag=True)
print('val_psnr: {0:.2f}, val_ssim: {1:.4f}'.format(val_psnr, val_ssim))