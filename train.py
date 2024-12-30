from __future__ import print_function
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model.modules.my_net import Net
# from model.mitnet import Net

from datasets.datasets2 import DehazingDataset
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from os.path import exists, join, basename
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from torchvision.models import vgg16
from model1.perceptual import LossNetwork
#from model1.pytorch_msssim import msssim
#from loss.edg_loss import edge_loss
from utils import to_psnr, validation, print_log

# 设置CUDA_VISIBLE_DEVICES为指定的GPU列表
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2'

# 打印可用的GPU设备信息
available_gpus = torch.cuda.device_count()
# print("Available GPUs:", available_gpus)
for i in range(available_gpus):
    print(f"GPU {i}: {torch.cuda.get_device_properties(i)}")

# 确保GPUs_list在可用GPU范围内
GPUs_list = [int(i) for i in os.environ["CUDA_VISIBLE_DEVICES"].split(',')]
# print("GPUs list:", GPUs_list)

# 确定当前可用的设备（GPU或CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Using device:", device)

# 构建模型
print('===> Building model')
model = Net()
# 定义损失函数
loss_function_at = nn.SmoothL1Loss().to(device)

# 多GPU设置
# print("Visible devices:", os.environ["CUDA_VISIBLE_DEVICES"])
valid_gpus = []
for gpu_id in GPUs_list:
    if gpu_id < available_gpus:
        print("Checking GPU ID:", gpu_id, "Properties:", torch.cuda.get_device_properties(gpu_id))
        valid_gpus.append(gpu_id)
    else:
        print(f"Skipping invalid GPU ID: {gpu_id}")

if len(valid_gpus) == 0:
    raise ValueError("No valid GPUs available for use.")


parser = argparse.ArgumentParser(description='Training hyper-parameters for neural network')
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=800, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.0001')
parser.add_argument("--lr_decay", type=float, default=0.75, help="Decay scale of learning rate, default=0.5")
parser.add_argument('--threads', type=int, default=15, help='number of threads for data loader to use')
parser.add_argument('--net', default='', help="path to net_Dehazing (to continue training)")
parser.add_argument('--continueEpochs', type=int, default=0, help='continue epochs')
parser.add_argument("--device", help='list of GPUs for training neural network', default='0,1,2')
parser.add_argument('--category', help='Set image category (train or val?)', default='val', type=str)
parser.add_argument('--n_GPUs', type=int, default=4, help='number of GPUs to use')
opt = parser.parse_args()
print(opt)

train_batch_size = opt.batchSize
test_batch_size = opt.testBatchSize
n_epochs = opt.nEpochs
data_threads = opt.threads


# 将模型移动到第一个有效的GPU上，并应用DataParallel
model = model.to(f"cuda:{valid_gpus[0]}")
model = nn.DataParallel(model, device_ids=valid_gpus)


# Define the MSE loss

# Set category-specific
train_data_dir = '/media/happy507/DataSpace1/zhangzijiao/ITS/train/'
train_batch_size = opt.batchSize
val_batch_size = opt.testBatchSize
train_epoch = opt.nEpochs
data_threads = opt.threads
GPUs_list = [int(i) for i in os.environ["CUDA_VISIBLE_DEVICES"].split(',')[:3]]  # 只使用前3个GPU

category = opt.category
continueEpochs = opt.continueEpochs

# Print the GPUs list
print("GPUs list:", GPUs_list)

if category == 'val':
    val_data_dir = '/media/happy507/DataSpace1/zhangzijiao/sots512/indoor/'
elif category == 'train':
    val_data_dir = '/media/happy507/DataSpace1/zhangzijiao/zijianshu/train/'
else:
    raise Exception('Wrong image category. Set it to indoor or outdoor for RESIDE dataset.')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Define the network
print('===> Building model')
model = Net()
print('model parameters:', sum(param.numel() for param in model.parameters()))

# --- Define the perceptual loss network --- #
vgg_model = vgg16(pretrained=True)
# vgg_model.load_state_dict(torch.load(os.path.join(args.vgg_model , 'vgg16.pth')))
vgg_model = vgg_model.features[:16].to(device)
for param in vgg_model.parameters():
    param.requires_grad = False
loss_function_at = nn.SmoothL1Loss().to(device)
loss_network = LossNetwork(vgg_model)
loss_network.eval()
# Multi-GPU setup
model = model.to(f"cuda:{valid_gpus[0]}")
model = nn.DataParallel(model, device_ids=valid_gpus)

# optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999)) #youhuaqi
scheduler = StepLR(optimizer, step_size=train_epoch // 2, gamma=0.1)

# Load training data and validation/test data
train_dataset = DehazingDataset(root_dir=train_data_dir, transform=transforms.Compose([transforms.ToTensor()]))
train_dataloader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, num_workers=data_threads, shuffle=True)

test_dataset = DehazingDataset(root_dir=val_data_dir, transform=transforms.Compose([transforms.ToTensor()]), train=False)
test_dataloader = DataLoader(test_dataset, batch_size=val_batch_size, num_workers=data_threads, shuffle=False)

old_val_psnr, old_val_ssim = validation(model, test_dataloader, device, category)
print('old_val_psnr: {0:.2f}, old_val_ssim: {1:.4f}'.format(old_val_psnr, old_val_ssim))

for epoch in range(1 + opt.continueEpochs, opt.nEpochs + 1 + opt.continueEpochs):
    print("Training...")

    scheduler.step()
    epoch_loss = 0
    psnr_list = []
    for iteration, inputs in enumerate(train_dataloader, 1):
        haze, gt = Variable(inputs['hazy_image']), Variable(inputs['clear_image'])
        haze = haze.to(device)
        gt = gt.to(device)

        optimizer.zero_grad()

        model.train()
        dehaze = model(haze)
        Loss1 = loss_function_at(dehaze, gt)
        # EDGE_loss = edge_loss(dehaze, gt, device)
        perceptual_loss = loss_network(dehaze, gt)
        Loss = Loss1  #+0.01 * perceptual_loss
        epoch_loss += Loss
        Loss.backward()
        optimizer.step()

        if iteration % 100 == 0:
            print("===>Epoch[{}]({}/{}): Loss: {:.4f} ".format(epoch, iteration, len(train_dataloader), Loss.item(), Loss1.item(), perceptual_loss.item()))

        psnr_list.extend(to_psnr(dehaze, gt))

    train_psnr = sum(psnr_list) / len(psnr_list)
    save_checkpoints = '/media/happy507/DataSpace1/liuyuxin/downloads/1/'

    torch.save(model.state_dict(), './checkpoints/{}_haze.pth'.format(category))

    model.eval()
    val_psnr, val_ssim = validation(model, test_dataloader, device, category)

    if epoch % 1 == 0:
        print('save')
        save_checkpoints_dir = save_checkpoints + str(epoch) + '_point.checkpoint.pth'
        torch.save(model.state_dict(), save_checkpoints_dir)

    print_log(epoch + 1, train_epoch, train_psnr, val_psnr, val_ssim, category)
