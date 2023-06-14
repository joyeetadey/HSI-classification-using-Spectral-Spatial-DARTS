## imports
import os
import sys
import time
import glob
import numpy as np
import torch
import math
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import random
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import scipy.io as sio
from torch.autograd import Variable
from skimage.restoration import denoise_tv_chambolle
import numpy as np
from sklearn.decomposition import PCA
from skimage.restoration import denoise_tv_chambolle
import scipy.io as sio
import torch
import pywt

# imports from other files
from model_search import Network
from architect import Architect
from utils import cutout


parser = argparse.ArgumentParser("HSI")
parser.add_argument('--num_class', type=int, default=16, help='classes of HSI dataset') # number of classes
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.0025, help='init learning rate') # learning rate
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=100, help='num of training epochs') # number of epochs
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=2, help='total number of layers')
parser.add_argument('--unrolled', action='store_true', default=True, help='use one-step unrolled validation loss')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--cutout', action='store_true', default=True, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=2, help='cutout length')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=3e-4, help='weight decay for arch encoding')
parser.add_argument('--Train', default=200, help='Train_num')
parser.add_argument('--Valid', default=100, help='Valid_num')
parser.add_argument('--num_cut', type=int, default=10, help='band cutout')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
args.manualSeed = random.randint(1, 10000)


log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler('/content/result/log.txt')
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


# read data
image_file = '/content/KSC_corrected.mat'
label_file = '/content/KSC_gt.mat'

image = sio.loadmat(image_file)
KSC = image['KSC']

label = sio.loadmat(label_file)
GroundTruth = label['KSC_gt']

KSC = (KSC - np.min(KSC)) / (np.max(KSC) - np.min(KSC))

## data shape
print("data shape: ",KSC.shape)

data = KSC

# # Apply wavelet decomposition of level 4

# data_wav = np.empty_like(data)
# for i in range(data.shape[2]):
#     cA, (cH, cV, cD) = pywt.dwt2(data[:,:,i], 'db4')
#     cA_resized = F.interpolate(torch.from_numpy(cA)[None,None,:,:], size=(512,217), mode='bilinear', align_corners=False)
#     data_wav[:,:,i] = cA_resized.numpy()[0,0,:,:]

# print("after wavelet decomposition level 1:",data_wav.shape)

# data_wav1 = np.empty_like(data_wav)
# for i in range(data_wav1.shape[2]):
#     cA1, (cH1, cV1, cD1) = pywt.dwt2(data_wav1[:,:,i], 'db4')
#     cA_resized1 = F.interpolate(torch.from_numpy(cA1)[None,None,:,:], size=(512,217), mode='bilinear', align_corners=False)
#     data_wav1[:,:,i] = cA_resized1.numpy()[0,0,:,:]

# print("after wavelet decomposition level 2:",data_wav1.shape)

# data_wav2 = np.empty_like(data_wav1)
# for i in range(data_wav1.shape[2]):
#     cA2, (cH2, cV2, cD2) = pywt.dwt2(data_wav1[:,:,i], 'db4')
#     cA_resized2 = F.interpolate(torch.from_numpy(cA2)[None,None,:,:], size=(512,217), mode='bilinear', align_corners=False)
#     data_wav2[:,:,i] = cA_resized2.numpy()[0,0,:,:]

# print("after wavelet decomposition level 3:",data_wav2.shape)

# data_wav3 = np.empty_like(data_wav2)
# for i in range(data_wav2.shape[2]):
#     cA3, (cH3, cV3, cD3) = pywt.dwt2(data_wav3[:,:,i], 'db4')
#     cA_resized3 = F.interpolate(torch.from_numpy(cA3)[None,None,:,:], size=(512,217), mode='bilinear', align_corners=False)
#     data_wav3[:,:,i] = cA_resized3.numpy()[0,0,:,:]

# print("after wavelet decomposition level 4:",data_wav3.shape)



# KSC_data = data_wav3

# # Apply PCA for dimensionality reduction
# num_components = 204
# pca = PCA(n_components=num_components, whiten=True)
# KSC_data_pca = pca.fit_transform(KSC_data.reshape(-1, KSC_data.shape[-1]))
# KSC_data_pca = KSC_data_pca.reshape(KSC_data.shape[:-1] + (num_components,))

# print("after PCA",KSC_data_pca.shape)

# # Apply 3D-Hyperspectral Total Variation denoising
# lambda_tv = 0.1  # regularization parameter
# KSC_data_denoised = np.zeros_like(KSC_data_pca)
# for i in range(num_components):
#     KSC_data_denoised[..., i] = denoise_tv_chambolle(KSC_data_pca[..., i], weight=lambda_tv, multichannel=False)
# KSC_data_denoised = torch.from_numpy(KSC_data_denoised).float()

# # Print the shape of the denoised data
# print("after 3dtv",KSC_data_denoised.shape)


[nRow, nColumn, nBand] = KSC.shape


nTrain = args.Train
nValid = args.Valid
num_class = int(np.max(GroundTruth))
HSI_CLASSES = num_class

HalfWidth = 16
Wid = 2 * HalfWidth

[row, col] = GroundTruth.shape

NotZeroMask = np.zeros([row, col])
NotZeroMask[HalfWidth + 1: -1 - HalfWidth + 1, HalfWidth + 1: -1 - HalfWidth + 1] = 1
G = GroundTruth * NotZeroMask

[Row, Column] = np.nonzero(G)
nSample = np.size(Row)


def main(seed, cut):

  print("Seed",seed)
  args.cutout = cut
  np.random.seed(seed)
  RandPerm = np.random.permutation(nSample)

  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  args.cutout=cut
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.manualSeed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.manualSeed)

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  model = Network(nBand, args.init_channels, HSI_CLASSES, args.layers, criterion)
  model = model.cuda()
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  optimizer = torch.optim.Adam(model.parameters(),args.learning_rate, weight_decay=args.weight_decay)
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.epochs // 2, 0.5)

  architect = Architect(model, args)

  min_valid_loss = 100
  lr_max = 0.0025
  lr_min = 0.0010

  genotype = model.genotype()
  logging.info('genotype = %s', genotype)
  for epoch in range(1, args.epochs+1):

    imdb = {}
    imdb['data'] = np.zeros([Wid, Wid, nBand, nTrain + nValid], dtype=np.float32)
    imdb['Labels'] = np.zeros([nTrain + nValid], dtype=np.int64)
    imdb['set'] = np.zeros([nTrain + nValid], dtype=np.int64)

    for iSample in range(nTrain):

        yy = KSC[Row[RandPerm[iSample]] - HalfWidth: Row[RandPerm[iSample]] + HalfWidth, \
             Column[RandPerm[iSample]] - HalfWidth: Column[RandPerm[iSample]] + HalfWidth, :]
        if args.cutout:
            xx = cutout(yy, args.cutout_length, args.num_cut)

            imdb['data'][:, :, :, iSample] = xx
        else:
            imdb['data'][:, :, :, iSample] = yy

        imdb['Labels'][iSample] = G[Row[RandPerm[iSample]], Column[RandPerm[iSample]]].astype(np.int64)

    for iSample in range(nValid):
        imdb['data'][:, :, :, iSample + nTrain] = KSC[Row[RandPerm[iSample + nTrain]] - HalfWidth: Row[RandPerm[
              iSample + nTrain]] + HalfWidth, \
                                                    Column[RandPerm[iSample + nTrain]] - HalfWidth: Column[RandPerm[
                                                        iSample + nTrain]] + HalfWidth, :]
        imdb['Labels'][iSample + nTrain] = G[Row[RandPerm[iSample + nTrain]],
                                            Column[RandPerm[iSample + nTrain]]].astype(np.int64)
    print('Data is OK.')
    imdb['Labels'] = imdb['Labels'] - 1

    imdb['set'] = np.hstack((np.ones([nTrain]), 3 * np.ones([nValid]))).astype(np.int64)
    #checking data
 
    Xtrain=imdb['data'][:,:,:,:nTrain]
    ytrain=imdb['Labels'][:nTrain]
    print('Xtrain :',Xtrain.shape)
    print('yTrain:',ytrain.shape)
    Xtest=imdb['data']
    ytest=imdb['Labels']
    print('Xtest :',Xtest.shape)
    print('ytest:',ytest.shape)

    Xtrain=Xtrain.transpose(3,2,0,1)
    Xtest=Xtest.transpose(3,2,0,1)
    print('after Xtrain shape:',Xtrain.shape)
    print('after Xtest shape:',Xtest.shape)

    ####Training
    class TrainDS(torch.utils.data.Dataset): 
        def __init__(self):
            self.len = Xtrain.shape[0]
            self.x_data = torch.FloatTensor(Xtrain)
            self.y_data = torch.LongTensor(ytrain)        
        def __getitem__(self, index):
           
            return self.x_data[index], self.y_data[index]
        def __len__(self): 
            
            return self.len

    """ Testing dataset"""
    class TestDS(torch.utils.data.Dataset): 
        def __init__(self):
            self.len = Xtest.shape[0]
            self.x_data = torch.FloatTensor(Xtest)
            self.y_data = torch.LongTensor(ytest)
        def __getitem__(self, index):
            
            return self.x_data[index], self.y_data[index]
        def __len__(self): 
            
            return self.len

    #trainloader testloader
    trainset = TrainDS()
    testset  = TestDS()
    train_queue = torch.utils.data.DataLoader(dataset=trainset, batch_size=200, shuffle=True, num_workers=0)
    valid_queue  = torch.utils.data.DataLoader(dataset=testset,  batch_size=200, shuffle=False, num_workers=0)
    # train_dataset = dset(imdb, train=True)

    # valid_dataset = dset(imdb, train=False)

    # train_queue = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
    #                                           shuffle=True, pin_memory=True, num_workers=0)
    # valid_queue = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size,
    #                                           shuffle=True, pin_memory=True, num_workers=0)

    tic = time.time()
    # scheduler.step()
    # lr = scheduler.get_lr()[0]
    lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(epoch / args.epochs * math.pi))
    # Set the learning rate for the optimizer
    optimizer.param_groups[0]['lr'] = lr
    print("learning rate:",lr)
    # training
    train_acc, train_obj, tar, pre = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr)

    # validation
    valid_acc, valid_obj, tar_v, pre_v = infer(valid_queue, model, criterion)

    toc = time.time()

    logging.info('Epoch %03d: train_loss = %f, train_acc = %f, val_loss = %f, val_acc = %f, time = %f', epoch,
                 train_obj, train_acc, valid_obj, valid_acc, toc - tic)

    if valid_obj < min_valid_loss:
        genotype = model.genotype()
        min_valid_loss = valid_obj
        logging.info('genotype = %s', genotype)

  return genotype


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  tar = np.array([])
  pre = np.array([])

  for step, (input, target) in enumerate(train_queue):
    model.train()
    n = input.size(0)
    print(n)
    input = Variable(input, requires_grad=False).cuda()
    target = Variable(target, requires_grad=False).cuda()
    # print(input,target)
    input_search, target_search = next(iter(valid_queue))
    input_search = Variable(input_search, requires_grad=False).cuda()
    target_search = Variable(target_search, requires_grad=False).cuda()

    architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

    optimizer.zero_grad()
    logits = model(input)
    loss = criterion(logits, target)

    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1,t,p= utils.accuracy(logits, target, topk=(1, ))
    objs.update(loss.data, n)
    top1.update(prec1[0].data, n)
    tar = np.append(tar, t.data.cpu().numpy())
    pre = np.append(pre, p.data.cpu().numpy())

  return top1.avg, objs.avg, tar, pre


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  model.eval()
  tar = np.array([])
  pre = np.array([])

  for step, (input, target) in enumerate(valid_queue):
    input = Variable(input, volatile=True).cuda()
    target = Variable(target, volatile=True).cuda()

    logits = model(input)
    loss = criterion(logits, target)

    prec1, t, p = utils.accuracy(logits, target, topk=(1, ))
    n = input.size(0)
    objs.update(loss.data, n)
    top1.update(prec1[0].data, n)
    tar = np.append(tar, t.data.cpu().numpy())
    pre = np.append(pre, p.data.cpu().numpy())

  return top1.avg, objs.avg, tar, pre


if __name__ == '__main__':
  genotype = main(seed=np.random.randint(low=0, high=10000, size=1), cut=True)
  print('Searched Neural Architecture:')
  print(genotype)
