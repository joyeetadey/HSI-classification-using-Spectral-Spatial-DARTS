import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import random
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from torch.autograd import Variable
from model import NetworkHSI as Network
from sklearn.metrics import confusion_matrix
from utils import cutout
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score


parser = argparse.ArgumentParser("HSI")
parser.add_argument('--num_class', type=int, default=13, help='classes of HSI dataset')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.016, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=150, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=3, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=2, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--arch', type=str, default='HSI', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--num_cut', type=int, default=10, help='band cutout')
parser.add_argument('--Train', type=int, default=500, help='Train_num')
parser.add_argument('--Valid', type=int, default=300, help='Valid_num')
args = parser.parse_args()
args.cuda = torch.cuda.is_available()
args.manualSeed = random.randint(1, 10000)


log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler('./result/log.txt')
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

# read data
image_file = '/content/KSC.mat'
label_file = '/content/KSC_gt.mat'

image = sio.loadmat(image_file)
KSC = image['KSC']

label = sio.loadmat(label_file)
GroundTruth = label['KSC_gt']

KSC = (KSC - np.min(KSC)) / (np.max(KSC) - np.min(KSC))

[nRow, nColumn, nBand] = KSC.shape

num_class = int(np.max(GroundTruth))

HalfWidth = 16
Wid = 2 * HalfWidth
[row, col] = GroundTruth.shape


NotZeroMask = np.zeros([row, col])
Wid = 2 * HalfWidth
NotZeroMask[HalfWidth + 1: -1 - HalfWidth + 1, HalfWidth + 1: -1 - HalfWidth + 1] = 1
G = GroundTruth * NotZeroMask

[Row, Column] = np.nonzero(G)
nSample = np.size(Row)

nTrain = args.Train
nValidate = args.Valid
total = nTrain+nValidate
nTest = (nSample-total)
batchtr = nTrain
numbatch1 = nTrain // batchtr

batchva = 1000
numbatch2 = nTest // batchva

HSI_CLASSES = num_class


def main(genotype, seed, cut):

  np.random.seed(seed)
  RandPerm = np.random.permutation(nSample)

  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)
  args.cutout = cut

  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.manualSeed)
  cudnn.enabled = True
  torch.cuda.manual_seed(args.manualSeed)

  model = Network(nBand, args.init_channels, HSI_CLASSES, args.layers, args.auxiliary, genotype)
  model = model.cuda()

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()

  optimizer = torch.optim.Adam(model.parameters(), args.learning_rate, weight_decay=args.weight_decay)
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.epochs // 5, 0.5)

  min_val_obj = 100
  for epoch in range(1, args.epochs+1):
    scheduler.step()

    model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

    predict = np.array([], dtype=np.int64)
    labels = np.array([], dtype=np.int64)

    imdb = {}
    imdb['data'] = np.zeros([2 * HalfWidth, 2 * HalfWidth, nBand, nTrain+nValidate], dtype=np.float32)
    imdb['Labels'] = np.zeros([nTrain+nValidate], dtype=np.int64)
    imdb['set'] = np.hstack((np.ones([nTrain]), 3 * np.ones([nValidate]))).astype(np.int64)

    for iSample in range(nTrain):

        yy = KSC[Row[RandPerm[iSample]] - HalfWidth: Row[RandPerm[iSample]] + HalfWidth, \
                  Column[RandPerm[iSample]] - HalfWidth: Column[RandPerm[iSample]] + HalfWidth, :]
        if args.cutout:
            xx = cutout(yy, args.cutout_length, args.num_cut)

            imdb['data'][:, :, :, iSample] = xx
        else:
            imdb['data'][:, :, :, iSample] = yy

        imdb['Labels'][iSample] = G[Row[RandPerm[iSample]], Column[RandPerm[iSample]]].astype(np.int64)

    for iSample in range(nValidate):
        imdb['data'][:, :, :, iSample + nTrain] = KSC[Row[RandPerm[iSample + nTrain]] - HalfWidth: Row[RandPerm[
            iSample + nTrain]] + HalfWidth, \
                                                  Column[RandPerm[iSample + nTrain]] - HalfWidth: Column[RandPerm[
                                                      iSample + nTrain]] + HalfWidth, :]
        imdb['Labels'][iSample + nTrain] = G[Row[RandPerm[iSample + nTrain]],
                                             Column[RandPerm[iSample + nTrain]]].astype(np.int64)
    imdb['Labels'] = imdb['Labels'] - 1

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
    train_queue = torch.utils.data.DataLoader(dataset=trainset, batch_size=128, shuffle=True, num_workers=0)
    valid_queue  = torch.utils.data.DataLoader(dataset=testset,  batch_size=128, shuffle=True, num_workers=0)

    tic = time.time()
    train_acc, train_obj, tar, pre = train(train_queue, model, criterion, optimizer)

    # validation
    valid_acc, valid_obj, tar_v, pre_v = infer(valid_queue, model, criterion)
    toc = time.time()
    logging.info('Epoch: %03d train_loss: %f train_acc: %f valid_loss: %f valid_acc: %f time = %f' %(epoch, train_obj, train_acc, valid_obj, valid_acc,toc - tic))

    if epoch > args.epochs * 0.8 and valid_obj < min_val_obj:
        min_val_obj = valid_obj
        utils.save(model, './result/weights.pt')

    # test
    if epoch == args.epochs:
        utils.load(model, './result/weights.pt')
        for i in range(numbatch2):
            imdb = {}
            imdb['data'] = np.zeros([2 * HalfWidth, 2 * HalfWidth, nBand, batchva], dtype=np.float32)
            imdb['Labels'] = np.zeros([batchva], dtype=np.int64)
            imdb['set'] = 3* np.ones([batchva], dtype=np.int64)

            for iSample in range(batchva):
                imdb['data'][:, :, :, iSample] = KSC[Row[RandPerm[iSample + total + i * batchva]] - HalfWidth: Row[RandPerm[
                    iSample +total+ i * batchva]] + HalfWidth, \
                                                 Column[RandPerm[iSample + total + i * batchva]] - HalfWidth: Column[RandPerm[
                                                     iSample +total+ i * batchva]] + HalfWidth, :]

                imdb['Labels'][iSample] = G[Row[RandPerm[iSample + total+ i * batchva]], Column[RandPerm[iSample + total+ i * batchva]]].astype(np.int64)

            imdb['Labels'] = imdb['Labels'] - 1

            test_dataset = TestDS()

            test_queue = torch.utils.data.DataLoader(test_dataset, batch_size=50,
                                                     shuffle=False, num_workers=0)

            valid_acc, valid_obj, tar_v, pre_v = infer(test_queue, model, criterion)

            predict = np.append(predict, pre_v)
            labels = np.append(labels, tar_v)

        OA_V = sum(map(lambda x, y: 1 if x == y else 0, predict, labels)) / (numbatch2*batchva)
        C1 = confusion_matrix(labels, predict)


        # # Create a figure and axis
        # fig, ax = plt.subplots(figsize=(8, 6))

        # # Create a heatmap using seaborn
        # heatmap = sns.heatmap(C1, annot=True, cmap='Blues', fmt=".2f", xticklabels=class_labels, yticklabels=class_labels, ax=ax)

        # # Set axis labels and title
        # ax.set_xlabel('Predicted')
        # ax.set_ylabel('True')
        # ax.set_title('Classification Heatmap')

        # plt.savefig('/content/result/heatmap_NasNET.png', dpi=300)

        logging.info('test_loss= %f'%(OA_V))
        logging.info('test_accuracy= %f'%(valid_acc))
        print("=========================================================")
        print('Test loss=',OA_V)
        print('Test accuracy=',valid_acc)
        print("=========================================================")
        # print("Confusion matrix")
        # print(C1)
        # print(summary(model, (200,200,32)))
        print("each class accuracy")
        print(classification_report(predict,labels,digits=4))
        print("=========================================================")
        return C1

def train(train_queue, model, criterion, optimizer):

  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  model.train()
  tar = np.array([])
  pre = np.array([])

  for step, (input, target) in enumerate(train_queue):
    input = Variable(input).cuda()
    target = Variable(target).cuda()

    optimizer.zero_grad()
    logits, logits_aux = model(input)
    loss = criterion(logits, target)
    if args.auxiliary:
      loss_aux = criterion(logits_aux, target)
      loss += args.auxiliary_weight*loss_aux
    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, t, p = utils.accuracy(logits, target, topk=(1,))
    n = input.size(0)
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

    logits, _ = model(input)
    loss = criterion(logits, target)

    prec1, t, p = utils.accuracy(logits, target, topk=(1,))
    n = input.size(0)
    objs.update(loss.data, n)
    top1.update(prec1[0].data, n)
    tar = np.append(tar, t.data.cpu().numpy())
    pre = np.append(pre, p.data.cpu().numpy())

  return top1.avg, objs.avg, tar, pre


def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=np.float)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA

if __name__ == '__main__':

  genotype = eval('genotypes.{}'.format(args.arch))
  matrix = main(genotype=genotype, seed=np.random.randint(low=0, high=10000, size=1), cut=False)

  print("\nSearched Architecture")
  print(genotype)

  OA, AA_mean, Kappa, AA = cal_results(matrix)
  print("=========================================================")
  print("Printing Evaluation Metrics")
  print("=========================================================")
  print("Overall accuracy: ",OA)
  print("Mean accuracy: ",AA_mean)
  print("Kappa: ",Kappa)

