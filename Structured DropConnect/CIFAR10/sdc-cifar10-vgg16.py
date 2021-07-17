'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function
import time
import numpy as np
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import os
import argparse
import logging
from uncertainty_measurements import *

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description='None')
parser.add_argument("--savepath",    default='.', type=str)
parser.add_argument("--repeattimes", default='100epoch_cifar10_sdc_batch256_test(1)', type=str)
#parser.add_argument("--card",        default="0", type=str)
args = parser.parse_args()

save_path = args.savepath + '/' + args.repeattimes
if not os.path.exists(save_path):
    os.mkdir(save_path)

#os.environ["CUDA_VISIBLE_DEVICES"] = args.card
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
BATCH_SIZE = 256
nb_epoch = 100
lr = 7.5e-4
cycle_len = 70
rho = 0.1   #dropout rate
loop = 1/rho
#loop = 0

print("OPENING " + save_path + '/results_train.csv')
print("OPENING " + save_path + '/results_test.csv')

results_train_file = open(save_path + '/results_train.csv', 'w')
results_train_file.write('epoch,train_acc,train_loss\n')
results_train_file.flush()

results_test_file = open(save_path + '/results_test.csv', 'w')
results_test_file.write('epoch,test_acc\n')
results_test_file.flush()

use_cuda = torch.cuda.is_available()

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
])

trainset = torchvision.datasets.CIFAR10(root='/data2/zhengwenqing/data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(root='/data2/zhengwenqing/data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

#for uncertainty inference
uiset = torchvision.datasets.CIFAR10(root='/data2/zhengwenqing/data', train=False, download=True, transform=transform_test)
uiloader = torch.utils.data.DataLoader(uiset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# Model

net = models.vgg16_bn(pretrained=False)
net = list(net.features)
for i in range(len(net)):
    if isinstance(net[i], nn.ReLU):
        net[i] = nn.LeakyReLU(0.2, inplace=True)
net = nn.Sequential(*net)

class model_bn(nn.Module):
    def __init__(self, model, feature_size, classes_num):
        super(model_bn, self).__init__() 
        self.features_1 = net
        self.num_ftrs = 512*1*1

        self.batch1 = nn.BatchNorm1d(self.num_ftrs)
        self.fc1 = DropConnectLinear(self.num_ftrs, feature_size, rho, loop, mask1)

        self.batch2 = nn.BatchNorm1d(feature_size)
        self.LeakyReLU = nn.LeakyReLU(0.2, inplace=True)
        self.fc2 = DropConnectLinear(feature_size, feature_size, rho, loop, mask2)

        self.batch3 = nn.BatchNorm1d(feature_size)
        self.LeakyReLU = nn.LeakyReLU(0.2, inplace=True)
        self.fc3 = DropConnectLinear(feature_size, classes_num, rho, loop, mask3)


    def forward(self, x, index):
        out = self.features_1(x)
        out = out.view(out.size(0), -1)

        out = self.batch1(out)
        out = self.fc1(out, index)

        out = self.batch2(out)
        out = self.LeakyReLU(out)
        out = self.fc2(out,index)

        out = self.batch3(out)
        out = self.LeakyReLU(out)
        out = self.fc3(out,index)

        return out



#DropConnectLinear
class DropConnectLinear(nn.Module):
    def __init__(self, in_features, out_features, rho, loop, mask, bias=True):
        super(DropConnectLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rho = rho
        self.loop = loop
        self.mask = mask
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input, index):
        if self.training:
            m = torch.ones(self.out_features, self.in_features) * (1 - self.rho) 
            mask_dc = torch.bernoulli(m).cuda()
            out = F.linear(input, self.weight.mul(mask_dc), self.bias)
        else:
            if(loop == 0):      #normal test
                out = F.linear(input, self.weight, self.bias)
            else:    #test with SDC
                out = F.linear(input, self.weight.mul(self.mask[index]), self.bias)
        return out


def mask(out_features, in_features):
    mask = []

    idx = torch.arange(out_features * in_features)
    idx0 = [i for i in range(len(idx))]
    random.shuffle(idx0)
    n = out_features * in_features
    for loop1 in range(int(1/rho)):
        m = torch.ones(out_features , in_features)
        m = m.view(-1)
        for loop2 in idx[idx0][int(loop1 * rho * n) : int((loop1 + 1) * rho * n)]:
            m[loop2] = 0
        mask.append(m)
    mask = torch.cat(mask, dim = 0).reshape(int(1 / rho), out_features, in_features)
    return mask

mask1 = mask(2048, 512).cuda()
mask2 = mask(2048, 2048).cuda()
mask3 = mask(10, 2048).cuda()



classes_num = 10
net = model_bn(net, 2048, classes_num)

device = torch.device("cuda")

net = net.to(device)
net.features_1.to(device)
net.features_1 = torch.nn.DataParallel(net.features_1, device_ids = [0])

#cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=5e-4)

# Training
alpha = 1.
beta = 1.
gamma = 1.
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    idx = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        idx = batch_idx
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs,0)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.detach().cpu()
        _, predicted = torch.max(outputs, -1)
        correct += predicted.eq(targets.data).sum().item()
        total += targets.size(0)
        
    train_acc = 100. * correct / total
    train_loss = train_loss / (idx + 1)
    logging.info('Iteration %d, train_acc_cls = %.4f, train_loss = %.4f' % (epoch, train_acc, train_loss))
    results_train_file.write('%d,%.4f,%.4f\n' % (epoch, train_acc, train_loss))
    results_train_file.flush()

    return train_acc, train_loss

def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    idx = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        idx = batch_idx
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs,0)

        loss = criterion(outputs, targets)

        test_loss += loss.detach().cpu()
        total += targets.size(0)
        _, predicted = torch.max(outputs, -1)
        correct += predicted.eq(targets.data).sum().item()

    test_acc = 100. * correct / total
    test_loss = test_loss / (idx + 1)
    logging.info('test, test_acc_cls = %.4f, test_loss = %.4f' % (test_acc, test_loss))
    results_test_file.write('%d,%.4f,%.4f\n' % (epoch, test_acc, test_loss))
    results_test_file.flush()

    return test_acc, test_loss


def sdc_uncetainty():
    net.eval()
    probs_list = []
    alphas_list = []
    targets_list = []
    for batch_idx, (inputs, targets) in enumerate(uiloader):
        inputs = inputs.to(device)
        structured_outputs = []
        for index in range(int(1/rho)):
            outputs = net(inputs, index)
            structured_outputs.append(outputs)
        structured_outputs = torch.cat(structured_outputs, dim = 0).reshape(int(1 / rho), -1, classes_num)    #[10,256,10]
        structured_outputs = F.softmax(structured_outputs, dim = 2)

        sdc_outputs_mean = torch.mean(structured_outputs, dim = 0)    #[256,10]
        sdc_outputs_var = torch.std(structured_outputs, dim = 0)    #[256,10]
        #print('sdc_outputs_mean:{}'.format(sdc_outputs_mean))
        #print('sdc_outputs_var:{}'.format(sdc_outputs_var))

        outputs_softmax = F.softmax(sdc_outputs_mean, dim=1)
        #print('outputs_softmax:{}'.format(outputs_softmax))
        alpha0 = sdc_outputs_mean.mul(1 - sdc_outputs_mean) / sdc_outputs_var - 1
        alpha = alpha0 * sdc_outputs_mean

        probs_list += list(outputs_softmax.data.cpu().numpy())
        targets_list += list(targets.data.numpy())
        alphas_list += list(alpha.data.cpu().numpy())

    probs = np.array(probs_list)
    alphas = np.array(alphas_list)
    targets = np.array(targets_list)
    auc_max_prob, auc_ent, auc_mi, auc_diff_ent, aupr_max_prob, aupr_ent, aupr_mi,aupr_diff_ent = dpn_auc(probs, alphas, targets)
    print('AUROC of Max.P/Ent.: %.4f %.4f' % (auc_max_prob, auc_ent))
    print('AUROC of MI/Diff_Ent.: %.4f %.4f' % (auc_mi, auc_diff_ent))
    print('AUPR  of Max.P/Ent.: %.4f %.4f' % (aupr_max_prob, aupr_ent))
    print('AUPR  of MI/Diff_Ent.: %.4f %.4f' % (aupr_mi, aupr_diff_ent))
    print('###################\n\n')



def cosine_anneal_schedule(t):
    cos_inner = np.pi * (t % (nb_epoch))  # t - 1 is used when t has 1-based indexing.
    cos_inner /= (nb_epoch)
    cos_out = np.cos(cos_inner) + 1
    return float( 0.1 / 2 * cos_out)

def dnn_uncetainty():
    net.eval()
    probs_list = []
    targets_list = []
    for batch_idx, (inputs, targets) in enumerate(uiloader):
        inputs = inputs.to(device)
        outputs = net(inputs,0)
        outputs = F.softmax(outputs, dim=1)

        probs_list += list(outputs.data.cpu().numpy())
        targets_list += list(targets.data.numpy())

    probs = np.array(probs_list)
    targets = np.array(targets_list)
    auc_max_prob, auc_ent, aupr_max_prob, aupr_ent = dnn_auc(probs, targets)
    print('AUROC of Max.P/Ent.: %.4f %.4f' % (auc_max_prob, auc_ent))
    print('AUPR  of Max.P/Ent.: %.4f %.4f' % (aupr_max_prob, aupr_ent))
    print('###################\n\n')

def mcdropout(t_mc=100):
    net.train()
    probs_list = []
    targets_list = []
    for batch_idx, (inputs, targets) in enumerate(uiloader):
        inputs = inputs.to(device)
        outputs = torch.zeros(inputs.size(0), t_mc, 10)
        for i in range(t_mc):
            outputs[:, i, :] = net(inputs,0).data.cpu()
        outputs = F.softmax(outputs, dim=2)

        probs_list += list(outputs.data.numpy())
        targets_list += list(targets.data.numpy())

    probs = np.array(probs_list)
    targets = np.array(targets_list)
    auc_max_prob, auc_ent, auc_mi, aupr_max_prob, aupr_ent, aupr_mi = mcdp_auc(probs, targets)
    print('AUROC of Max.P/Ent./M.I.: %.4f %.4f %.4f' % (auc_max_prob, auc_ent, auc_mi))
    print('AUPR  of Max.P/Ent./M.I.: %.4f %.4f %.4f' % (aupr_max_prob, aupr_ent, aupr_mi))
    print('###################\n\n')

def cycle_learning_rate(init_lr, cycle_len, n_epoch):
    lr = []
    for i in range(cycle_len // 2):
        lr += [init_lr / 10. + (init_lr - init_lr / 10.) / (cycle_len / 2) * i]
    for i in range(cycle_len - cycle_len // 2):
        lr += [init_lr / 10. + (init_lr - init_lr / 10.) / (cycle_len / 2) * (cycle_len / 2 - i)]
    for i in range(n_epoch - cycle_len):
        lr += [init_lr / 10. - (init_lr / 10. - 1e-6) / (n_epoch - cycle_len) * i]
    return lr
    
lr = cycle_learning_rate(lr, cycle_len, nb_epoch)

if os.path.exists(save_path + '/checkpoint.pth'):
    net = torch.load(save_path + '/checkpoint.pth')
else:
    max_val_acc = 0
    for epoch in range(nb_epoch):
        # if epoch == 150 or epoch == 225:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = param_group['lr'] / 10

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr[epoch]

        for param_group in optimizer.param_groups:
            print(param_group['lr'])

        train(epoch)
        acc = test(epoch)
        val_acc = acc[0]
        torch.save(net, save_path + '/checkpoint.pth')
        if val_acc > max_val_acc:
            max_val_acc = val_acc
            torch.save(net, save_path + '/model_best.pth')
        print("the max_val_acc === ", max_val_acc)


'''
if(loop != 0):
    print('SDC Uncertainty')
    sdc_uncetainty()

print('\nCheckpoint:')
test(100)
print('DNN Uncertainty')
dnn_uncetainty()

print('MC Dropout Uncertainty')
t_mc = 100
print('Times: %d' % t_mc)
mcdropout(t_mc=t_mc)
t_mc = 10
print('Times: %d' % t_mc)
mcdropout(t_mc=t_mc)

print('\nBest model:')
net = torch.load(save_path + '/model_best.pth')
test(100)
print('DNN Uncertainty')
dnn_uncetainty()

print('MC Dropout Uncertainty')
t_mc = 100
print('Times: %d' % t_mc)
mcdropout(t_mc=t_mc)
t_mc = 10
print('Times: %d' % t_mc)
mcdropout(t_mc=t_mc)
'''
