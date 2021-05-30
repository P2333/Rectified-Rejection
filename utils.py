import numpy as np
from collections import namedtuple
import torch
import torch.nn as nn
import torchvision
from sklearn.metrics import roc_auc_score, f1_score, roc_curve


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

################################################################
## Components from https://github.com/davidcpage/cifar10-fast ##
################################################################

#####################
## data preprocessing
#####################

cifar10_mean = (0.4914, 0.4822, 0.4465) # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (0.2471, 0.2435, 0.2616) # equals np.std(train_set.train_data, axis=(0,1,2))/255

svhn_mean = (0.5, 0.5, 0.5)
svhn_std = (0.5, 0.5, 0.5)

mu_svhn = torch.tensor(svhn_mean).view(3,1,1).cuda()
std_svhn = torch.tensor(svhn_std).view(3,1,1).cuda()

mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
std = torch.tensor(cifar10_std).view(3,1,1).cuda()

def CW_loss(x, y, SUM=False):
    x_sorted, ind_sorted = x.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()
    
    loss_value = -(x[np.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind))
    return loss_value.mean() if SUM else loss_value.sum()


def calculate_auc_scores(correct, wrong):
    labels_all = torch.cat((torch.ones_like(correct), torch.zeros_like(wrong)), dim=0).cpu().numpy()
    prediction_all = torch.cat((correct, wrong), dim=0).cpu().numpy()
    return roc_auc_score(labels_all, prediction_all)

def calculate_FPR_TPR(correct, wrong, tpr_ref=0.95):
    labels_all = torch.cat((torch.ones_like(correct), torch.zeros_like(wrong)), dim=0).cpu().numpy()
    prediction_all = torch.cat((correct, wrong), dim=0).cpu().numpy()
    fpr, tpr, thresholds = roc_curve(labels_all, prediction_all)
    index = np.argmin(np.abs(tpr - tpr_ref))
    T = thresholds[index]
    FPR_thred = fpr[index]
    index_c = (torch.where(correct > T)[0]).size(0)
    index_w = (torch.where(wrong > T)[0]).size(0)
    acc = index_c / (index_c + index_w)
    return FPR_thred, acc

def normalize(X):
    return (X - mu)/std

def normalize_svhn(X):
    return (X - mu_svhn)/std_svhn

upper_limit, lower_limit = 1,0

def sample_targetlabel(y, num_classes=10):
    y_target = torch.randint_like(y, 0, 10).cuda()
    index = torch.where(y_target == y)[0]
    while index.size(0)!= 0:
        y_target_new = torch.randint(0, 10, (index.size(0),)).cuda()
        y_target[index] = y_target_new
        index = torch.where(y_target == y)[0]
    return y_target

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def normalise(x, mean=cifar10_mean, std=cifar10_std):
    x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
    x -= mean*255
    x *= 1.0/(255*std)
    return x

def pad(x, border=4):
    return np.pad(x, [(0, 0), (border, border), (border, border), (0, 0)], mode='reflect')

def transpose(x, source='NHWC', target='NCHW'):
    return x.transpose([source.index(d) for d in target]) 

#####################
## data augmentation
#####################

class Crop(namedtuple('Crop', ('h', 'w'))):
    def __call__(self, x, x0, y0):
        return x[:,y0:y0+self.h,x0:x0+self.w]

    def options(self, x_shape):
        C, H, W = x_shape
        return {'x0': range(W+1-self.w), 'y0': range(H+1-self.h)}
    
    def output_shape(self, x_shape):
        C, H, W = x_shape
        return (C, self.h, self.w)
    
class FlipLR(namedtuple('FlipLR', ())):
    def __call__(self, x, choice):
        return x[:, :, ::-1].copy() if choice else x 
        
    def options(self, x_shape):
        return {'choice': [True, False]}

class Cutout(namedtuple('Cutout', ('h', 'w'))):
    def __call__(self, x, x0, y0):
        x = x.copy()
        x[:,y0:y0+self.h,x0:x0+self.w].fill(0.0)
        return x

    def options(self, x_shape):
        C, H, W = x_shape
        return {'x0': range(W+1-self.w), 'y0': range(H+1-self.h)} 
    
    
class Transform():
    def __init__(self, dataset, transforms):
        self.dataset, self.transforms = dataset, transforms
        self.choices = None
        
    def __len__(self):
        return len(self.dataset)
           
    def __getitem__(self, index):
        data, labels = self.dataset[index]
        for choices, f in zip(self.choices, self.transforms):
            args = {k: v[index] for (k,v) in choices.items()}
            data = f(data, **args)
        return data, labels
    
    def set_random_choices(self):
        self.choices = []
        x_shape = self.dataset[0][0].shape
        N = len(self)
        for t in self.transforms:
            options = t.options(x_shape)
            x_shape = t.output_shape(x_shape) if hasattr(t, 'output_shape') else x_shape
            self.choices.append({k:np.random.choice(v, size=N) for (k,v) in options.items()})

#####################
## dataset
#####################

def cifar10(root):
    train_set = torchvision.datasets.CIFAR10(root=root, train=True, download=True)
    test_set = torchvision.datasets.CIFAR10(root=root, train=False, download=True)
    return {
        'train': {'data': train_set.data, 'labels': train_set.targets},
        'test': {'data': test_set.data, 'labels': test_set.targets}
    }

def cifar100(root):
    train_set = torchvision.datasets.CIFAR100(root=root, train=True, download=True)
    test_set = torchvision.datasets.CIFAR100(root=root, train=False, download=True)
    return {
        'train': {'data': train_set.data, 'labels': train_set.targets},
        'test': {'data': test_set.data, 'labels': test_set.targets}
    }

#####################
## data loading
#####################

class Batches():
    def __init__(self, dataset, batch_size, shuffle, set_random_choices=False, num_workers=0, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.set_random_choices = set_random_choices
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=shuffle, drop_last=drop_last
        )
    
    def __iter__(self):
        if self.set_random_choices:
            self.dataset.set_random_choices() 
        return ({'input': x.to(device).half(), 'target': y.to(device).long()} for (x,y) in self.dataloader)
    
    def __len__(self): 
        return len(self.dataloader)

#####################
## Auto file name
#####################
def get_auto_fname(args):
    if args.ATframework == 'PGDAT':
        names = 'PGDAT_' + args.model_name
    elif args.ATframework == 'TRADES':
        names = 'TRADES' + str(args.TRADESlambda) + '_' + args.model_name
    elif args.ATframework == 'CCAT':
        names = 'CCAT_ratio' + str(args.CCATratio) + 'iter' + str(args.CCATiter) + 'rho' + str(args.CCATrho) + 'step' + str(args.CCATstep) + '_' + args.model_name

    if args.useBN:
        names += 'BN'
    if args.adaptiveattack:
        names += '_adaptiveA' + str(args.adaptiveattacklambda)
        if args.uniform_lambda:
            names += 'uniform'
        if args.twosign:
            names += 'twosign'
    if args.BNeval:
        names += '_BNeval'

    if args.adaptivetrain:
        names += '_adaptiveT' + str(args.adaptivetrainlambda)
        if args.selfreweightCalibrate:
            names += '_selfreweightCalibrate'
            if args.tempC != 1.:
                names += '_tempC' + str(args.tempC)
            if args.tempC_trueonly != 1.:
                names += '_tempCtrueonly' + str(args.tempC_trueonly)
            if args.SGconfidenceW:
                names += '_SGconW'
            if args.ConfidenceOnly:
                names += '_ConfidenceOnly'
            if args.AuxiliaryOnly:
                names += '_AuxiliaryOnly'
            
        elif args.selfreweightNIPS20:
            names += '_selfreweightNIPS20' + 'mi' + str(args.m_in) + 'mo' + str(args.m_out)

        elif args.selfreweightATRO:
            names += '_selfreweightATRO' + 'cost' + str(args.ATRO_cost) + 'coe' + str(args.ATRO_coefficient)

        elif args.selfreweightCARL:
            names += '_selfreweightCARL' + 'lambda' + str(args.CARL_lambda) + 'eta' + str(args.CARL_eta)
            #names += '_selfreweightCARL' + 'lambda' + str(args.CARL_lambda)

    if args.weight_decay != 5e-4:
        names = names + '_wd' + str(args.weight_decay)
    if args.epochs != 110:
        names += '_epochs' + str(args.epochs)
    if args.batch_size != 128:
        names += '_bs' + str(args.batch_size)
    if args.epsilon != 8:
        names += '_eps' + str(args.epsilon)
    if args.CCATscale != 1:
        names += '_scale' + str(args.CCATscale)
    names += '_seed' + str(args.seed)
    print('File name: ', names)
    return names