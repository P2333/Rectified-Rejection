import argparse
import logging
import sys
import time
import math
from torchvision import datasets, transforms
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import Dataset
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score, f1_score, roc_curve
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os

from models import *
from utils import *
from PIL import Image


def get_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--model', default='PreActResNet18')
    parser.add_argument('--model_name', type=str, default='PreActResNet18')
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--dataset', default='CIFAR-10', type=str)
    parser.add_argument('--data-dir', default='../cifar-data', type=str)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr-max', default=0.1, type=float)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--attack-iters', default=10, type=int)
    parser.add_argument('--restarts', default=1, type=int)
    parser.add_argument('--pgd-alpha', default=2, type=float)
    parser.add_argument('--fgsm-alpha', default=1.25, type=float)
    parser.add_argument('--norm', default='l_inf', type=str, choices=['l_inf', 'l_2'])
    parser.add_argument('--fgsm-init', default='random', choices=['zero', 'random', 'previous'])
    parser.add_argument('--fname', default='cifar_model', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default=0, type=int)

    parser.add_argument('--load_epoch', default=101, type=int)
    parser.add_argument('--evalset', default='test', choices=['test'])
    parser.add_argument('--target', action='store_true') # whether use target-mode attack

    parser.add_argument('--ConfidenceOnly', action='store_true')
    parser.add_argument('--AuxiliaryOnly', action='store_true')

    # two branch
    parser.add_argument('--twobranch', action='store_true')
    parser.add_argument('--out_dim', default=10, type=int)
    parser.add_argument('--useBN', action='store_true')
    parser.add_argument('--along', action='store_true')
    parser.add_argument('--selfreweightCalibrate', action='store_true') # Calibrate
    parser.add_argument('--selfreweightSelectiveNet', action='store_true')
    parser.add_argument('--selfreweightATRO', action='store_true')
    parser.add_argument('--selfreweightCARL', action='store_true')
    parser.add_argument('--lossversion', default='onehot', choices=['onehot', 'category'])
    parser.add_argument('--tempC', default=1., type=float)
    parser.add_argument('--evalonAA', action='store_true')# evaluate on AutoAttack
    parser.add_argument('--evalonCWloss', action='store_true')# evaluate on PGD with CW loss
    parser.add_argument('--evalonGAMA_FW', action='store_true')# evaluate on GAMA-FW
    parser.add_argument('--evalonGAMA_PGD', action='store_true')# evaluate on GAMA-FW
    parser.add_argument('--evalonMultitarget', action='store_true')# evaluate on GAMA-FW
    return parser.parse_args()


# corruptes = ['brightness', 'elastic_transform', 'gaussian_blur', 'impulse_noise',
#         'motion_blur', 'shot_noise', 'speckle_noise', 'contrast', 'fog', 'gaussian_noise',
#         'jpeg_compression', 'pixelate', 'snow', 'zoom_blur', 'defocus_blur', 'frost', 'glass_blur',
#         'saturate', 'spatter']

corruptes = ['glass_blur', 'motion_blur', 'zoom_blur',
'snow', 'frost', 'fog',
'brightness', 'contrast', 'elastic_transform', 'jpeg_compression']

kwargs = {'num_workers': 4, 'pin_memory': True}

class CIFAR10_C(Dataset):
    def __init__(self, root, name, transform=None, target_transform=None):
        self.data = []
        self.targets = []
        self.transform = transform
        self.target_transform = target_transform
        assert name in corruptes
        file_path = os.path.join(root, 'CIFAR10-C', name+'.npy')
        lable_path = os.path.join(root, 'CIFAR10-C', 'labels.npy')
        self.data = np.load(file_path)
        self.targets = np.load(lable_path)
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    def __len__(self):
        return len(self.data)

def eval_adv_test_general(args, model, name, logger):
    """
    evaluate model by white-box attack
    """
    # set up data loader
    transform_test = transforms.Compose([transforms.ToTensor(),])
    testset = CIFAR10_C(root='../cifar-data', name = name, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, **kwargs)

    model.eval()
    test_acc, test_n = 0, 0
    test_classes_correct, test_classes_wrong = [], []

    # record con
    test_con_correct = []
    test_con_wrong = []

    # record evi
    test_evi_correct = []
    test_evi_wrong = []

    for idx, (data, target) in enumerate(test_loader):
        X, y = data.cuda(), target.long().cuda()

        if args.twobranch:
            output, output_aux = model(normalize(X))[0:2]
            con_pre, _ = torch.softmax(output * args.tempC, dim=1).max(1) # predicted label and confidence

            if args.selfreweightCalibrate:
                output_aux = output_aux.sigmoid().squeeze()
                test_evi_all = con_pre * output_aux
                if args.ConfidenceOnly:
                    test_evi_all = con_pre
                if args.AuxiliaryOnly:
                    test_evi_all = output_aux

            elif args.selfreweightSelectiveNet:
                test_evi_all = output_aux.sigmoid().squeeze()

            elif args.selfreweightATRO:
                test_evi_all = output_aux.tanh().squeeze()

            elif args.selfreweightCARL:
                output_all = torch.cat((output, output_aux), dim=1) # bs x 11 or bs x 101
                softmax_output = F.softmax(output_all, dim=1)
                test_evi_all = softmax_output[torch.tensor(range(X.size(0))), -1]
            
        else:
            output = model(normalize(X))
            test_evi_all = output.logsumexp(dim=1)

        output_s = F.softmax(output, dim=1)
        out_con, out_pre = output_s.max(1)

        # output labels
        labels = torch.where(out_pre == y)[0]
        labels_n = torch.where(out_pre != y)[0]

        # ground labels
        test_classes_correct += y[labels].tolist()
        test_classes_wrong += y[labels_n].tolist()
        
        # accuracy
        test_acc += labels.size(0)

        # confidence
        test_con_correct += out_con[labels].tolist()
        test_con_wrong += out_con[labels_n].tolist()
                 
        # evidence
        test_evi_correct += test_evi_all[labels].tolist()
        test_evi_wrong += test_evi_all[labels_n].tolist()

        test_n += y.size(0)
        
    # confidence
    test_con_correct = torch.tensor(test_con_correct)
    test_con_wrong = torch.tensor(test_con_wrong)

    # evidence
    test_evi_correct = torch.tensor(test_evi_correct)
    test_evi_wrong = torch.tensor(test_evi_wrong)

    test_acc = test_acc/test_n

    print('### Basic statistics ###')
    logger.info('Clean       | acc: %.4f | con cor: %.3f (%.3f) | con wro: %.3f (%.3f) | evi cor: %.3f (%.3f) | evi wro: %.3f (%.3f)', 
            test_acc, 
            test_con_correct.mean().item(), test_con_correct.std().item(),
            test_con_wrong.mean().item(), test_con_wrong.std().item(),
            test_evi_correct.mean().item(), test_evi_correct.std().item(),
            test_evi_wrong.mean().item(), test_evi_wrong.std().item())


    print('')
    print('### ROC-AUC scores (confidence) ###')
    clean_clean = calculate_auc_scores(test_con_correct, test_con_wrong)
    _, acc95 = calculate_FPR_TPR(test_con_correct, test_con_wrong, tpr_ref=0.95)
    _, acc99 = calculate_FPR_TPR(test_con_correct, test_con_wrong, tpr_ref=0.99)
    logger.info('clean_clean: %.3f', 
            clean_clean)
    logger.info('TPR 95 clean acc: %.4f; 99 clean acc: %.4f', 
            acc95, acc99)
        

    print('')
    print('### ROC-AUC scores (evidence) ###')
    clean_clean = calculate_auc_scores(test_evi_correct, test_evi_wrong)
    _, acc95 = calculate_FPR_TPR(test_evi_correct, test_evi_wrong, tpr_ref=0.95)
    _, acc99 = calculate_FPR_TPR(test_evi_correct, test_evi_wrong, tpr_ref=0.99)
    logger.info('clean_clean: %.3f', 
            clean_clean)
    logger.info('TPR 95 clean acc: %.4f; 99 clean acc: %.4f', 
            acc95, acc99)
            

def main():
    args = get_args()

    # define a logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(args.fname, 'eval.log')),
            logging.StreamHandler()
        ])
    logger.info(args)

    # set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    num_cla = 10

    if args.selfreweightCalibrate or args.selfreweightSelectiveNet or args.selfreweightCARL or args.selfreweightATRO:
        along = True
        args.out_dim = 1

    # load pretrained model
    if args.model_name == 'PreActResNet18':
        model = PreActResNet18(num_classes=num_cla)
    elif args.model_name == 'PreActResNet18_twobranch_DenseV1':
        model = PreActResNet18_twobranch_DenseV1(num_classes=num_cla, out_dim=args.out_dim, use_BN=args.useBN, along=along)
    elif args.model_name == 'WideResNet':
        model = WideResNet(34, num_cla, widen_factor=10, dropRate=0.0)
    elif args.model_name == 'WideResNet_twobranch_DenseV1':
        model = WideResNet_twobranch_DenseV1(34, num_cla, widen_factor=10, dropRate=0.0, along=along, use_BN=args.useBN, out_dim=args.out_dim)
    elif args.model_name == 'PreActResNet18_threebranch_DenseV1':
        model = PreActResNet18_threebranch_DenseV1(num_classes=num_cla, out_dim=args.out_dim, use_BN=args.useBN, along=along)
    elif args.model_name == 'WideResNet_threebranch_DenseV1':
        model = WideResNet_threebranch_DenseV1(34, num_cla, widen_factor=10, dropRate=0.0, use_BN=args.useBN, along=along, out_dim=args.out_dim)
    else:
        raise ValueError("Unknown model")

    model = nn.DataParallel(model).cuda()
    if args.load_epoch > 0:       
        model_dict = torch.load(os.path.join(args.fname, f'model_{args.load_epoch}.pth'))
        logger.info(f'Resuming at epoch {args.load_epoch}')
    else:
        model_dict = torch.load(os.path.join(args.fname, f'model_best.pth'))
        logger.info(f'Resuming at best epoch')

    if 'state_dict' in model_dict.keys():
        model.load_state_dict(model_dict['state_dict'])
    else:
        model.load_state_dict(model_dict)


    for name in corruptes:
        print('')
        print('')
        print('====== test ' + name + ' =====')
        eval_adv_test_general(args, model, name, logger)
    
if __name__ == "__main__":
    main()
