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
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score, f1_score, roc_curve
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import os

from models import *
from utils import *

from adaptiveCWattack.carlini_wagner import CarliniWagnerL2Attack, CarliniWagnerLinfAttack
from adaptiveCWattack.carlini_wagner import CarliniWagnerL2Attack_twobranch, CarliniWagnerLinfAttack_twobranch

# def class_model(model):
#     def pick0(*args, **kwargs):
#         return model(*args, **kwargs)[0]
#     return pick0

# def evi_model(model):
#     def pick1(*args, **kwargs):
#         return model(*args, **kwargs)[1]
#     return pick1


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

def calculate_RR_train_median(model, train_batches, args):
    record_train_evi = torch.tensor([]).cuda()
    for i, (X, y) in enumerate(train_batches):
        X, y = X.cuda(), y.cuda()
        if args.twobranch:
            output, output_aux = model(normalize(X))
            output_aux = output_aux.sigmoid().squeeze().detach()
            con_pre, label_pre = torch.softmax(output, dim=1).max(1)
            evi_batch = con_pre.detach() * output_aux  
            indexs = torch.where(label_pre == y)[0]
        record_train_evi = torch.cat((record_train_evi, evi_batch[indexs]), dim=0)
    return record_train_evi

def calculate_Selective_train_median(model, train_batches, args):
    record_train_evi = torch.tensor([]).cuda()
    for i, (X, y) in enumerate(train_batches):
        X, y = X.cuda(), y.cuda()
        if args.twobranch:
            output, output_aux = model(normalize(X))
            output_aux = output_aux.sigmoid().squeeze().detach()
            _, label_pre = torch.softmax(output, dim=1).max(1)
            evi_batch = output_aux  
            indexs = torch.where(label_pre == y)[0]
        record_train_evi = torch.cat((record_train_evi, evi_batch[indexs]), dim=0)
    return record_train_evi

def calculate_NIPS20_train_median(model, train_batches, args):
    record_train_evi = torch.tensor([]).cuda()
    for i, (X, y) in enumerate(train_batches):
        X, y = X.cuda(), y.cuda()
        output = model(normalize(X)).detach()
        indexs = torch.where(output.max(1)[1] == y)[0]
        evi_batch = output.logsumexp(dim=1)
        record_train_evi = torch.cat((record_train_evi, evi_batch[indexs]), dim=0)
    return record_train_evi

def calculate_con_train_median(model, train_batches, args):
    record_train_con = torch.tensor([]).cuda()
    for i, (X, y) in enumerate(train_batches):
        X, y = X.cuda(), y.cuda()
        if args.twobranch:
            con_batch, label_pre = F.softmax(model(normalize(X))[0].detach(), dim=1).max(1)
        else:
            con_batch, label_pre = F.softmax(model(normalize(X)).detach(), dim=1).max(1)
        indexs = torch.where(label_pre == y)[0]
        record_train_con = torch.cat((record_train_con, con_batch[indexs]), dim=0)
    return record_train_con

def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts=1, norm='l_inf', twobranch=False):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    norm_func = normalize
    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda()
        if norm == "l_inf":
            delta.uniform_(-epsilon, epsilon)
        elif norm == "l_2":
            delta.normal_()
            d_flat = delta.view(delta.size(0),-1)
            n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r/n*epsilon
        else:
            raise ValueError
        delta = clamp(delta, lower_limit-X, upper_limit-X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            if twobranch:
                output = model(normalize(X + delta))[0]
            else:
                output = model(normalize(X + delta))
            loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            if norm == "l_inf":
                d = torch.clamp(delta + alpha * torch.sign(grad), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(grad.view(grad.shape[0],-1),dim=1).view(-1,1,1,1)
                scaled_g = grad/(g_norm + 1e-10)
                d = (delta + scaled_g*alpha).view(delta.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(delta)
            d = clamp(d, lower_limit - X, upper_limit - X)
            delta.data = d
            delta.grad.zero_()
        if twobranch:
            all_loss = F.cross_entropy(model(normalize(X+delta))[0], y, reduction='none')
        else:
            all_loss = F.cross_entropy(model(normalize(X+delta)), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta



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
    parser.add_argument('--evalset', default='adaptiveCWtest', choices=['adaptiveCWtest'])
    parser.add_argument('--target', action='store_true') # whether use target-mode attack

    parser.add_argument('--detectmetric', default='RR')

    # hyper-parameters for CW
    parser.add_argument('--binary_search_steps', default=9, type=int)
    parser.add_argument('--CW_iter', default=1000, type=int)
    parser.add_argument('--CW_confidence', default=0, type=float)
    parser.add_argument('--threatmodel', default='l2', choices=['l2', 'linf'])
    parser.add_argument('--reportmodel', default='l2', choices=['l2', 'linf'])

    # two branch
    parser.add_argument('--twobranch', action='store_true')
    parser.add_argument('--out_dim', default=10, type=int)
    parser.add_argument('--useBN', action='store_true')
    parser.add_argument('--along', action='store_true')
    parser.add_argument('--selfreweightCalibrate', action='store_true') # Calibrate
    parser.add_argument('--lossversion', default='onehot', choices=['onehot', 'category'])
    parser.add_argument('--selfreweightSelectiveNet', action='store_true')

    return parser.parse_args()


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

    epsilon = (args.epsilon / 255.)
    pgd_alpha = (args.pgd_alpha / 255.)

    useBN = True if args.useBN else False
    along = True if args.along else False

    if args.selfreweightCalibrate or args.selfreweightSelectiveNet:
        along = True
        args.out_dim = 1
        
    transform_chain = transforms.Compose([transforms.ToTensor()])
    if args.dataset == 'CIFAR-10':
        item_train = datasets.CIFAR10(root=args.data_dir, train=True, transform=transform_chain, download=True)
        item_test = datasets.CIFAR10(root=args.data_dir, train=False, transform=transform_chain, download=True)
        num_cla = 10
    elif args.dataset == 'CIFAR-100':
        item_train = datasets.CIFAR100(root=args.data_dir, train=True, transform=transform_chain, download=True)
        item_test = datasets.CIFAR100(root=args.data_dir, train=False, transform=transform_chain, download=True)
        num_cla = 100

    # load pretrained model
    if args.model_name == 'PreActResNet18':
        model = PreActResNet18(num_classes=num_cla)
    elif args.model_name == 'PreActResNet18_twobranch_DenseV1':
        model = PreActResNet18_twobranch_DenseV1(num_classes=num_cla, out_dim=args.out_dim, use_BN=useBN, along=along)
    elif args.model_name == 'PreActResNet18_twobranch_DenseV1Multi':
        model = PreActResNet18_twobranch_DenseV1Multi(num_classes=num_cla, out_dim=args.out_dim, use_BN=useBN, along=along)
    elif args.model_name == 'PreActResNet18_twobranch_DenseV2':
        model = PreActResNet18_twobranch_DenseV2(num_classes=num_cla, out_dim=args.out_dim, use_BN=useBN, along=along)
    elif args.model_name == 'PreActResNet18_threebranch_DenseV1':
        model = PreActResNet18_threebranch_DenseV1(num_classes=num_cla, out_dim=args.out_dim, use_BN=useBN, along=along)
    elif args.model_name == 'WideResNet':
        model = WideResNet(34, num_cla, widen_factor=10, dropRate=0.0)
    elif args.model_name == 'WideResNet_twobranch_DenseV1':
        model = WideResNet_twobranch_DenseV1(34, num_cla, widen_factor=10, dropRate=0.0, along=along, use_BN=useBN, out_dim=args.out_dim)
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

    model.eval()

    if args.model_name == 'PreActResNet18_threebranch_DenseV1':
        def two_branch_model(x, m=model):
                return m(x)[:-1]
        model = two_branch_model

    if args.evalset == 'adaptiveCWtest':
        train_batches = data.DataLoader(item_train, batch_size=args.batch_size, shuffle=False, num_workers=4)
        test_batches = data.DataLoader(item_test, batch_size=args.batch_size, shuffle=False, num_workers=4)

        Classifer_succ_all = torch.tensor([]).bool().cuda()
        Detector_succ_all = torch.tensor([]).bool().cuda()
        Both_succ_all = torch.tensor([]).bool().cuda()
        ldis_all = torch.tensor([]).cuda()

        if args.threatmodel == 'l2':
            CWattack = CarliniWagnerL2Attack if not args.twobranch else CarliniWagnerL2Attack_twobranch
        elif args.threatmodel == 'linf':
            CWattack = CarliniWagnerLinfAttack if not args.twobranch else CarliniWagnerLinfAttack_twobranch


        adversary = CWattack(model, num_classes=num_cla, confidence=args.CW_confidence, targeted=args.target, learning_rate=0.01, 
                                    binary_search_steps=args.binary_search_steps, max_iterations=args.CW_iter, abort_early=True, 
                                    initial_const=0.001, clip_min=0.0, clip_max=1.0, 
                                    loss_fn=None, normalize_fn=normalize)

        # calculate train median
        if args.detectmetric == 'RR':
            evi_train_median = calculate_RR_train_median(model, train_batches, args)
            adversary_adaptive = CWattack(model, num_classes=num_cla, confidence=args.CW_confidence, targeted=args.target, learning_rate=0.01, 
                                    binary_search_steps=args.binary_search_steps, max_iterations=args.CW_iter, abort_early=True, 
                                    initial_const=0.001, clip_min=0.0, clip_max=1.0, 
                                    loss_fn=None, normalize_fn=normalize,
                                    adaptive_evi=True, evi_train_median=evi_train_median.median())
            print('Use evidence as the metric')
            print(evi_train_median.median().item())
            print(evi_train_median.std().item())

        elif args.detectmetric == 'con':
            con_train_median = calculate_con_train_median(model, train_batches, args)
            adversary_adaptive = CWattack(model, num_classes=num_cla, confidence=args.CW_confidence, targeted=args.target, learning_rate=0.01, 
                                    binary_search_steps=args.binary_search_steps, max_iterations=args.CW_iter, abort_early=True, 
                                    initial_const=0.001, clip_min=0.0, clip_max=1.0, 
                                    loss_fn=None, normalize_fn=normalize,
                                    adaptive_con=True, con_train_median=con_train_median.median())
            print('Use confidence as the metric')
            print(con_train_median.median().item())
            print(con_train_median.std().item())

        elif args.detectmetric == 'Selective':
            evi_train_median = calculate_Selective_train_median(model, train_batches, args)
            adversary_adaptive = CWattack(model, num_classes=num_cla, confidence=args.CW_confidence, targeted=args.target, learning_rate=0.01, 
                                    binary_search_steps=args.binary_search_steps, max_iterations=args.CW_iter, abort_early=True, 
                                    initial_const=0.001, clip_min=0.0, clip_max=1.0, 
                                    loss_fn=None, normalize_fn=normalize,
                                    adaptive_Selective=True, evi_train_median=evi_train_median.median())
            print('Use evidence (Selective) as the metric')
            print(evi_train_median.median().item())
            print(evi_train_median.std().item())

        elif args.detectmetric == 'NIPS20':
            evi_train_median = calculate_NIPS20_train_median(model, train_batches, args)
            adversary_adaptive = CWattack(model, num_classes=num_cla, confidence=args.CW_confidence, targeted=args.target, learning_rate=0.01, 
                                    binary_search_steps=args.binary_search_steps, max_iterations=args.CW_iter, abort_early=True, 
                                    initial_const=0.001, clip_min=0.0, clip_max=1.0, 
                                    loss_fn=None, normalize_fn=normalize,
                                    adaptive_evi=True, evi_train_median=evi_train_median.median())
            print('Use evidence (NIPS20) as the metric')
            print(evi_train_median.median().item())
            print(evi_train_median.std().item())

        for i, (X, y) in enumerate(test_batches):
            X, y = X.cuda(), y.cuda()
            print('Running CW attack for number ', i)

            # Phase one: non-adaptive attack
            if args.target:
                y_target = sample_targetlabel(y, num_classes=num_cla)
                adv_input = adversary.perturb(X, y_target)
            else:
                adv_input = adversary.perturb(X, y)

            if args.twobranch:
                final_output_class, final_output_evi = model(normalize(adv_input))
            else:
                final_output_class = model(normalize(adv_input))
                final_output_evi = final_output_class

            con_output, labels_output = F.softmax(final_output_class, dim=1).max(1)

            if args.selfreweightCalibrate:
                evidence_output = con_output * final_output_evi.sigmoid().squeeze()
            elif args.selfreweightSelectiveNet:
                evidence_output = final_output_evi.sigmoid().squeeze()
            elif not args.twobranch:
                evidence_output = final_output_evi.logsumexp(dim=1)

            Classifer_succ = labels_output != y

            if args.detectmetric == 'con':
                Detector_succ = con_output > con_train_median.median()
            else:
                Detector_succ = evidence_output > evi_train_median.median()

            Both_succ = Classifer_succ & Detector_succ

            if args.reportmodel == 'l2':
                ldis = ((adv_input - X) ** 2).view(X.size()[0], -1).sum(dim=1)
                logger.info('Middle outputs | Classifer succ: %.4f, Detector succ: %.4f, Both succ: %.4f, l2dis: %.4f',
                    Classifer_succ.float().mean(), Detector_succ.float().mean(), Both_succ.float().mean(), ldis.mean())
            else:
                ldis,_ = torch.max(torch.abs(adv_input - X).view(X.size()[0], -1), dim=1)
                ldis *= 255
                logger.info('Middle outputs | Classifer succ: %.4f, Detector succ: %.4f, Both succ: %.4f, linfdis: %.4f',
                    Classifer_succ.float().mean(), Detector_succ.float().mean(), Both_succ.float().mean(), ldis.mean())

            # Phase two: adaptive attack
            if args.target:
                adv_input_adaptive = adversary_adaptive.perturb(adv_input, y_target)
            else:
                adv_input_adaptive = adversary_adaptive.perturb(adv_input, y)

            if args.twobranch:
                final_output_class, final_output_evi = model(normalize(adv_input_adaptive))
            else:
                final_output_class = model(normalize(adv_input_adaptive))
                final_output_evi = final_output_class

            con_output, labels_output = F.softmax(final_output_class, dim=1).max(1)

            if args.selfreweightCalibrate:
                evidence_output = con_output * final_output_evi.sigmoid().squeeze()
            elif args.selfreweightSelectiveNet:
                evidence_output = final_output_evi.sigmoid().squeeze()
            elif not args.twobranch:
                evidence_output = final_output_evi.logsumexp(dim=1)

            Classifer_succ = labels_output != y

            if args.detectmetric == 'con':
                Detector_succ = con_output > con_train_median.median()
            else:
                Detector_succ = evidence_output > evi_train_median.median()
            
            Both_succ = Classifer_succ & Detector_succ

            if args.reportmodel == 'l2':
                ldis = ((adv_input_adaptive - X) ** 2).view(X.size()[0], -1).sum(dim=1)
                logger.info('Final outputs  | Classifer succ: %.4f, Detector succ: %.4f, Both succ: %.4f, l2dis: %.4f',
                    Classifer_succ.float().mean(), Detector_succ.float().mean(), Both_succ.float().mean(), ldis.mean())
            else:
                ldis,_ = torch.max(torch.abs(adv_input_adaptive - X).view(X.size()[0], -1), dim=1)
                ldis *= 255
                logger.info('Final outputs  | Classifer succ: %.4f, Detector succ: %.4f, Both succ: %.4f, linfdis: %.4f',
                    Classifer_succ.float().mean(), Detector_succ.float().mean(), Both_succ.float().mean(), ldis.mean())

            Classifer_succ_all = torch.cat((Classifer_succ_all, Classifer_succ), dim=0)
            Detector_succ_all = torch.cat((Detector_succ_all, Detector_succ), dim=0)
            Both_succ_all = torch.cat((Both_succ_all, Both_succ), dim=0)
            ldis_all = torch.cat((ldis_all, ldis), dim=0)

        logger.info('Total results on test set | Classifer succ: %.4f, Detector succ: %.4f, Both succ: %.4f, ldis: %.4f',
                Classifer_succ_all.float().mean(), Detector_succ_all.float().mean(), Both_succ_all.float().mean(), ldis_all.mean())
           

if __name__ == "__main__":
    main()
