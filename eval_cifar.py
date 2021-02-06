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
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from autoattack import AutoAttack
import os

from models import *
from utils import *

def CW_loss(x, y):
    x_sorted, ind_sorted = x.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()
    
    loss_value = -(x[np.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind))
    return loss_value.mean()

def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts=1, norm='l_inf',
                twobranch=False, use_CWloss=False):
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
            if use_CWloss:
                loss = CW_loss(output, y)
            else:
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
    parser.add_argument('--lossversion', default='onehot', choices=['onehot', 'category'])
    parser.add_argument('--tempC', default=1., type=float)
    parser.add_argument('--evalonAA', action='store_true')# evaluate on AutoAttack
    parser.add_argument('--evalonCWloss', action='store_true')# evaluate on PGD with CW loss
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

    if args.selfreweightCalibrate:
        along = True
        args.out_dim = 1

    transform_chain = transforms.Compose([transforms.ToTensor()])
    if args.dataset == 'CIFAR-10':
        item = datasets.CIFAR10(root=args.data_dir, train=False, transform=transform_chain, download=True)
        num_cla = 10
    elif args.dataset == 'CIFAR-100':
        item = datasets.CIFAR100(root=args.data_dir, train=False, transform=transform_chain, download=True)
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
    elif args.model_name == 'WideResNet':
        model = WideResNet(34, num_cla, widen_factor=10, dropRate=0.0)
    elif args.model_name == 'WideResNet_twobranch_DenseV1':
        model = WideResNet_twobranch_DenseV1(34, num_cla, widen_factor=10, dropRate=0.0, along=along, use_BN=useBN, out_dim=args.out_dim)
    elif args.model_name == 'WideResNet_20':
        model = WideResNet(34, num_cla, widen_factor=20, dropRate=0.0)
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


    if args.twobranch:
        def normalize_model(x):
            return model(normalize(x))[0]
    else:
        def normalize_model(x):
            return model(normalize(x))

    adversary_AA = AutoAttack(normalize_model, norm='Linf', eps=epsilon, version='standard', verbose=True)

    if args.evalset == 'test':
        test_batches = data.DataLoader(item, batch_size=128, shuffle=False, num_workers=4)
        
        test_acc, test_robust_acc, test_n = 0, 0, 0
        test_classes_correct, test_classes_wrong = [], []
        test_classes_robust_correct, test_classes_robust_wrong = [], []

         # record con
        test_con_correct, test_robust_con_correct = [], []
        test_con_wrong, test_robust_con_wrong = [], []

        # record evi
        test_evi_correct, test_robust_evi_correct = [], []
        test_evi_wrong, test_robust_evi_wrong = [], []

        # record truecon
        test_truecon_correct, test_robust_truecon_correct = [], []
        test_truecon_wrong, test_robust_truecon_wrong = [], []

        for i, (X, y) in enumerate(test_batches):
            X, y = X.cuda(), y.cuda()

            if args.evalonAA:
                X_adv = adversary_AA.run_standard_evaluation(X, y, bs=128)

            else:
                if args.target:
                    y_target = sample_targetlabel(y, num_classes=10)
                    delta = attack_pgd(model, X, y_target, epsilon, pgd_alpha, args.attack_iters, args.restarts, args.norm, target=True, twobranch=args.twobranch, use_CWloss=args.evalonCWloss)
                else:
                    delta = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters, args.restarts, args.norm, twobranch=args.twobranch, use_CWloss=args.evalonCWloss)
                delta = delta.detach()
                X_adv = X + delta

            if args.twobranch:
                output, output_aux = model(normalize(X))
                robust_output, robust_output_aux = model(normalize(torch.clamp(X_adv, min=lower_limit, max=upper_limit)))

                con_pre, _ = torch.softmax(output * args.tempC, dim=1).max(1) # predicted label and confidence
                robust_con_pre, _ = torch.softmax(robust_output * args.tempC, dim=1).max(1) # predicted label and confidence

                if args.selfreweightCalibrate:
                    output_aux = output_aux.sigmoid().squeeze()
                    robust_output_aux = robust_output_aux.sigmoid().squeeze() # bs x 1, Calibration function A \in [0,1]
                    test_evi_all = con_pre * output_aux
                    test_robust_evi_all = robust_con_pre * robust_output_aux
                    if args.ConfidenceOnly:
                        test_evi_all = con_pre
                        test_robust_evi_all = robust_con_pre
                    if args.AuxiliaryOnly:
                        test_evi_all = output_aux
                        test_robust_evi_all = robust_output_aux
            
            else:
                output = model(normalize(X))
                robust_output = model(normalize(torch.clamp(X_adv, min=lower_limit, max=upper_limit)))
                test_evi_all = output.logsumexp(dim=1)
                test_robust_evi_all = robust_output.logsumexp(dim=1)


            output_s = F.softmax(output, dim=1)
            out_con, out_pre = output_s.max(1)
            out_truecon = output_s[torch.tensor(range(X.size(0))), y]

            ro_output_s = F.softmax(robust_output, dim=1)
            ro_out_con, ro_out_pre = ro_output_s.max(1)
            ro_out_truecon = ro_output_s[torch.tensor(range(X.size(0))), y]

            # output labels
            labels = torch.where(out_pre == y)[0]
            robust_labels = torch.where(ro_out_pre == y)[0]
            labels_n = torch.where(out_pre != y)[0]
            robust_labels_n = torch.where(ro_out_pre != y)[0]

            # ground labels
            test_classes_correct += y[labels].tolist()
            test_classes_wrong += y[labels_n].tolist()
            test_classes_robust_correct += y[robust_labels].tolist()
            test_classes_robust_wrong += y[robust_labels_n].tolist()

            # accuracy
            test_acc += labels.size(0)
            test_robust_acc += robust_labels.size(0)

            # confidence
            test_con_correct += out_con[labels].tolist()
            test_con_wrong += out_con[labels_n].tolist()
            test_robust_con_correct += ro_out_con[robust_labels].tolist()
            test_robust_con_wrong += ro_out_con[robust_labels_n].tolist()

            # true confidence
            test_truecon_correct += out_truecon[labels].tolist()
            test_truecon_wrong += out_truecon[labels_n].tolist()
            test_robust_truecon_correct += ro_out_truecon[robust_labels].tolist()
            test_robust_truecon_wrong += ro_out_truecon[robust_labels_n].tolist()
                        
            # evidence
            test_evi_correct += test_evi_all[labels].tolist()
            test_evi_wrong += test_evi_all[labels_n].tolist()
            test_robust_evi_correct += test_robust_evi_all[robust_labels].tolist()
            test_robust_evi_wrong += test_robust_evi_all[robust_labels_n].tolist()
            
            test_n += y.size(0)

            print('Finish ', i)

        np.savetxt('eval_results/test_classes_correct.txt', np.array(test_classes_correct))
        np.savetxt('eval_results/test_classes_wrong.txt', np.array(test_classes_wrong))
        np.savetxt('eval_results/test_classes_robust_correct.txt', np.array(test_classes_robust_correct))
        np.savetxt('eval_results/test_classes_robust_wrong.txt', np.array(test_classes_robust_wrong))

        # confidence
        test_con_correct = torch.tensor(test_con_correct)
        test_robust_con_correct = torch.tensor(test_robust_con_correct)
        test_con_wrong = torch.tensor(test_con_wrong)
        test_robust_con_wrong = torch.tensor(test_robust_con_wrong)

        # true confidence
        test_truecon_correct = torch.tensor(test_truecon_correct)
        test_robust_truecon_correct = torch.tensor(test_robust_truecon_correct)
        test_truecon_wrong = torch.tensor(test_truecon_wrong)
        test_robust_truecon_wrong = torch.tensor(test_robust_truecon_wrong)

        # evidence
        test_evi_correct = torch.tensor(test_evi_correct)
        test_robust_evi_correct = torch.tensor(test_robust_evi_correct)
        test_evi_wrong = torch.tensor(test_evi_wrong)
        test_robust_evi_wrong = torch.tensor(test_robust_evi_wrong)

        test_acc = test_acc/test_n
        test_robust_acc = test_robust_acc/test_n

        print('### Basic statistics ###')
        logger.info('Clean       | acc: %.4f | con cor: %.3f (%.3f) | con wro: %.3f (%.3f) | evi cor: %.3f (%.3f) | evi wro: %.3f (%.3f)', 
            test_acc, 
            test_con_correct.mean().item(), test_con_correct.std().item(),
            test_con_wrong.mean().item(), test_con_wrong.std().item(),
            test_evi_correct.mean().item(), test_evi_correct.std().item(),
            test_evi_wrong.mean().item(), test_evi_wrong.std().item())

        logger.info('Robust      | acc: %.4f | con cor: %.3f (%.3f) | con wro: %.3f (%.3f) | evi cor: %.3f (%.3f) | evi wro: %.3f (%.3f)', 
            test_robust_acc, 
            test_robust_con_correct.mean().item(), test_robust_con_correct.std().item(),
            test_robust_con_wrong.mean().item(), test_robust_con_wrong.std().item(),  
            test_robust_evi_correct.mean().item(), test_robust_evi_correct.std().item(),
            test_robust_evi_wrong.mean().item(), test_robust_evi_wrong.std().item())


        print('')
        print('### ROC-AUC scores (confidence) ###')
        clean_clean = calculate_auc_scores(test_con_correct, test_con_wrong)
        _, acc95 = calculate_FPR_TPR(test_con_correct, test_con_wrong, tpr_ref=0.95)
        _, acc99 = calculate_FPR_TPR(test_con_correct, test_con_wrong, tpr_ref=0.99)
        robust_robust = calculate_auc_scores(test_robust_con_correct, test_robust_con_wrong)
        _, ro_acc95 = calculate_FPR_TPR(test_robust_con_correct, test_robust_con_wrong, tpr_ref=0.95)
        _, ro_acc99 = calculate_FPR_TPR(test_robust_con_correct, test_robust_con_wrong, tpr_ref=0.99)
        logger.info('clean_clean: %.3f | robust_robust: %.3f', 
            clean_clean, robust_robust)
        logger.info('TPR 95 clean acc: %.4f; 99 clean acc: %.4f | TPR 95 robust acc: %.4f; 99 robust acc: %.4f', 
            acc95 - test_acc, acc99 - test_acc, ro_acc95 - test_robust_acc, ro_acc99 - test_robust_acc)
        
        np.savetxt('eval_results/test_robust_con_correct.txt', test_robust_con_correct.cpu().numpy())
        np.savetxt('eval_results/test_robust_con_wrong.txt', test_robust_con_wrong.cpu().numpy())
        np.savetxt('eval_results/test_con_correct.txt', test_con_correct.cpu().numpy())
        np.savetxt('eval_results/test_con_wrong.txt', test_con_wrong.cpu().numpy())

        np.savetxt('eval_results/test_robust_truecon_correct.txt', test_robust_truecon_correct.cpu().numpy())
        np.savetxt('eval_results/test_robust_truecon_wrong.txt', test_robust_truecon_wrong.cpu().numpy())
        np.savetxt('eval_results/test_truecon_correct.txt', test_truecon_correct.cpu().numpy())
        np.savetxt('eval_results/test_truecon_wrong.txt', test_truecon_wrong.cpu().numpy())

        
        print('')
        print('### ROC-AUC scores (evidence) ###')
        clean_clean = calculate_auc_scores(test_evi_correct, test_evi_wrong)
        _, acc95 = calculate_FPR_TPR(test_evi_correct, test_evi_wrong, tpr_ref=0.95)
        _, acc99 = calculate_FPR_TPR(test_evi_correct, test_evi_wrong, tpr_ref=0.99)
        robust_robust = calculate_auc_scores(test_robust_evi_correct, test_robust_evi_wrong)
        _, ro_acc95 = calculate_FPR_TPR(test_robust_evi_correct, test_robust_evi_wrong, tpr_ref=0.95)
        _, ro_acc99 = calculate_FPR_TPR(test_robust_evi_correct, test_robust_evi_wrong, tpr_ref=0.99)
        logger.info('clean_clean: %.3f | robust_robust: %.3f', 
            clean_clean, robust_robust)
        logger.info('TPR 95 clean acc: %.4f; 99 clean acc: %.4f | TPR 95 robust acc: %.4f; 99 robust acc: %.4f', 
            acc95 - test_acc, acc99 - test_acc, ro_acc95 - test_robust_acc, ro_acc99 - test_robust_acc)
        
        
        np.savetxt('eval_results/test_robust_evi_correct.txt', test_robust_evi_correct.cpu().numpy())
        np.savetxt('eval_results/test_robust_evi_wrong.txt', test_robust_evi_wrong.cpu().numpy())
        np.savetxt('eval_results/test_evi_correct.txt', test_evi_correct.cpu().numpy())
        np.savetxt('eval_results/test_evi_wrong.txt', test_evi_wrong.cpu().numpy())

if __name__ == "__main__":
    main()
