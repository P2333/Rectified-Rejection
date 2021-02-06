import argparse
import logging
import sys
import time
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import os
from models import *
from utils import *

def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, norm, BNeval=False,
                threebranch=False):
    if BNeval:
        model.eval()
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
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
            if threebranch:
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
        if threebranch:
            all_loss = F.cross_entropy(model(normalize(X+delta))[0], y, reduction='none')
        else:
            all_loss = F.cross_entropy(model(normalize(X+delta)), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    if BNeval:
        model.train()
    return max_delta, 0 


def get_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--model', default='PreActResNet18')
    parser.add_argument('--model_name', type=str, default='PreActResNet18')
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--dataset', default='CIFAR-10', type=str)
    parser.add_argument('--data-dir', default='../cifar-data', type=str)
    parser.add_argument('--epochs', default=110, type=int)
    parser.add_argument('--lr-schedule', default='piecewise')
    parser.add_argument('--lr-max', default=0.1, type=float)
    parser.add_argument('--lr-one-drop', default=0.01, type=float)
    parser.add_argument('--lr-drop-epoch', default=100, type=int)
    parser.add_argument('--attack', default='pgd', type=str, choices=['pgd', 'fgsm', 'free', 'none'])
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--attack-iters', default=10, type=int)
    parser.add_argument('--restarts', default=1, type=int)
    parser.add_argument('--pgd-alpha', default=2, type=float)
    parser.add_argument('--fgsm-alpha', default=1.25, type=float)
    parser.add_argument('--norm', default='l_inf', type=str, choices=['l_inf', 'l_2'])
    parser.add_argument('--fgsm-init', default='random', choices=['zero', 'random', 'previous'])
    parser.add_argument('--fname', default='cifar_model', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--width-factor', default=10, type=int)
    parser.add_argument('--resume', default=0, type=int)
    parser.add_argument('--cutout', action='store_true')
    parser.add_argument('--cutout-len', type=int)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--chkpt-iters', default=10, type=int)
    parser.add_argument('--weight_decay', default=5e-4, type=float)#weight decay
    parser.add_argument('--optimizer', default='SGD', type=str)
    parser.add_argument('--target', action='store_true') # whether use target-mode attack

    ### adaptive attack
    parser.add_argument('--BNeval', action='store_true') # whether use eval mode for BN when crafting adversarial examples

    ### adaptive training
    parser.add_argument('--adaptivetrain', action='store_true') # whether use adaptive term in train
    parser.add_argument('--adaptivetrainlambda', default=1., type=float)

    parser.add_argument('--selfreweightSelectiveNet', action='store_true')
    parser.add_argument('--Lambda', default=32, type=float)
    parser.add_argument('--coverage', default=0.7, type=float)

    # two branch for our selfreweightCalibrate (rectified rejection)
    parser.add_argument('--threebranch', action='store_true')
    parser.add_argument('--out_dim', default=10, type=int)
    parser.add_argument('--useBN', action='store_true')
    parser.add_argument('--along', action='store_true')

    parser.add_argument('--warmup', default='none', choices=['none', 'gradually','distinct'])
    parser.add_argument('--warmupepoch', default=30, type=int)
    
    return parser.parse_args()

def get_auto_fname(args):
    if args.attack == 'fgsm':
        names = 'FastAT_' + args.model_name
    else:
        names = 'PGDAT_' + args.model_name
    if args.useBN:
        names += 'BN'
    if args.BNeval:
        names += '_BNeval'
    if args.adaptivetrain:
        names += '_adaptiveT' + str(args.adaptivetrainlambda)
        if args.selfreweightSelectiveNet:
            names += '_selfreweightSelectiveNet' + 'Lambda' + str(args.Lambda) + 'coverage' + str(args.coverage)
    if args.warmup != 'none':
        names += '_' + args.warmup + str(args.warmupepoch)
    if args.weight_decay != 5e-4:
        names = names + '_wd' + str(args.weight_decay)
    if args.epochs != 110:
        names += '_epochs' + str(args.epochs)
    if args.batch_size != 128:
        names += '_bs' + str(args.batch_size)
    if args.epsilon != 8:
        names += '_eps' + str(args.epsilon)
    names += '_seed' + str(args.seed)
    print('File name: ', names)
    return names

def main():
    args = get_args()
    epsilon = (args.epsilon / 255.)
    pgd_alpha = (args.pgd_alpha / 255.)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.fname == 'auto':
        names = get_auto_fname(args)
        args.fname = 'trained_models/' + args.dataset + '/' + names
    else:
        args.fname = 'trained_models/' + args.dataset + '/' + args.fname

    if not os.path.exists(args.fname):
        os.makedirs(args.fname)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(args.fname, 'eval.log' if args.eval else 'output.log')),
            logging.StreamHandler()
        ])

    logger.info(args)


    # Prepare dataset
    transforms = [Crop(32, 32), FlipLR()]
    if args.cutout:
        transforms.append(Cutout(args.cutout_len, args.cutout_len))

    if args.dataset == 'CIFAR-10':
        dataset = cifar10(args.data_dir)
        num_cla = 10
    elif args.dataset == 'CIFAR-100':
        dataset = cifar100(args.data_dir)
        num_cla = 100

    train_set = list(zip(transpose(pad(dataset['train']['data'], 4)/255.),
        dataset['train']['labels']))
    train_set_x = Transform(train_set, transforms)
    train_batches = Batches(train_set_x, args.batch_size, shuffle=True, set_random_choices=True, num_workers=2)
    test_set = list(zip(transpose(dataset['test']['data']/255.), dataset['test']['labels']))
    test_batches = Batches(test_set, args.batch_size, shuffle=False, num_workers=2)

    useBN = True if args.useBN else False
    along = True if args.along else False

    if args.selfreweightSelectiveNet:
        along = True
        args.out_dim = 1
        
    # Creat model
    if args.model_name == 'PreActResNet18_threebranch_DenseV1':
        model = PreActResNet18_threebranch_DenseV1(num_classes=num_cla, out_dim=args.out_dim, use_BN=useBN, along=along)
    elif args.model_name == 'WideResNet_threebranch_DenseV1':
        model = WideResNet_threebranch_DenseV1(34, num_cla, widen_factor=10, dropRate=0.0, use_BN=useBN, along=along, out_dim=args.out_dim)
    else:
        raise ValueError("Unknown model")

    model = nn.DataParallel(model).cuda()
    model.train()
    params = model.parameters()

    if args.optimizer == 'SGD':
        opt = torch.optim.SGD(params, lr=args.lr_max, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        opt = torch.optim.Adam(params, lr=args.lr_max, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay, amsgrad=False)

    if args.attack == 'free':
        epochs = int(math.ceil(args.epochs / args.attack_iters))
    else:
        epochs = args.epochs

    if args.attack == 'fgsm':
        epochs = 20
        lr_max = 0.2
        epoch_alter = epochs / 2.
        def lr_schedule(t):
            if t < epoch_alter:
                return lr_max * t / epoch_alter
            else:
                return lr_max * (epochs - t) / epoch_alter
    else:
        def lr_schedule(t):
            if t < 100:
                return args.lr_max
            elif t < 105:
                return args.lr_max / 10.
            else:
                return args.lr_max / 100.


    best_test_robust_acc = 0
    best_val_robust_acc = 0
    start_epoch = 0

    criterion = nn.CrossEntropyLoss()
    criterion_none = nn.CrossEntropyLoss(reduction='none')
    BCEcriterion = nn.BCELoss()
    

    logger.info('Epoch \t Acc \t Robust Acc \t Evi \t Robust Evi')
    for epoch in range(start_epoch, epochs):
        model.train()
        start_time = time.time()
        for i, batch in enumerate(train_batches):
            if args.eval:
                break
            X, y = batch['input'], batch['target']
            epoch_now = epoch + (i + 1) / len(train_batches)
            lr = lr_schedule(epoch_now)
            opt.param_groups[0].update(lr=lr)

            if args.attack == 'pgd':
                # Random initialization
                delta, _ = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters, args.restarts, args.norm,
                 BNeval=args.BNeval, threebranch=args.threebranch)
                delta = delta.detach()
            elif args.attack == 'fgsm':
                delta, _ = attack_pgd(model, X, y, epsilon, args.fgsm_alpha*epsilon, 1, 1, args.norm,
                 BNeval=args.BNeval, threebranch=args.threebranch)
            # Standard training
            elif args.attack == 'none':
                delta = torch.zeros_like(X)


            # whether use three branches
            if args.threebranch:
                robust_output, robust_output_aux, robust_output_H = model(normalize(torch.clamp(X + delta, min=lower_limit, max=upper_limit)))
                robust_output_aux = robust_output_aux.sigmoid().squeeze() # bs, Calibration function A \in [0,1]
            
            if args.adaptivetrain:
                if args.selfreweightSelectiveNet:
                    beta = 0.5
                    robust_loss_H = criterion(robust_output_H, y)
                    robust_loss_1 = beta * robust_loss_H

                    sum_aux = robust_output_aux.mean(dim=0)
                    robust_loss_2 = (1-beta) * args.Lambda * torch.pow(F.relu(args.coverage - sum_aux), 2)

                    robust_loss_none = criterion_none(robust_output, y)
                    robust_output_aux_re = (1-beta) * args.adaptivetrainlambda * robust_output_aux / robust_output_aux.sum(dim=0)
                    robust_loss_3 = (robust_loss_none * robust_output_aux_re).sum(dim=0)

                    robust_loss = robust_loss_1 + robust_loss_2 + robust_loss_3

            opt.zero_grad()
            robust_loss.backward()
            opt.step()

        if args.attack == 'fgsm':
            l = lr_schedule(epoch)
            print('lr: ', l)

        model.eval()
        test_acc = 0
        test_robust_acc = 0

        test_evi_correct = 0
        test_robust_evi_correct = 0

        test_evi_wrong = 0
        test_robust_evi_wrong = 0

        test_n = 0
        for i, batch in enumerate(test_batches):
            X, y = batch['input'], batch['target']

            # Random initialization
            if args.attack == 'none':
                delta = torch.zeros_like(X)
            else:
                delta,_ = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters, args.restarts, args.norm, threebranch=args.threebranch)
            
            delta = delta.detach()

            if args.threebranch:
                output, output_aux, _ = model(normalize(X))
                robust_output, robust_output_aux, _ = model(normalize(torch.clamp(X + delta, min=lower_limit, max=upper_limit)))
                output_aux = output_aux.sigmoid().squeeze()
                robust_output_aux = robust_output_aux.sigmoid().squeeze() # bs x 1, Calibration function A \in [0,1]

                if args.selfreweightSelectiveNet:
                    test_evi_all = output_aux
                    test_robust_evi_all = robust_output_aux
                   

            # output labels
            labels = torch.where(output.max(1)[1] == y)[0]
            robust_labels = torch.where(robust_output.max(1)[1] == y)[0]

            # accuracy
            test_acc += labels.size(0)
            test_robust_acc += robust_labels.size(0)

            # standard evidence 
            test_evi_correct += test_evi_all[labels].sum().item()
            test_evi_wrong += test_evi_all.sum().item() - test_evi_all[labels].sum().item()

            # robust evidence
            test_robust_evi_correct += test_robust_evi_all[robust_labels].sum().item()
            test_robust_evi_wrong += test_robust_evi_all.sum().item() - test_robust_evi_all[robust_labels].sum().item()

            test_n += y.size(0)

        test_time = time.time()


        logger.info('%d \t %.4f \t %.4f \t (%.4f / %.4f) \t (%.4f / %.4f)', epoch, test_acc/test_n, test_robust_acc/test_n,
            test_evi_correct/test_acc, test_evi_wrong/(test_n-test_acc),
            test_robust_evi_correct/test_robust_acc, test_robust_evi_wrong/(test_n-test_robust_acc))

        # save best
        if test_robust_acc/test_n > best_test_robust_acc:
            torch.save({
                    'state_dict':model.state_dict(),
                    'test_robust_acc':test_robust_acc/test_n,
                    'test_acc':test_acc/test_n,
                }, os.path.join(args.fname, f'model_best.pth'))
            best_test_robust_acc = test_robust_acc/test_n


 
    # calculate AUC
    if True:
        model_dict = torch.load(os.path.join(args.fname, f'model_best.pth'))
        logger.info(f'Resuming at best epoch')

        if 'state_dict' in model_dict.keys():
            model.load_state_dict(model_dict['state_dict'])
        else:
            model.load_state_dict(model_dict)

        model.eval()
       
        test_acc = 0
        test_robust_acc = 0
        test_n = 0
        test_classes_correct = []
        test_classes_wrong = []
        test_classes_robust_correct = []
        test_classes_robust_wrong = []

        # record con
        test_con_correct = []
        test_robust_con_correct = []
        test_con_wrong = []
        test_robust_con_wrong = []

        # record evi
        test_evi_correct = []
        test_robust_evi_correct = []
        test_evi_wrong = []
        test_robust_evi_wrong = []

        for i, batch in enumerate(test_batches):
            X, y = batch['input'].cuda(), batch['target'].cuda()

            if args.target:
                y_target = sample_targetlabel(y, num_classes=num_cla)
                delta,_ = attack_pgd(model, X, y_target, epsilon, pgd_alpha, args.attack_iters, args.restarts, args.norm, target=True, threebranch=args.threebranch)
            else:
                delta,_ = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters, args.restarts, args.norm, threebranch=args.threebranch)
            
            delta = delta.detach()

            if args.threebranch:
                output, output_aux, _ = model(normalize(X))
                robust_output, robust_output_aux, _ = model(normalize(torch.clamp(X + delta, min=lower_limit, max=upper_limit)))
                output_aux = output_aux.sigmoid().squeeze()
                robust_output_aux = robust_output_aux.sigmoid().squeeze() # bs x 1, Calibration function A \in [0,1]

                if args.selfreweightSelectiveNet:
                    test_evi_all = output_aux
                    test_robust_evi_all = robust_output_aux                   
        
            output_s = F.softmax(output, dim=1)
            out_con, out_pre = output_s.max(1)

            ro_output_s = F.softmax(robust_output, dim=1)
            ro_out_con, ro_out_pre = ro_output_s.max(1)
            
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
                        
            # evidence
            test_evi_correct += test_evi_all[labels].tolist()
            test_evi_wrong += test_evi_all[labels_n].tolist()
            test_robust_evi_correct += test_robust_evi_all[robust_labels].tolist()
            test_robust_evi_wrong += test_robust_evi_all[robust_labels_n].tolist()
            
            test_n += y.size(0)

            print('Finish ', i)

        # confidence
        test_con_correct = torch.tensor(test_con_correct)
        test_robust_con_correct = torch.tensor(test_robust_con_correct)
        test_con_wrong = torch.tensor(test_con_wrong)
        test_robust_con_wrong = torch.tensor(test_robust_con_wrong)

        # evidence
        test_evi_correct = torch.tensor(test_evi_correct)
        test_robust_evi_correct = torch.tensor(test_robust_evi_correct)
        test_evi_wrong = torch.tensor(test_evi_wrong)
        test_robust_evi_wrong = torch.tensor(test_robust_evi_wrong)

        print('### Basic statistics ###')
        logger.info('Clean       | acc: %.4f | con cor: %.3f (%.3f) | con wro: %.3f (%.3f) | evi cor: %.3f (%.3f) | evi wro: %.3f (%.3f)', 
            test_acc/test_n, 
            test_con_correct.mean().item(), test_con_correct.std().item(),
            test_con_wrong.mean().item(), test_con_wrong.std().item(),
            test_evi_correct.mean().item(), test_evi_correct.std().item(),
            test_evi_wrong.mean().item(), test_evi_wrong.std().item())

        logger.info('Robust      | acc: %.4f | con cor: %.3f (%.3f) | con wro: %.3f (%.3f) | evi cor: %.3f (%.3f) | evi wro: %.3f (%.3f)', 
            test_robust_acc/test_n, 
            test_robust_con_correct.mean().item(), test_robust_con_correct.std().item(),
            test_robust_con_wrong.mean().item(), test_robust_con_wrong.std().item(),  
            test_robust_evi_correct.mean().item(), test_robust_evi_correct.std().item(),
            test_robust_evi_wrong.mean().item(), test_robust_evi_wrong.std().item())


        print('')
        print('### ROC-AUC scores (confidence) ###')
        clean_clean = calculate_auc_scores(test_con_correct, test_con_wrong)
        robust_robust = calculate_auc_scores(test_robust_con_correct, test_robust_con_wrong)
        logger.info('clean_clean: %.3f | robust_robust: %.3f', 
            clean_clean, robust_robust)
        

        
        print('')
        print('### ROC-AUC scores (evidence) ###')
        clean_clean = calculate_auc_scores(test_evi_correct, test_evi_wrong)
        robust_robust = calculate_auc_scores(test_robust_evi_correct, test_robust_evi_wrong)
        logger.info('clean_clean: %.3f | robust_robust: %.3f', 
            clean_clean, robust_robust)
        
        


if __name__ == "__main__":
    main()
