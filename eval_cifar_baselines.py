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

import os

from models import *
from utils import *

def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts=1, norm='l_inf', temp=1.):
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
            output,_ = model(normalize(X + delta))
            loss = F.cross_entropy(output / temp, y)
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
        out,_ = model(normalize(X+delta))
        all_loss = F.cross_entropy(out / temp, y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta

def Kernel_density_train(model, train_batches, feature_dim=512, dataset='CIFAR-10'):
    print('Crafting kernel density points on training set')
    if dataset == 'CIFAR-10':
        num = 1000
        num_class = 10
    elif dataset == 'CIFAR-100':
        num = 100
        num_class = 100
    return_back = torch.zeros(num_class, num, feature_dim)
    counts = np.array([0] * num_class)
    for i, (X, y) in enumerate(train_batches):
        if np.sum(counts) == (num * num_class):
            break
        X, y = X.cuda(), y.cuda()
        output, features = model(normalize(X)) # features: 128 x 512
        _, pre_labels = output.max(1) # pre_labels : 128
        c_or_w = torch.where(pre_labels == y)[0]
        for j in range(c_or_w.size(0)):
            l = y[c_or_w[j]]
            if counts[l] < num:
                return_back[l, counts[l], :] = features[c_or_w[j]].detach()
                counts[l] += 1
    print('Finished!')
    return return_back.cuda()

def LID_train(model, train_batches, num=1000, feature_dim=512, dataset='CIFAR-10'):
    print('Crafting LID references on training set')
    return_back = torch.zeros(num, feature_dim)
    counts = 0
    for i, (X, y) in enumerate(train_batches):
        if counts == num:
            break
        X, y = X.cuda(), y.cuda()
        output, features = model(normalize(X)) # features: 128 x 512
        _, pre_labels = output.max(1) # pre_labels : 128
        c_or_w = torch.where(pre_labels == y)[0]
        for j in range(c_or_w.size(0)):
            if counts < num:
                return_back[counts] = features[c_or_w[j]].detach()
                counts += 1
    print('Finished!')
    return return_back.cuda()

def GDA_train(model, train_batches, feature_dim=512, dataset='CIFAR-10'):
    print('Crafting GDA parameters on training set')
    if dataset == 'CIFAR-10':
        num_class = 10
    elif dataset == 'CIFAR-100':
        num_class = 100
    dic = {}
    for i in range(num_class):
        dic[str(i)] = torch.tensor([]).cuda()
    for i, (X, y) in enumerate(train_batches):
        print(i)
        X, y = X.cuda(), y.cuda()
        output, features = model(normalize(X)) # features: 128 x 512
        _, pre_labels = output.max(1) # pre_labels : 128
        c_or_w = (pre_labels == y)
        for j in range(num_class):
            is_j = torch.bitwise_and(c_or_w, (y == j))
            indexs = torch.where(is_j)[0]
            dic[str(j)] = torch.cat((dic[str(j)], features[indexs].detach()), dim=0)        
    mu = torch.zeros(num_class, feature_dim).cuda()
    sigma, num = 0, 0
    for i in range(num_class):
        dic_i = dic[str(i)]
        num += dic_i.size(0)
        mu[i] = dic_i.mean(dim=0)
        gap = dic_i - mu[i].unsqueeze(dim=0) # 1 x 512
        sigma += torch.mm(gap.t(), gap) # 512 x 512
    sigma += 1e-10 * torch.eye(feature_dim).cuda()
    sigma /= num
    print('Finished!')
    return mu, sigma

def GDAstar_train(model, train_batches, feature_dim=512, dataset='CIFAR-10'):
    print('Crafting GMM parameters on training set')
    if dataset == 'CIFAR-10':
        num_class = 10
    elif dataset == 'CIFAR-100':
        num_class = 100
    dic = {}
    for i in range(num_class):
        dic[str(i)] = torch.tensor([]).cuda()
    for i, (X, y) in enumerate(train_batches):
        print(i)
        X, y = X.cuda(), y.cuda()
        output, features = model(normalize(X)) # features: 128 x 512
        _, pre_labels = output.max(1) # pre_labels : 128
        c_or_w = (pre_labels == y)
        for j in range(num_class):
            is_j = torch.bitwise_and(c_or_w, (y == j))
            indexs = torch.where(is_j)[0]
            dic[str(j)] = torch.cat((dic[str(j)], features[indexs].detach()), dim=0)        
    mu = torch.zeros(num_class, feature_dim).cuda()
    sigma = torch.zeros(num_class, feature_dim, feature_dim).cuda()
    for i in range(num_class):
        dic_i = dic[str(i)]
        mu[i] = dic_i.mean(dim=0)
        gap = dic_i - mu[i].unsqueeze(dim=0) # 1 x 512
        sigma[i] = (torch.mm(gap.t(), gap) + 1e-10 * torch.eye(feature_dim).cuda()) / dic_i.size(0) # 512 x 512
    print('Finished!')
    return mu, sigma

def GMM_train(model, train_batches, feature_dim=512, dataset='CIFAR-10'):
    print('Crafting GMM parameters on training set')
    if dataset == 'CIFAR-10':
        num_class = 10
    elif dataset == 'CIFAR-100':
        num_class = 100
    dic = {}
    for i in range(num_class):
        dic[str(i)] = torch.tensor([]).cuda()
    for i, (X, y) in enumerate(train_batches):
        print(i)
        X, y = X.cuda(), y.cuda()
        output, features = model(normalize(X)) # features: 128 x 512
        _, pre_labels = output.max(1) # pre_labels : 128
        c_or_w = (pre_labels == y)
        for j in range(num_class):
            is_j = torch.bitwise_and(c_or_w, (y == j))
            indexs = torch.where(is_j)[0]
            dic[str(j)] = torch.cat((dic[str(j)], features[indexs].detach()), dim=0)        
    mu = torch.zeros(num_class, feature_dim).cuda()
    sigma = torch.zeros(num_class, feature_dim, feature_dim).cuda()
    for i in range(num_class):
        dic_i = dic[str(i)]
        mu[i] = dic_i.mean(dim=0)
        gap = dic_i - mu[i].unsqueeze(dim=0) # 1 x 512
        sigma[i] = (torch.mm(gap.t(), gap) + 1e-10 * torch.eye(feature_dim).cuda()) / dic_i.size(0) # 512 x 512
    print('Finished!')
    return mu, sigma

def compute_features_tsne(model, train_batches):
    print('computing features for t-sne visualization on training set')
    saved_fea_correct = torch.tensor([]).cuda()
    saved_Plabels_correct = torch.tensor([]).cuda()
    saved_fea_wrong = torch.tensor([]).cuda()
    saved_Plabels_wrong = torch.tensor([]).cuda()
    saved_Tlabels_wrong = torch.tensor([]).cuda()

    for i, (X, y) in enumerate(train_batches):
        X, y = X.cuda(), y.cuda()
        output, features = model(normalize(X)) # features: 128 x 512
        _, pre_labels = output.max(1) # pre_labels : 128
        features = features.detach()
        c_index = torch.where(pre_labels == y)[0]
        w_index = torch.where(pre_labels != y)[0]
        saved_fea_correct = torch.cat((saved_fea_correct, features[c_index]), dim=0)
        saved_Plabels_correct = torch.cat((saved_Plabels_correct, pre_labels[c_index]), dim=0)
        saved_fea_wrong = torch.cat((saved_fea_wrong, features[w_index]), dim=0)
        saved_Plabels_wrong = torch.cat((saved_Plabels_wrong, pre_labels[w_index]), dim=0)
        saved_Tlabels_wrong = torch.cat((saved_Tlabels_wrong, y[w_index]), dim=0)
    saved_fea_all = torch.cat((saved_fea_correct, saved_fea_wrong), dim=0)
    saved_fea_all = PCA(n_components=30).fit_transform(saved_fea_correct.cpu().numpy())
    print(saved_fea_all.shape)
    saved_fea_all = TSNE(n_components=2, verbose=1, learning_rate=20., perplexity=40., n_iter=3000).fit_transform(saved_fea_all)
    np.savetxt('t-sne/saved_fea_all.txt', saved_fea_all)
    np.savetxt('t-sne/saved_Plabels_correct.txt', saved_Plabels_correct.cpu().numpy())
    np.savetxt('t-sne/saved_Plabels_wrong.txt', saved_Plabels_wrong.cpu().numpy())
    np.savetxt('t-sne/saved_Tlabels_wrong.txt', saved_Tlabels_wrong.cpu().numpy())
    print('Finished!')
    return 0

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
    parser.add_argument('--evalset', default='test', choices=['AutoAttack','test', 'random_rany', 'random_maxy', 'svhn_test', 'adaptiveCWtest'])
    parser.add_argument('--target', action='store_true') # whether use target-mode attack

    # two branch
    parser.add_argument('--twobranch', action='store_true')
    
    # baselines
    parser.add_argument('--baselines', default='KD', choices=['KD', 'LID', 'GDA', 'GMM', 'GDAstar'])
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

    transform_chain = transforms.Compose([transforms.ToTensor()])
    if args.dataset == 'CIFAR-10':
        item_train = datasets.CIFAR10(root=args.data_dir, train=True, transform=transform_chain, download=True)
        item_test = datasets.CIFAR10(root=args.data_dir, train=False, transform=transform_chain, download=True)
        num_cla = 10
        sigma = 1e-3
        K = 600
    elif args.dataset == 'CIFAR-100':
        item_train = datasets.CIFAR100(root=args.data_dir, train=True, transform=transform_chain, download=True)
        item_test = datasets.CIFAR100(root=args.data_dir, train=False, transform=transform_chain, download=True)
        num_cla = 100
        sigma = 1e-2
        K = 20

    # load pretrained model
    if args.model_name == 'PreActResNet18':
        model = PreActResNet18(num_classes=num_cla, return_out=True)
        fea_dim = 512
    elif args.model_name == 'WideResNet':
        model = WideResNet(34, num_cla, widen_factor=10, dropRate=0.0, return_out=True)
        fea_dim = 640
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
      
    train_batches = data.DataLoader(item_train, batch_size=128, shuffle=False, num_workers=4)
    test_batches = data.DataLoader(item_test, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # eval on test set (adversarial examples)
    if True:  
        test_acc, test_robust_acc, test_n = 0, 0, 0
        test_classes_correct, test_classes_wrong = [], []
        test_classes_robust_correct, test_classes_robust_wrong = [], []

        # record con
        test_con_correct, test_robust_con_correct = [], []
        test_con_wrong, test_robust_con_wrong = [], []

        # record evi
        test_evi_correct, test_robust_evi_correct = [], []
        test_evi_wrong, test_robust_evi_wrong = [], []

        # calculate statistics on training set
        if args.baselines == 'KD':
            return_back = Kernel_density_train(model, train_batches, feature_dim=fea_dim, dataset=args.dataset)
        elif args.baselines == 'LID':
            return_back = LID_train(model, train_batches, num=10000, feature_dim=fea_dim, dataset=args.dataset) # num x 512
            return_back = return_back.unsqueeze_(dim=0) # 1 x num x 512
        elif args.baselines == 'GDA':
            mu, sigma = GDA_train(model, train_batches, feature_dim=fea_dim, dataset=args.dataset) # mu: 10 x 512, sigma: 512 x 512
            mu = mu.unsqueeze(dim=0) # 1 x 10 x 512
            sigma = torch.inverse(sigma.unsqueeze(dim=0)) # 1 x 512 x 512
        elif args.baselines == 'GDAstar':
            mu, sigma = GDAstar_train(model, train_batches, feature_dim=fea_dim, dataset=args.dataset) # mu: 10 x 512, sigma: 10 x 512 x 512
            mu = mu.unsqueeze(dim=0) # 1 x 10 x 512
            sigma = torch.inverse(sigma.unsqueeze(dim=0)) # 1 x 10 x 512 x 512
        elif args.baselines == 'GMM':
            mu, sigma = GMM_train(model, train_batches, feature_dim=fea_dim, dataset=args.dataset) # mu: 10 x 512, sigma: 10 x 512 x 512
            mu = mu.unsqueeze(dim=0) # 1 x 10 x 512
            sigma = torch.inverse(sigma.unsqueeze(dim=0)) # 1 x 10 x 512 x 512

        for i, (X, y) in enumerate(test_batches):
            X, y = X.cuda(), y.cuda()

            if args.target:
                y_target = sample_targetlabel(y, num_classes=num_cla)
                delta = attack_pgd(model, X, y_target, epsilon, pgd_alpha, args.attack_iters, args.restarts, args.norm, target=True)
            else:
                delta = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters, args.restarts, args.norm)    
            delta = delta.detach()

            output, features = model(normalize(X))
            robust_output, ro_features = model(normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)))
            features = features.detach()
            ro_features = ro_features.detach()
            
            output_s = F.softmax(output, dim=1)
            out_con, out_pre = output_s.max(1)

            ro_output_s = F.softmax(robust_output, dim=1)
            ro_out_con, ro_out_pre = ro_output_s.max(1)

            mm = torch.matmul
            bs = torch.tensor(range(X.size(0)))

            if args.baselines == 'KD':
                ref_vectors = torch.index_select(return_back, 0, out_pre) # 128 x 1000 x 512
                ro_ref_vectors = torch.index_select(return_back, 0, ro_out_pre) # 128 x 1000 x 512
                gap = ref_vectors - features.unsqueeze(dim=1)
                ro_gap = ro_ref_vectors - ro_features.unsqueeze(dim=1)
                test_evi_all = torch.exp(- torch.pow(torch.norm(gap, p=2, dim=2), 2) * sigma) # 128 x 1000
                test_robust_evi_all = torch.exp(- torch.pow(torch.norm(ro_gap, p=2, dim=2), 2) * sigma) # 128 x 1000
                test_evi_all = test_evi_all.mean(dim=1)
                test_robust_evi_all = test_robust_evi_all.mean(dim=1)

            elif args.baselines == 'LID':
                gap = torch.norm(return_back - features.unsqueeze(dim=1), p=2, dim=2) # 128 x num
                ro_gap = torch.norm(return_back - ro_features.unsqueeze(dim=1), p=2, dim=2) # 128 x num
                top_K = torch.log(torch.sort(gap, dim=1)[0][:, :K]) # 128 x K
                ro_top_K = torch.log(torch.sort(ro_gap, dim=1)[0][:, :K]) # 128 x K
                test_evi_all = 1. / (top_K.mean(dim=1) - top_K[:, -1])
                test_robust_evi_all = 1. / (ro_top_K.mean(dim=1) - ro_top_K[:, -1])

            elif args.baselines == 'GDA':
                mean_v = features.unsqueeze(dim=1) - mu # 128 x 10 x 512
                ro_mean_v = ro_features.unsqueeze(dim=1) - mu # 128 x 10 x 512
                score_v = - torch.diagonal(mm(mm(mean_v, sigma), mean_v.transpose(-2, -1)), dim1=-2, dim2=-1) # 128 x 10
                ro_score_v = - torch.diagonal(mm(mm(ro_mean_v, sigma), ro_mean_v.transpose(-2, -1)), dim1=-2, dim2=-1) # 128 x 10
                test_evi_all = score_v.max(1)[0]
                test_robust_evi_all = ro_score_v.max(1)[0]

            elif args.baselines == 'GDAstar':
                mean_v = (features.unsqueeze(dim=1) - mu).unsqueeze(dim=2) # 128 x 10 x 1 x 512
                ro_mean_v = (ro_features.unsqueeze(dim=1) - mu).unsqueeze(dim=2) # 128 x 10 x 1 x 512
                score_v = - mm(mm(mean_v, sigma), mean_v.transpose(-2, -1)) # 128 x 10 x 1 x 1 
                ro_score_v = - mm(mm(ro_mean_v, sigma), ro_mean_v.transpose(-2, -1)) # 128 x 10 x 1 x 1
                test_evi_all = score_v.squeeze().max(1)[0]
                test_robust_evi_all = ro_score_v.squeeze().max(1)[0]

            elif args.baselines == 'GMM':
                SIG = sigma.expand(X.size(0), -1, -1, -1) # 128 x 10 x 512 x 512
                mean_v = features.unsqueeze(dim=1) - mu # 128 x 10 x 512
                mean_v = mean_v[bs, out_pre, :].unsqueeze(dim=1) # 128 x 1 x 512
                covariance = SIG[bs, out_pre, :, :] # 128 x 512 x 512
                ro_mean_v = ro_features.unsqueeze(dim=1) - mu # 128 x 10 x 512
                ro_mean_v = ro_mean_v[bs, ro_out_pre, :].unsqueeze(dim=1) # 128 x 1 x 512
                ro_covariance = SIG[bs, ro_out_pre, :, :] # 128 x 512 x 512
                score_v = - mm(mm(mean_v, covariance), mean_v.transpose(-2, -1)) # 128 x 10
                ro_score_v = - mm(mm(ro_mean_v, ro_covariance), ro_mean_v.transpose(-2, -1)) # 128 x 10
                test_evi_all = score_v.squeeze()
                test_robust_evi_all = ro_score_v.squeeze()
        
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
        FPR_thred, acc = calculate_FPR_TPR(test_con_correct, test_con_wrong)
        robust_robust = calculate_auc_scores(test_robust_con_correct, test_robust_con_wrong)
        ro_FPR_thred, ro_acc = calculate_FPR_TPR(test_robust_con_correct, test_robust_con_wrong)
        logger.info('clean_clean: %.3f | robust_robust: %.3f', 
            clean_clean, robust_robust)
        logger.info('TPR 95 clean acc improve: %.4f | TPR 95 robust acc improve: %.4f', 
            acc - test_acc, ro_acc - test_robust_acc)
               
        print('')
        print('### ROC-AUC scores (evidence) ###')
        clean_clean = calculate_auc_scores(test_evi_correct, test_evi_wrong)
        FPR_thred, acc = calculate_FPR_TPR(test_evi_correct, test_evi_wrong)
        robust_robust = calculate_auc_scores(test_robust_evi_correct, test_robust_evi_wrong)
        ro_FPR_thred, ro_acc = calculate_FPR_TPR(test_robust_evi_correct, test_robust_evi_wrong)
        logger.info('clean_clean: %.3f | robust_robust: %.3f', 
            clean_clean, robust_robust)
        logger.info('TPR 95 clean acc improve: %.4f | TPR 95 robust acc improve: %.4f', 
            acc - test_acc, ro_acc - test_robust_acc)

if __name__ == "__main__":
    main()
