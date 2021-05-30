import argparse
import logging
import sys
import time
import math
from utils import *
from models import *
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from ATRO_loss import MaxHingeLossWithRejection, WeightPenalty
import os

criterion_kl = nn.KLDivLoss(reduction='batchmean')

# def normalize(X):
#     return X

def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, norm,
                adaptive_evidence=False, adaptive_lambda=1., uniform_lambda=False, BNeval=False,
                twobranch=False, twosign=False):
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
        # uniform sampling for adaptive lambda
        if uniform_lambda:
            if twosign:
                a_lambda = torch.zeros(y.shape[0]).uniform_(- adaptive_lambda,adaptive_lambda).cuda()
            else:
                a_lambda = torch.zeros(y.shape[0]).uniform_(0.,adaptive_lambda).cuda()
        else:
            a_lambda = adaptive_lambda

        for _ in range(attack_iters):
            if twobranch:
                output, output_evi = model(normalize(X + delta))
                evi = output_evi.logsumexp(dim=1)
            else:
                output = model(normalize(X + delta))
                evi = output.logsumexp(dim=1)
            loss = F.cross_entropy(output, y)
            # if apply adaptive attacks for the evidence detection
            if adaptive_evidence:
                loss += (a_lambda * evi).mean()
            grad = torch.autograd.grad(loss, delta)[0]
            if norm == "l_inf":
                d = torch.clamp(delta + alpha * torch.sign(grad), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(grad.view(grad.shape[0],-1),dim=1).view(-1,1,1,1)
                scaled_g = grad/(g_norm + 1e-10)
                d = (delta + scaled_g*alpha).view(delta.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(delta)
            d = clamp(d, lower_limit - X, upper_limit - X)
            delta.data = d
        if twobranch:
            all_loss = F.cross_entropy(model(normalize(X+delta))[0], y, reduction='none')
        else:
            all_loss = F.cross_entropy(model(normalize(X+delta)), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    if BNeval:
        model.train()
    return max_delta, a_lambda 


def attack_trades(model, X, y, epsilon, alpha, attack_iters, restarts, norm, BNeval=True, twobranch=False):
    model.eval()
    clean_output = model(normalize(X))[0] if twobranch else model(normalize(X))
    clean_output = F.softmax(clean_output.detach(), dim=1)
    #delta = torch.zeros_like(X).uniform_(-epsilon, epsilon).cuda()
    delta = 0.001 * torch.randn(X.shape).cuda().detach()
    delta = clamp(delta, lower_limit-X, upper_limit-X)
    delta.requires_grad = True
    for _ in range(attack_iters):
        output = model(normalize(X + delta))[0] if twobranch else model(normalize(X + delta))
        loss = criterion_kl(F.log_softmax(output, dim=1), clean_output)
        grad = torch.autograd.grad(loss, delta)[0]
        if norm == "l_inf":
            d = torch.clamp(delta + alpha * torch.sign(grad), min=-epsilon, max=epsilon)
        # elif norm == "l_2":
        #     g_norm = torch.norm(grad.view(grad.shape[0],-1),dim=1).view(-1,1,1,1)
        #     scaled_g = grad/(g_norm + 1e-10)
        #     d = (delta + scaled_g*alpha).view(delta.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(delta)
        delta.data = clamp(d, lower_limit - X, upper_limit - X)
    model.train()
    return delta.detach()

def attack_ATRO(MHRLoss, num_cla, model, X, y, epsilon, alpha, attack_iters, restarts, norm, BNeval=True, twobranch=True):
    model.eval()
    delta = torch.zeros_like(X).uniform_(-epsilon, epsilon).cuda()
    delta = clamp(delta, lower_limit-X, upper_limit-X)
    delta.requires_grad = True
    for _ in range(attack_iters):
        output, output_aux = model(normalize(X + delta))
        loss,_ = MHRLoss(F.softmax(output, dim=1), output_aux.tanh(), y, num_cla)
        #loss,_ = MHRLoss(output, output_aux.tanh(), y, num_cla)
        grad = torch.autograd.grad(loss, delta)[0]
        if norm == "l_inf":
            d = torch.clamp(delta + alpha * torch.sign(grad), min=-epsilon, max=epsilon)
        delta.data = clamp(d, lower_limit - X, upper_limit - X)
    model.train()
    return delta.detach()

def attack_CARL(model, X, y, epsilon, alpha, attack_iters, restarts, norm, BNeval=False, twobranch=True):
    if BNeval:
        model.eval()
    delta = torch.zeros_like(X).uniform_(-epsilon, epsilon).cuda()
    delta = clamp(delta, lower_limit-X, upper_limit-X)
    delta.requires_grad = True
    for _ in range(attack_iters):
        output, output_aux = model(normalize(X + delta))
        output_all = torch.cat((output, output_aux), dim=1) # bs x 11 or bs x 101
        softmax_output = F.softmax(output_all, dim=1)
        so_y = softmax_output[torch.tensor(range(X.size(0))), y]
        so_a = softmax_output[torch.tensor(range(X.size(0))), -1]
        loss = - torch.log(so_y + so_a)
        grad = torch.autograd.grad(loss.mean(), delta)[0]
        if norm == "l_inf":
            d = torch.clamp(delta + alpha * torch.sign(grad), min=-epsilon, max=epsilon)
        delta.data = clamp(d, lower_limit - X, upper_limit - X)
    if BNeval:
        model.train()
    return delta.detach()

def attack_ccat(model, X, y, epsilon, alpha, attack_iters, restarts, norm, BNeval=False, twobranch=False, 
                    beta=0.9, lr_decay=1.5):
    if BNeval:
        model.eval()
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    ber = torch.distributions.bernoulli.Bernoulli(0.5)

    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda()
        ber_samples = ber.sample(torch.Size([y.shape[0]]))
        if norm == "l_inf":
            #delta.uniform_(-epsilon, epsilon)
            d = delta[ber_samples > 0]
            d.normal_()
            u = torch.zeros(d.size(0)).uniform_(0, 1).cuda()
            linf_norm = u / torch.max(d.abs().view(d.size(0),-1), dim=1)[0]
            d = epsilon * d * linf_norm.view(d.size(0), 1, 1, 1)
            delta[ber_samples > 0] = d
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
        alpha_batch = alpha * torch.ones(y.size(0), 1, 1, 1).half().cuda()
        momentum_grad = 0
        best_loss = torch.zeros(y.size(0)).cuda()
        for ai in range(attack_iters):
            output = model(normalize(X + delta))[0] if twobranch else model(normalize(X + delta))

            # choose the max labels except for the true ones
            softmax_output = F.softmax(output, dim=1)
            softmax_output[torch.arange(X.size(0)), y] = -1
            y_max = torch.max(softmax_output, dim=1)[1].detach()
            loss = - F.cross_entropy(output, y_max)
            grad = torch.autograd.grad(loss, delta)[0]            
            if norm == "l_inf":
                # momentum_grad = torch.sign(grad) if ai == 0 else beta * momentum_grad + (1 - beta) * torch.sign(grad)
                momentum_grad = beta * momentum_grad + (1 - beta) * torch.sign(grad)
                d = torch.clamp(delta + alpha_batch * momentum_grad, min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(grad.view(grad.shape[0],-1),dim=1).view(-1,1,1,1)
                scaled_g = grad/(g_norm + 1e-10)
                d = (delta + scaled_g*alpha_batch).view(delta.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(delta)

            #backtrack
            d = clamp(d, lower_limit - X, upper_limit - X)
            output = model(normalize(X + d))[0] if twobranch else model(normalize(X + d))
            loss_d = F.cross_entropy(output.detach(), y, reduction='none')
            alpha_batch[loss_d <= best_loss] = alpha_batch[loss_d <= best_loss] / lr_decay
            delta.data[loss_d >= best_loss] = d[loss_d >= best_loss]
            best_loss[loss_d >= best_loss] = loss_d[loss_d >= best_loss]

        if twobranch:
            all_loss = F.cross_entropy(model(normalize(X+delta))[0], y, reduction='none')
        else:
            all_loss = F.cross_entropy(model(normalize(X+delta)), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    if BNeval:
        model.train()
    return max_delta

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='PreActResNet18')
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--dataset', default='CIFAR-10', type=str)
    parser.add_argument('--data-dir', default='../cifar-data', type=str)
    parser.add_argument('--epochs', default=110, type=int)
    parser.add_argument('--lr-max', default=0.1, type=float)
    parser.add_argument('--lr-schedule', default='piecewise', type=str)
    parser.add_argument('--attack', default='pgd', type=str, choices=['pgd', 'free', 'none'])
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--attack-iters', default=10, type=int)
    parser.add_argument('--restarts', default=1, type=int)
    parser.add_argument('--pgd-alpha', default=2, type=float)
    parser.add_argument('--fgsm-alpha', default=1.25, type=float)
    parser.add_argument('--norm', default='l_inf', type=str, choices=['l_inf', 'l_2'])
    parser.add_argument('--fname', default='cifar_model', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--weight_decay', default=5e-4, type=float)#weight decay
    parser.add_argument('--optimizer', default='SGD', type=str)
    parser.add_argument('--target', action='store_true') # whether use target-mode attack

    parser.add_argument('--ATframework', default='PGDAT', type=str, choices=['PGDAT', 'TRADES', 'CCAT'])
    parser.add_argument('--TRADESlambda', default=1., type=float)
    parser.add_argument('--CCATiter', default=20, type=int)
    parser.add_argument('--CCATrho', default=1, type=int)
    parser.add_argument('--CCATstep', default=1., type=float)
    parser.add_argument('--CCATratio', default=1., type=float)
    parser.add_argument('--CCATscale', default=1., type=float)

    ### adaptive attack
    parser.add_argument('--adaptiveattack', action='store_true') # whether use adaptive term in the attacks
    parser.add_argument('--adaptiveattacklambda', default=1., type=float)
    parser.add_argument('--uniform_lambda', action='store_true') # whether use uniform distribution for lambda in adaptive attack
    parser.add_argument('--BNeval', action='store_true') # whether use eval mode for BN when crafting adversarial examples
    parser.add_argument('--twosign', action='store_true')

    ### adaptive training
    parser.add_argument('--adaptivetrain', action='store_true') # whether use adaptive term in train
    parser.add_argument('--adaptivetrainlambda', default=1., type=float)

    parser.add_argument('--selfreweightCalibrate', action='store_true') # Calibrate
    parser.add_argument('--temp', default=1., type=float)
    parser.add_argument('--tempC', default=1., type=float)
    parser.add_argument('--tempC_trueonly', default=1., type=float) # stop gradient for the confidence term    
    parser.add_argument('--SGconfidenceW', action='store_true') # stop gradient for the confidence term

    parser.add_argument('--ConfidenceOnly', action='store_true')
    parser.add_argument('--AuxiliaryOnly', action='store_true')

    # two branch for our selfreweightCalibrate (rectified rejection)
    parser.add_argument('--twobranch', action='store_true')
    parser.add_argument('--out_dim', default=10, type=int)
    parser.add_argument('--useBN', action='store_true')
    parser.add_argument('--along', action='store_true')

    ### EBD baseline
    parser.add_argument('--selfreweightNIPS20', action='store_true') # Energy-based Out-of-distribution Detection
    parser.add_argument('--m_in', default=6, type=float)
    parser.add_argument('--m_out', default=3, type=float)

    ### ATRO baseline
    parser.add_argument('--selfreweightATRO', action='store_true') # ATRO https://github.com/MasaKat0/ATRO
    parser.add_argument('--ATRO_cost', default=0.3, type=float)
    parser.add_argument('--ATRO_coefficient', default=0.3, type=float)

    ### CARL baseline
    parser.add_argument('--selfreweightCARL', action='store_true') # CARL https://github.com/cassidylaidlaw/playing-it-safe
    parser.add_argument('--CARL_lambda', default=0.5, type=float)
    parser.add_argument('--CARL_eta', default=0.02, type=float)
    
    return parser.parse_args()

def main():
    args = get_args()
    epsilon = (args.epsilon / 255.)
    pgd_alpha = (args.pgd_alpha / 255.)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    torch.backends.cudnn.benchmark=True 

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
            logging.FileHandler(os.path.join(args.fname, 'output.log')),
            logging.StreamHandler()
        ])

    logger.info(args)


    # Prepare dataset
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()])
    transform_test = transforms.Compose([transforms.ToTensor()])

    if args.dataset == 'CIFAR-10':
        trainset = torchvision.datasets.CIFAR10(root='../cifar-data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='../cifar-data', train=False, download=True, transform=transform_test)
        num_cla = 10
    elif args.dataset == 'CIFAR-100':
        trainset = torchvision.datasets.CIFAR100(root='../cifar-data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='../cifar-data', train=False, download=True, transform=transform_test)
        num_cla = 100

    train_batches = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_batches = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    if args.selfreweightCalibrate or args.selfreweightATRO or args.selfreweightCARL:
        along = True
        args.out_dim = 1
        
    # Creat model
    if args.model_name == 'PreActResNet18':
        model = PreActResNet18(num_classes=num_cla)
    elif args.model_name == 'PreActResNet18_twobranch_DenseV1':
        model = PreActResNet18_twobranch_DenseV1(num_classes=num_cla, out_dim=args.out_dim, use_BN=args.useBN, along=along)
    elif args.model_name == 'PreActResNet18_twobranch_DenseV1Multi':
        model = PreActResNet18_twobranch_DenseV1Multi(num_classes=num_cla, out_dim=args.out_dim, use_BN=args.useBN, along=along)
    elif args.model_name == 'PreActResNet18_twobranch_DenseV2':
        model = PreActResNet18_twobranch_DenseV2(num_classes=num_cla, out_dim=args.out_dim, use_BN=args.useBN, along=along)
    elif args.model_name == 'WideResNet':
        model = WideResNet(34, num_cla, widen_factor=10, dropRate=0.0)
    elif args.model_name == 'WideResNet_twobranch_DenseV1':
        model = WideResNet_twobranch_DenseV1(34, num_cla, widen_factor=10, dropRate=0.0, use_BN=args.useBN, along=along, out_dim=args.out_dim)
    elif args.model_name == 'WideResNet_20':
        model = WideResNet(34, num_cla, widen_factor=20, dropRate=0.0)
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

    def lr_schedule(t):
        if t < 100:
            return args.lr_max
        elif t < 105:
            return args.lr_max / 10.
        else:
            return args.lr_max / 100.

    best_test_robust_acc, best_val_robust_acc, start_epoch = 0, 0, 0

    criterion = nn.CrossEntropyLoss()
    criterion_none = nn.CrossEntropyLoss(reduction='none')
    BCEcriterion = nn.BCELoss(reduction='none')
    MSEcriterion = nn.MSELoss()
    MHRLoss = MaxHingeLossWithRejection(args.ATRO_cost)

    # logger.info('Epoch \t Acc \t Robust Acc \t Evi \t Robust Evi')
    logger.info('Epoch \t Acc \t Robust Acc')
    for epoch in range(start_epoch, epochs):
        model.train()
        start_time = time.time()
        for i, (data, target) in enumerate(train_batches):
            X, y = data.cuda(), target.cuda()
            epoch_now = epoch + (i + 1) / len(train_batches)
            lr = lr_schedule(epoch_now)
            opt.param_groups[0].update(lr=lr)

            if args.selfreweightATRO:
                delta = attack_ATRO(MHRLoss, num_cla, model, X, y, epsilon, pgd_alpha, args.attack_iters, args.restarts, args.norm, 
                    BNeval=True, twobranch=True)
            elif args.selfreweightCARL:
                delta = attack_CARL(model, X, y, epsilon, pgd_alpha, args.attack_iters, args.restarts, args.norm, 
                    BNeval=args.BNeval, twobranch=True)
            elif args.ATframework == 'TRADES':
                delta = attack_trades(model, X, y, epsilon, pgd_alpha, args.attack_iters, args.restarts, args.norm, 
                    BNeval=True, twobranch=args.twobranch)
                delta = delta.detach()
            elif args.ATframework == 'PGDAT':
                delta, adaptive_l = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters, args.restarts, args.norm,
                 adaptive_evidence=args.adaptiveattack, adaptive_lambda=args.adaptiveattacklambda,
                 uniform_lambda=args.uniform_lambda, BNeval=args.BNeval, twobranch=args.twobranch, twosign=args.twosign)
                delta = delta.detach()
            elif args.ATframework == 'CCAT':
                output = model(normalize(X))[0] if args.twobranch else model(normalize(X))
                delta = attack_ccat(model, X, y, epsilon, args.CCATstep / 255., args.CCATiter, args.restarts, args.norm, 
                    BNeval=args.BNeval, twobranch=args.twobranch)
                delta = delta.detach()
            # Standard training
            elif args.attack == 'none':
                delta = torch.zeros_like(X)


            # whether use two branches
            if args.twobranch:
                robust_output, robust_output_aux = model(normalize(torch.clamp(X + delta, min=lower_limit, max=upper_limit)))
            else:
                robust_output = model(normalize(torch.clamp(X + delta, min=lower_limit, max=upper_limit)))


            # choose between PGDAT, CCAT and TRADES
            if args.ATframework == 'PGDAT':
                robust_loss = criterion(robust_output, y)
            elif args.ATframework == 'TRADES':
                output = model(normalize(X))[0] if args.twobranch else model(normalize(X))
                KL_term = criterion_kl(F.log_softmax(robust_output, dim=1), F.softmax(output, dim=1))
                robust_loss = criterion(output, y) + args.TRADESlambda * KL_term
            elif args.ATframework == 'CCAT':
                sl = 1 - torch.max(delta.detach().view(y.size(0), -1).abs(), dim=1, keepdim=True)[0] / (args.CCATscale * epsilon)
                sl = torch.pow(sl, args.CCATrho) # bs x num_cla
                #print(sl)
                smoothed_label = sl * F.one_hot(y, num_classes=num_cla) + (1 - sl) / num_cla
                robust_loss = criterion(output, y) + args.CCATratio * criterion_kl(F.log_softmax(robust_output, dim=1), smoothed_label.float())
                

            if args.adaptivetrain:
                if args.selfreweightCalibrate:
                    robust_output_s = torch.softmax(robust_output * args.tempC, dim=1)
                    robust_con_pre, robust_con_label = robust_output_s.max(1) # predicted label and confidence
                    robust_output_s_ = torch.softmax(robust_output * args.tempC_trueonly, dim=1)
                    robust_con_y = robust_output_s_[torch.tensor(range(X.size(0))), y].detach() # predicted prob on the true label y
                    
                    if args.SGconfidenceW:
                        correct_index = torch.where(robust_output.max(1)[1] == y)[0]
                        robust_con_pre[correct_index] = robust_con_pre[correct_index].detach()

                    robust_output_aux = robust_output_aux.sigmoid().squeeze() # bs, Calibration function A \in [0,1]
                    robust_detector = robust_con_pre * robust_output_aux
                    
                    ### ConfidenceOnly and AuxiliaryOnly are used for ablation studies
                    if args.ConfidenceOnly:
                        robust_detector = robust_con_pre
                    if args.AuxiliaryOnly:
                        robust_detector = robust_output_aux

                    aux_loss = BCEcriterion(robust_detector, robust_con_y)
                    robust_loss += args.adaptivetrainlambda * aux_loss.mean(dim=0)

                elif args.selfreweightNIPS20:
                    wrong_index = torch.where(robust_output.max(1)[1] != y)[0]
                    correct_index = torch.where(robust_output.max(1)[1] == y)[0]
                    logp_robust_all = robust_output.logsumexp(dim=1)
                    if wrong_index.size(0) > 0 and correct_index.size(0) > 0:  
                        logp_robust_wrong = logp_robust_all[wrong_index]
                        logp_robust_correct = logp_robust_all[correct_index]
                        L_en = torch.pow(F.relu(logp_robust_wrong - args.m_out), 2).mean() \
                                + torch.pow(F.relu(args.m_in - logp_robust_correct), 2).mean()
                        robust_loss += args.adaptivetrainlambda * L_en

                elif args.selfreweightATRO:
                    robust_output_aux = robust_output_aux.tanh() # -1 to 1
                    robust_loss += args.ATRO_coefficient * MHRLoss(F.softmax(robust_output, dim=1), robust_output_aux, y, num_cla)[0]
                    #robust_loss += args.ATRO_coefficient * MHRLoss(robust_output, robust_output_aux, y, num_cla)[0]
                    #robust_loss += 0.5 * WeightPenalty()(model)

                elif args.selfreweightCARL:
                    ro_output_all = torch.cat((robust_output, robust_output_aux), dim=1) # bs x 11 or bs x 101
                    ro_softmax_output = F.softmax(ro_output_all, dim=1)
                    ro_so_y = ro_softmax_output[torch.tensor(range(X.size(0))), y]
                    ro_so_a = ro_softmax_output[torch.tensor(range(X.size(0))), -1]
                    l1_loss = - torch.log(ro_so_y + ro_so_a)
                    X.requires_grad_(True)
                    output = model(normalize(X))[0] if args.twobranch else model(normalize(X))
                    grad_CE = torch.autograd.grad(CW_loss(output, y, SUM=True), X, create_graph=True)[0]
                    grad_norm = torch.norm(grad_CE.view(X.size(0), -1), p=1, dim=1)
                    # robust_loss = criterion(output, y) + args.CARL_lambda * l1_loss.mean() + args.CARL_eta * grad_norm.mean()
                    robust_loss += args.CARL_lambda * l1_loss.mean() + args.CARL_eta * grad_norm.mean()


            opt.zero_grad()
            robust_loss.backward()
            opt.step()


        model.eval()
        test_acc = 0
        test_robust_acc = 0
        test_evi_correct = 0
        test_robust_evi_correct = 0
        test_evi_wrong = 0
        test_robust_evi_wrong = 0
        test_n = 0
        for i, (data, target) in enumerate(test_batches):
            X, y = data.cuda(), target.cuda()

            # Random initialization
            delta, _ = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters, args.restarts, args.norm, twobranch=args.twobranch)
            delta = delta.detach()

            if args.twobranch:
                output, output_aux = model(normalize(X))
                robust_output, robust_output_aux = model(normalize(torch.clamp(X + delta, min=lower_limit, max=upper_limit)))

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

                elif args.selfreweightATRO:
                    test_evi_all = output_aux.tanh().squeeze()
                    test_robust_evi_all = robust_output_aux.tanh().squeeze() # bs x 1, Calibration function A \in [0,1]

                elif args.selfreweightCARL:
                    output_all = torch.cat((output, output_aux), dim=1) # bs x 11 or bs x 101
                    ro_output_all = torch.cat((robust_output, robust_output_aux), dim=1) # bs x 11 or bs x 101
                    softmax_output = F.softmax(output_all, dim=1)
                    ro_softmax_output = F.softmax(ro_output_all, dim=1)
                    test_evi_all = softmax_output[torch.tensor(range(X.size(0))), -1]
                    test_robust_evi_all = ro_softmax_output[torch.tensor(range(X.size(0))), -1]

                   
            else:
                output = model(normalize(X))
                robust_output = model(normalize(torch.clamp(X + delta, min=lower_limit, max=upper_limit)))
                test_evi_all = output.logsumexp(dim=1)
                test_robust_evi_all = robust_output.logsumexp(dim=1)


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


        # logger.info('%d \t %.4f \t %.4f \t (%.4f / %.4f) \t (%.4f / %.4f)', epoch, test_acc/test_n, test_robust_acc/test_n,
        #     test_evi_correct/test_acc, test_evi_wrong/(test_n-test_acc),
        #     test_robust_evi_correct/test_robust_acc, test_robust_evi_wrong/(test_n-test_robust_acc))

        logger.info('%d \t %.4f \t %.4f', epoch, test_acc/test_n, test_robust_acc/test_n)

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

        for i, (data, target) in enumerate(test_batches):
            X, y = data.cuda(), target.cuda()

            if args.target:
                y_target = sample_targetlabel(y, num_classes=num_cla)
                delta,_ = attack_pgd(model, X, y_target, epsilon, pgd_alpha, args.attack_iters, args.restarts, args.norm, target=True, twobranch=args.twobranch)
            else:
                delta,_ = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters, args.restarts, args.norm, twobranch=args.twobranch)
            
            delta = delta.detach()

            if args.twobranch:
                output, output_aux = model(normalize(X))
                robust_output, robust_output_aux = model(normalize(torch.clamp(X + delta, min=lower_limit, max=upper_limit)))

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

                elif args.selfreweightATRO:
                    test_evi_all = output_aux.tanh().squeeze()
                    test_robust_evi_all = robust_output_aux.tanh().squeeze() # bs x 1, Calibration function A \in [0,1]

                elif args.selfreweightCARL:
                    output_all = torch.cat((output, output_aux), dim=1) # bs x 11 or bs x 101
                    ro_output_all = torch.cat((robust_output, robust_output_aux), dim=1) # bs x 11 or bs x 101
                    softmax_output = F.softmax(output_all, dim=1)
                    ro_softmax_output = F.softmax(ro_output_all, dim=1)
                    test_evi_all = softmax_output[torch.tensor(range(X.size(0))), -1]
                    test_robust_evi_all = ro_softmax_output[torch.tensor(range(X.size(0))), -1]

            else:
                output = model(normalize(X))
                robust_output = model(normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)))
                test_evi_all = output.logsumexp(dim=1)
                test_robust_evi_all = robust_output.logsumexp(dim=1)

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
        # clean_clean = calculate_auc_scores(test_con_correct, test_con_wrong)
        # robust_robust = calculate_auc_scores(test_robust_con_correct, test_robust_con_wrong)
        # logger.info('clean_clean: %.3f | robust_robust: %.3f', 
        #     clean_clean, robust_robust)
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
        

        
        print('')
        print('### ROC-AUC scores (evidence) ###')
        # clean_clean = calculate_auc_scores(test_evi_correct, test_evi_wrong)
        # robust_robust = calculate_auc_scores(test_robust_evi_correct, test_robust_evi_wrong)
        # logger.info('clean_clean: %.3f | robust_robust: %.3f', 
        #     clean_clean, robust_robust)
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
        
        


if __name__ == "__main__":
    main()
