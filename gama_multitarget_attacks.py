from utils import *
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
from torch.utils.data.sampler import SubsetRandomSampler
import time

CE_loss = nn.CrossEntropyLoss()

def max_margin_loss(x, y, num_cla=10):
    B = y.size(0)
    corr = x[range(B),y]
    x_new = x - 1000 * torch.eye(num_cla)[y].cuda()
    tar = x[range(B),x_new.argmax(dim=1)]
    loss = tar - corr
    loss = torch.mean(loss)
    return loss

def GAMA_FW(model, data, target, eps, gamma=0.5, steps=100, SCHED=[60,85], drop=5, w_reg=50, lin=25, twobranch=False):
    tar = Variable(target.cuda())
    data = data.cuda()
    B,C,H,W = data.size()
    delta  = torch.rand_like(data).cuda()
    delta = eps*torch.sign(delta-0.5)
    delta.requires_grad = True
    orig_img = data + delta
    orig_img = Variable(orig_img,requires_grad=False) 
    WREG = w_reg
    for step in range(steps): 
        if step in SCHED:
            gamma /= drop 
        delta = Variable(delta,requires_grad=True)
        # make gradient of img to zeros
        zero_gradients(delta) 
        if step<lin:            
            out_all = model(normalize(torch.cat((orig_img,data+delta),0)))
            out_all = out_all[0] if twobranch else out_all
            P_out = nn.Softmax(dim=1)(out_all[:B,:])
            Q_out = nn.Softmax(dim=1)(out_all[B:,:])
            cost =  max_margin_loss(Q_out,tar) + WREG*((Q_out - P_out)**2).sum(1).mean(0)     
            WREG -= w_reg/lin
        else:
            out  = model(normalize(data+delta))
            out = out[0] if twobranch else out
            Q_out = nn.Softmax(dim=1)(out)
            cost =  max_margin_loss(Q_out,tar)    
        cost.backward()
        delta.grad =  torch.sign(delta.grad)*eps
        delta = (1-gamma)*delta + gamma*delta.grad
        delta = (data+delta).clamp(0.0,1.0) - data
        delta.data.clamp_(-eps, eps)
    return data + delta.detach()
    

def GAMA_PGD(model, data, target, eps, eps_iter, bounds=np.array([[0,1],[0,1],[0,1]]), steps=100, 
    w_reg=50, lin=25, SCHED=[60,85], drop=10, twobranch=False):
    #Raise error if in training mode
    if model.training:
        assert 'Model is in training mode'
    tar = Variable(target.cuda())
    data = data.cuda()
    B,C,H,W = data.size()
    noise  = torch.FloatTensor(np.random.uniform(-eps,eps,(B,C,H,W))).cuda()
    noise  = eps*torch.sign(noise)
    img_arr = []
    W_REG = w_reg
    orig_img = data+noise
    orig_img = Variable(orig_img,requires_grad=True)
    for step in range(steps):
        # convert data and corresponding into cuda variable
        img = data + noise
        img = Variable(img,requires_grad=True)
        if step in SCHED:
            eps_iter /= drop
        # make gradient of img to zeros
        zero_gradients(img) 
        # forward pass        
        orig_out = model(normalize(orig_img))
        orig_out = orig_out[0] if twobranch else orig_out
        P_out = nn.Softmax(dim=1)(orig_out)
        out  = model(normalize(img))
        out = out[0] if twobranch else out
        Q_out = nn.Softmax(dim=1)(out)
        #compute loss using true label
        if step <= lin:
            cost =  W_REG*((P_out - Q_out)**2.0).sum(1).mean(0) + max_margin_loss(Q_out,tar)
            W_REG -= w_reg/lin
        else:
            cost = max_margin_loss(Q_out,tar)
        #backward pass
        cost.backward()
        #get gradient of loss wrt data
        per =  torch.sign(img.grad.data)
        #convert eps 0-1 range to per channel range 
        per[:,0,:,:] = (eps_iter * (bounds[0,1] - bounds[0,0])) * per[:,0,:,:]
        if(per.size(1)>1):
            per[:,1,:,:] = (eps_iter * (bounds[1,1] - bounds[1,0])) * per[:,1,:,:]
            per[:,2,:,:] = (eps_iter * (bounds[2,1] - bounds[2,0])) * per[:,2,:,:]
        #  ascent
        adv = img.data + per.cuda()
        #clip per channel data out of the range
        img.requires_grad =False
        img[:,0,:,:] = torch.clamp(adv[:,0,:,:],bounds[0,0],bounds[0,1])
        if(per.size(1)>1):
            img[:,1,:,:] = torch.clamp(adv[:,1,:,:],bounds[1,0],bounds[1,1])
            img[:,2,:,:] = torch.clamp(adv[:,2,:,:],bounds[2,0],bounds[2,1])
        img = img.data
        noise = img - data
        noise  = torch.clamp(noise,-eps,eps)
    return data + noise.delta()


### Multi-target attack
def multitarget_loss(x, y, index=0, num_cla=10):# index =0, 1, ..., num_cla-2
    y_k = index * torch.ones_like(y).cuda()
    y_k[y_k >= y] = y_k[y_k >= y] + 1
    loss_value = - x[np.arange(x.shape[0]), y] + x[np.arange(x.shape[0]), y_k]
    return loss_value.mean()


def multitarget_attack(model, X, y, epsilon, alpha, attack_iters, restarts=20, norm='l_inf',
                twobranch=False, num_cla=10):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    norm_func = normalize
    for _ in range(restarts):
        for ind in range(num_cla - 1):
            delta = torch.zeros_like(X).uniform_(-epsilon, epsilon).cuda()
            delta = clamp(delta, lower_limit-X, upper_limit-X)
            delta.requires_grad = True
            for _ in range(attack_iters):
                output = model(normalize(X + delta))[0] if twobranch else model(normalize(X + delta))
                loss = multitarget_loss(output, y, index=ind, num_cla=num_cla)
                grad = torch.autograd.grad(loss, delta)[0]
                d = torch.clamp(delta + alpha * torch.sign(grad), min=-epsilon, max=epsilon)
                d = clamp(d, lower_limit - X, upper_limit - X)
                delta.data = d
            output = model(normalize(X + delta))[0] if twobranch else model(normalize(X + delta))
            all_loss = F.cross_entropy(output, y, reduction='none')
            max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
            max_loss = torch.max(max_loss, all_loss)
    return X + max_delta        