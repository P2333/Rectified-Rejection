import torch
from collections import OrderedDict

class WeightPenalty(torch.nn.Module):
    def __init__(self, norm='l2'):
        super(WeightPenalty, self).__init__()
        self.norm = norm

    def forward(self, model):
        
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                w = m.weight
                if self.norm == 'l2':
                    weight_loss = torch.norm(w)**2
                    break
                else:
                    raise NotImplementedError

        return weight_loss

class MaxHingeLossWithRejection(torch.nn.Module):
    def __init__(self, cost:float, alpha:float=None, beta:float=None, use_softplus=True):
        super(MaxHingeLossWithRejection, self).__init__()
        assert 0 < cost < 0.5

        self.cost = cost
        self.alpha = alpha if alpha else 1.0
        self.beta = beta if beta else 1.0/(1.0-2.0*cost)
        self.use_softplus = use_softplus

        assert self.alpha > 0
        assert self.beta > 0

    def forward(self, prediction_out, rejection_out, target, num_classes:int=10):
        """
            prediction_out (B, #class): f(x)
            rejection_out  (B, 1): r(x)
        """
        # prediction_out_t = torch.gather(prediction_out, dim=1, index=target.view(-1,1)) # (B,1)
        # A = 1.0 + (prediction_out - prediction_out_t) # (B,#class)

        # convert target to (-1, 1)
        target = torch.nn.functional.one_hot(target, num_classes=num_classes)
        target = torch.where(target==1, target, -1*torch.ones_like(target))
        target = target.float()

        # braod cast to (b, #class) and take mean about #class.
        A = 1.0 + (self.alpha/2.0)*(rejection_out - target*prediction_out) # (b, #class)
        A = torch.mean(A, dim=-1) # (b)
        B = self.cost*(1.0 - self.beta*rejection_out) # (b, 1)
        B = B.view(-1) # (b)

        # take max and squared
        # zeros = torch.zeros_like(A, dtype=torch.float32, device='cuda', requires_grad=True)
        # max_squared = torch.max(torch.max(A,B), zeros)**2 # (b)
        # max_squared = torch.sum(max_squared, dim=-1) #(b)
        max_squared = torch.max(A,B)
        
        maxhinge_loss = max_squared.mean()

        # loss information dict
        loss_dict = OrderedDict()
        loss_dict['A mean'] = A.detach().mean().cpu().item()
        loss_dict['B mean'] = B.detach().mean().cpu().item()

        return maxhinge_loss, loss_dict