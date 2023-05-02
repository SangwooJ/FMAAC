import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.misc import onehot_from_logits, categorical_sample

class BasePolicy(nn.Module):
    """
    Base policy network
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.leaky_relu,
                 norm_in=True, onehot_dim=0):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(BasePolicy, self).__init__()

        input_dim1 = 27
        input_dim2 = 9
        input_dim3 = 48

        if norm_in:  # normalize inputs
            self.in_fn1 = lambda x: x
            self.in_fn2 = nn.BatchNorm1d(input_dim2, affine=False)
            self.in_fn3 = nn.BatchNorm1d(input_dim3, affine=False)
        else:
            self.in_fn1 = lambda x: x
            self.in_fn2 = lambda x: x
            self.in_fn3 = lambda x: x
        self.fc1_1 = nn.Linear(input_dim1, hidden_dim)
        self.fc1_2 = nn.Linear(input_dim2, hidden_dim // 2)
        self.fc1_3 = nn.Linear(input_dim3, hidden_dim)
        self.fc2 = nn.Linear(2*hidden_dim + (hidden_dim//2), hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations (optionally a tuple that
                                additionally includes a onehot label)
        Outputs:
            out (PyTorch Matrix): Actions
        """
        X1 = X[:, :27]
        X2 = X[:, 27:36]
        X3 = X[:, 36:]

        onehot = None
        if type(X) is tuple:
            X, onehot = X
        inp1 = self.in_fn1(X1)  # don't batchnorm onehot
        inp2 = self.in_fn2(X2)
        inp3 = self.in_fn3(X3)
        #if onehot is not None:
        #    inp = torch.cat((onehot, inp), dim=1)
        h1 = self.nonlin(torch.cat([self.fc1_1(inp1), self.fc1_2(inp2), self.fc1_3(inp3)], dim=-1))
        h2 = self.nonlin(self.fc2(h1))
        out = self.fc3(h2)
        return out



class DiscretePolicy(BasePolicy):
    """
    Policy Network for discrete action spaces
    """
    def __init__(self, *args, **kwargs):
        super(DiscretePolicy, self).__init__(*args, **kwargs)

    def forward(self, obs, sample=True, return_all_probs=False,
                return_log_pi=False, regularize=False,
                return_entropy=False):
        out = super(DiscretePolicy, self).forward(obs)
        probs = F.softmax(out, dim=1)
        on_gpu = next(self.parameters()).is_cuda
        if sample:
            int_act, act = categorical_sample(probs, use_cuda=on_gpu)
        else:
            act = onehot_from_logits(probs)
        rets = [act]
        if return_log_pi or return_entropy:
            log_probs = F.log_softmax(out, dim=1)
        if return_all_probs:
            rets.append(probs)
        if return_log_pi:
            # return log probability of selected action
            rets.append(log_probs.gather(1, int_act))
        if regularize:
            rets.append([(out**2).mean()])
        if return_entropy:
            rets.append(-(log_probs * probs).sum(1).mean())
        if len(rets) == 1:
            return rets[0]
        return rets
