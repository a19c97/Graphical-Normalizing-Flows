"""
Implements Masked AutoEncoder for Density Estimation, by Germain et al. 2015
Re-implementation by Andrej Karpathy based on https://arxiv.org/abs/1502.03509
Modified by Antoine Wehenkel
"""

import numpy as np
from .Conditioner import Conditioner
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import gurobipy as gp
from gurobipy import GRB


"""
Functions to optimize masks
"""
def optimize_all_masks(hidden_sizes, A, opt_type):
    # Function returns mask list in order for layers from inputs to outputs
    # This order matches how the masks are assigned to the networks in MADE
    masks = []

    if torch.is_tensor(A):
        A = A.cpu().numpy()

    constraint = np.copy(A)
    for l in hidden_sizes:
        if opt_type == 'greedy':
            (M1, M2) = optimize_single_mask_greedy(constraint, l)
        elif opt_type == 'IP':
            (M1, M2) = optimize_single_mask_gurobi(constraint, l)
        elif opt_type == 'IP_alt':
            (M1, M2) = optimize_single_mask_gurobi(constraint, l, alt=True)
        elif opt_type == 'LP_relax':
            (M1, M2) = optimize_single_mask_gurobi(constraint, l, relax=True)
        elif opt_type == 'IP_var':
            (M1, M2) = optimize_single_mask_gurobi(constraint, l, var_pen=True)
        else:
            raise ValueError('opt_type is not recognized: ' + str(opt_type))

        constraint = M1
        masks = masks + [M2.T]   # take transpose for size: (n_inputs x n_hidden/n_output)
    masks = masks + [M1.T]

    return masks

def optimize_single_mask_greedy(A, n_hidden):
    # decompose A as M1 * M2
    # A size: (n_outputs x n_inputs)
    # M1 size: (n_outputs x n_hidden)
    # M2 size: (n_hidden x n_inputs)

    # find non-zero rows and define M2
    A_nonzero = A[~np.all(A == 0, axis=1),:]
    n_nonzero_rows = A_nonzero.shape[0]
    M2 = np.zeros((n_hidden, A.shape[1]))
    for i in range(n_hidden):
        M2[i,:] = A_nonzero[i % n_nonzero_rows]

    # find each row of M1
    M1 = np.ones((A.shape[0],n_hidden))
    for i in range(M1.shape[0]):
        # Find indices where A is zero on the ith row
        Ai_zero = np.where(A[i,:] == 0)[0]

        # find row using closed-form solution
        # find unique entries (rows) in j-th columns of M2 where Aij = 0
        row_idx = np.unique(np.where(M2[:,Ai_zero] == 1)[0])
        M1[i,row_idx] = 0.0

    return M1, M2


def optimize_single_mask_gurobi(A, n_hidden, alt=False, relax=False, var_pen=False):
    try:
        with gp.Env(empty=True) as env:
            env.setParam('LogToConsole', 0)
            env.start()
            with gp.Model(env=env) as m:
                # Create variables
                if relax:
                    # LP relaxation
                    M1 = m.addMVar((A.shape[0], n_hidden), lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="M1")
                    M2 = m.addMVar((n_hidden, A.shape[1]), lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="M2")
                    m.params.NonConvex = 2
                else:
                    # Original integer program
                    M1 = m.addMVar((A.shape[0], n_hidden), lb=0.0, ub=1.0, obj=0.0, vtype=GRB.BINARY, name="M1")
                    M2 = m.addMVar((n_hidden, A.shape[1]), lb=0.0, ub=1.0, obj=0.0, vtype=GRB.BINARY, name="M2")

                # Set constraints and objective
                if alt:
                    # Alternate formulation: might violate adjacency structure
                    m.setObjective(
                        sum(M1[i,:] @ M2[:,j] for i in range(A.shape[0]) for j in range(A.shape[1]) if A[i,j]==1) - \
                            sum(M1[i,:] @ M2[:,j] for i in range(A.shape[0]) for j in range(A.shape[1]) if A[i,j]==0),
                        GRB.MAXIMIZE
                    )
                else:
                    # Original formulation: guarantees adjacency structure is respected
                    m.addConstrs(
                        (M1[i, :] @ M2[:, j] <= A[i, j] for i in range(A.shape[0]) for j in range(A.shape[1]) if
                         A[i, j] == 0),
                        name='matrixconstraints')
                    m.addConstrs(
                        (M1[i, :] @ M2[:, j] >= A[i, j] for i in range(A.shape[0]) for j in range(A.shape[1]) if
                         A[i, j] > 0),
                        name='matrixconstraints2')

                    if var_pen:
                        # Variance-penalized objective
                        # m.setObjective(sum(A_prime) - diff(A_prime), GRB.MAXIMIZE) #<- doesn't work - can't multiply QuadExpr
                        A_prime = {}
                        for i in range(A.shape[0]):
                            A_prime[i] = {}
                            for j in range(A.shape[1]):
                                A_prime[i][j] = m.addVar(lb=0.0, ub=A.shape[0], obj=0.0, vtype=GRB.INTEGER, name=f"A_prime_{i}{j}")
                                m.addConstr((A_prime[i][j] == M1[i, :] @ M2[:, j]), name=f'constr_A_prime_{i}{j}')
                        m.setObjective(sum(A_prime) - variance(A_prime), GRB.MAXIMIZE)
                    else:
                        # Original objective
                        A_prime = [M1[i, :] @ M2[:, j] for i in range(A.shape[0]) for j in range(A.shape[1])]
                        m.setObjective(sum(A_prime), GRB.MAXIMIZE)

                # Optimize model
                m.optimize()
                obj_val = m.getObjective().getValue()

                # Obtain optimized results
                result = {}
                result['M1'] = np.zeros((A.shape[0], n_hidden))
                result['M2'] = np.zeros((n_hidden,A.shape[1]))
                for v in m.getVars():
                    if v.varName[0] == 'M':
                        nm = v.varName.split('[')[0]
                        idx = (v.varName.split('[')[1].replace(']','')).split(',')
                        col_idx = int(idx[0])
                        row_idx = int(idx[1])
                        if relax:
                            # Round real-valued solution to nearest integer
                            val = v.x
                            if val <= 0.5:
                                result[nm][col_idx][row_idx] = 0
                            else:
                                result[nm][col_idx][row_idx] = 1
                        else:
                            result[nm][col_idx][row_idx] = v.x

        print(f'Successful opt! obj = {obj_val}')
    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))
    except AttributeError:
        print('Encountered an attribute error')
    return result['M1'], result['M2']


def variance(data):
    n = len(data)
    mean = sum(data) / n
    squared_diff_sum = gp.LinExpr()
    for x in data:
        squared_diff_sum += gp.QuadExpr((x - mean) * (x - mean))
    variance = squared_diff_sum / n
    return variance


def diff(data):
    return max(data) - min(data)



class MaskedLinear(nn.Linear):
    """ same as Linear except has a configurable mask on the weights """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.ones(out_features, in_features))

    def set_mask(self, mask):
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))

    def forward(self, input):
        return F.linear(input, self.mask * self.weight, self.bias)


class MADE(nn.Module):
    def __init__(self, nin, hidden_sizes, nout, opt_type, act_type="relu",
                 num_masks=1, A_prior=None, natural_ordering=False,
                 random=False, device="cpu"):
        """
        nin: integer; number of inputs
        hidden sizes: a list of integers; number of units in hidden layers
        nout: integer; number of outputs, which usually collectively parameterize some kind of 1D distribution
              note: if nout is e.g. 2x larger than nin (perhaps the mean and std), then the first nin
              will be all the means and the second nin will be stds. i.e. output dimensions depend on the
              same input dimensions in "chunks" and should be carefully decoded downstream appropriately.
              the output of running the tests for this file makes this a bit more clear with examples.
        num_masks: can be used to train ensemble over orderings/connections
        natural_ordering: force natural ordering of dimensions, don't use random permutations
        """
        super().__init__()
        self.random = random
        self.nin = nin
        self.nout = nout
        self.hidden_sizes = hidden_sizes
        assert self.nout % self.nin == 0, "nout must be integer multiple of nin"
        self.opt_type = opt_type

        if act_type == "relu":
            self.act = nn.ReLU
        elif act_type == "tanh":
            self.act = nn.Tanh
        else:
            raise ValueError("Unknown Activation.")

        # Set adjacency matrix
        self.A = A_prior if not (A_prior is None) else np.tril(np.ones((nin, nin)), -1)

        # define a simple MLP neural net
        self.net = []
        hs = [nin] + hidden_sizes + [nout]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend([
                MaskedLinear(h0, h1),
                self.act(),
            ])
        self.net.pop()  # pop the last activation for the output layer
        self.net = nn.Sequential(*self.net)

        # seeds for orders/connectivities of the model ensemble
        self.natural_ordering = natural_ordering
        self.num_masks = num_masks
        self.seed = 0  # for cycling through num_masks orderings

        self.m = {}
        self.update_masks()  # builds the initial self.m connectivity
        # note, we could also precompute the masks and cache them, but this
        # could get memory expensive for large number of masks.

    def update_masks(self):
        masks = optimize_all_masks(self.hidden_sizes, self.A, self.opt_type)
        self.check_masks(masks, self.A)

        # handle the case where nout = nin * k, for integer k > 1
        if self.nout > self.nin:
            k = int(self.nout / self.nin)
            # replicate the mask across the other outputs
            masks[-1] = np.concatenate([masks[-1]] * k, axis=1)

        # set the masks in all MaskedLinear layers
        layers = [l for l in self.net.modules() if isinstance(l, MaskedLinear)]

        for l, m in zip(layers, masks):
            l.set_mask(m)

    def check_masks(self, mask_list, A):
        if torch.is_tensor(A):
            A = A.cpu().numpy()
        # compute mask product
        mask_prod = mask_list[-1].T
        for i in np.arange(len(mask_list)-2,-1,-1):
            mask_prod = np.dot(mask_prod,mask_list[i].T)
        constraint = (mask_prod>0) * 1. - A
        if np.any(constraint != 0.):
            raise ValueError('Constraints are not met. Do not proceed with masks.')

    def MADE_masks(self):
        if self.m and self.num_masks == 1: return  # only a single seed, skip for efficiency
        L = len(self.hidden_sizes)

        # fetch the next seed and construct a random stream
        rng = np.random.RandomState(self.seed)
        self.seed = (self.seed + 1) % self.num_masks

        # sample the order of the inputs and the connectivity of all neurons
        if self.random:
            self.m[-1] = np.arange(self.nin) if self.natural_ordering else rng.permutation(self.nin)
            for l in range(L):
                self.m[l] = rng.randint(self.m[l - 1].min(), self.nin - 1, size=self.hidden_sizes[l])
        else:
            self.m[-1] = np.arange(self.nin)
            for l in range(L):
                self.m[l] = np.array([self.nin - 1 - (i % self.nin) for i in range(self.hidden_sizes[l])])

        # construct the mask matrices
        masks = [self.m[l - 1][:, None] <= self.m[l][None, :] for l in range(L)]
        masks.append(self.m[L - 1][:, None] < self.m[-1][None, :])

        return masks

    def forward(self, x):
        return self.net(x).view(x.shape[0], -1, x.shape[1]).permute(0, 2, 1)


# ------------------------------------------------------------------------------

class ConditionalMADE(MADE):
    def __init__(self, nin, cond_in, hidden_sizes, nout, opt_type, A_prior,
                 num_masks=1, natural_ordering=False, random=False,
                 act_type="relu", device="cpu"):
        """
        nin: integer; number of inputs
        hidden sizes: a list of integers; number of units in hidden layers
        nout: integer; number of outputs, which usually collectively parameterize some kind of 1D distribution
              note: if nout is e.g. 2x larger than nin (perhaps the mean and std), then the first nin
              will be all the means and the second nin will be stds. i.e. output dimensions depend on the
              same input dimensions in "chunks" and should be carefully decoded downstream appropriately.
              the output of running the tests for this file makes this a bit more clear with examples.
        num_masks: can be used to train ensemble over orderings/connections
        natural_ordering: force natural ordering of dimensions, don't use random permutations
        """

        super().__init__(nin + cond_in, hidden_sizes, nout, opt_type, act_type,
                         num_masks, A_prior, natural_ordering, random, device)
        self.nin_non_cond = nin
        self.cond_in = cond_in

    def forward(self, x, context):
        if context is not None:
            out = super().forward(torch.cat((context, x), 1))
        else:
            out = super().forward(x)
        out = out.contiguous()[:, self.cond_in:, :]
        return out

class StrAFConditioner(Conditioner):
    def __init__(self, in_size, hidden, out_size, opt_type, A_prior,
                 act_type="relu", device="cpu", cond_in=0):
        super().__init__()
        self.in_size = in_size
        self.masked_autoregressive_net = ConditionalMADE(
            nin=in_size, A_prior=A_prior, cond_in=cond_in, hidden_sizes=hidden,
            nout=out_size*in_size, opt_type=opt_type, device=device, act_type=act_type
        )

    def forward(self, x, context=None):
        return self.masked_autoregressive_net(x, context)

    def depth(self):
        return self.in_size - 1


class StrODENet(StrAFConditioner):
    def __init__(self, in_size, hidden, A_prior, device, opt_type="greedy"):
        super().__init__(in_size, hidden, 1, opt_type, A_prior,
                         device=device, act_type="tanh")

    def forward(self, t, x):
        return super().forward(x)
