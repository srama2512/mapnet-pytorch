import pdb
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from einops import rearrange, reduce, asnumpy, parse_shape
from einops.layers.torch import Rearrange, Reduce

class SupervisedMapNet:
    def __init__(self, config):

        # Models
        self.mapnet        = config['mapnet']

        # Optimization
        self.max_grad_norm = config['max_grad_norm']
        lr                 = config['lr']

        # Others
        self.angles        = config['angles']

        # Define optimizers
        fltr               = lambda x: [param for param in x if param.requires_grad == True]
        self.optimizer     = optim.Adam(fltr(self.mapnet.parameters()), lr=lr)

    def update(self, inputs, debug_flag=False):
        gt_poses = inputs['gt_poses']
        outputs = self.mapnet(inputs)

        # Self-localization loss
        loc_loss = 0.0
        logp = (outputs['poses'] + 1e-12).log() # ((L-1), bs, nangles, map_size, map_size)
        logp = rearrange(logp, 't b o h w -> (t b) o h w')
        gt_poses = rearrange(gt_poses.long(), 't b n -> (t b) n') # ((L-1)*bs, 3)
        logp_at_gt_poses = logp[range(logp.shape[0]), gt_poses[:, 2], gt_poses[:, 1], gt_poses[:, 0]]
        loss = -logp_at_gt_poses.mean()

        self.optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(self.mapnet.parameters(), self.max_grad_norm)

        self.optimizer.step()

        losses = {} 
        losses['loss'] = loss.item()

        return losses
