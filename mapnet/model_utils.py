import sys
import pdb
import copy
import math
import numpy as np

import torch
import torch_scatter
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, reduce, asnumpy, parse_shape
from einops.layers.torch import Rearrange, Reduce

def rotate_tensor(r, t):
    """
    Inputs:
        r     - (bs, f, h, w) Tensor
        t     - (bs, ) Tensor of angles
    Outputs:
        r_rot - (bs, f, h, w) Tensor
    """
    device = r.device

    sin_t  = torch.sin(t)
    cos_t  = torch.cos(t)
    # This R convention means Y axis is downwards.
    A      = torch.zeros(r.size(0), 2, 3).to(device)
    A[:, 0, 0] = cos_t
    A[:, 0, 1] = sin_t
    A[:, 1, 0] = -sin_t
    A[:, 1, 1] = cos_t

    grid   = F.affine_grid(A, r.size())
    r_rot  = F.grid_sample(r, grid)

    return r_rot

def rotation_resample(x, angles_, offset=0.0):
    """
    Inputs:
        x       - (bs, f, s, s) feature maps
        angles_ - (nangles, ) set of angles to sample
    Outputs:
        x_rot   - (bs, nangles, f, s, s)
    """
    angles      = angles_.clone() # (nangles, )
    bs, f, s, s = x.shape
    nangles     = angles.shape[0]
    x_rep       = x.unsqueeze(1).expand(-1, nangles, -1, -1, -1) # (bs, nangles, f, s, s)
    x_rep       = rearrange(x_rep, 'b o e h w -> (b o) e h w')
    angles      = angles.unsqueeze(0).expand(bs, -1).contiguous().view(-1) # (bs * nangles, )
    x_rot       = rotate_tensor(x_rep, angles + math.radians(offset)) # (bs * nangles, f, s, s)
    x_rot       = rearrange(x_rot, '(b o) e h w -> b o e h w', b=bs)

    return x_rot

def localize(x_rot, maps):
    """ 
    Localize input features in maps
    Inputs: 
        x_rot  - (bs, nangles, f, s, s) ground projection of image 
        maps   - (bs, f, H, W) full map 
        angles - (nangles, ) set of angles to sample
    Outputs:
        poses  - (bs, nangles, H, W) softmax over all poses
        x_rot  - (bs, nangles, f, s, s)
    """ 
    bs, nangles, f, s, s = x_rot.shape
    _, _, H, W = maps.shape

    # Look at https://github.com/BarclayII/hart-pytorch/blob/master/dfn.py 
    # for how batch-wise convolutions are performed
    x_rot = rearrange(x_rot, 'b o e h w -> (b o) e h w')
    maps  = rearrange(maps, 'b e h w -> () (b e) h w')
    poses = F.conv2d(maps, x_rot, stride=1, padding=s//2, groups=bs) # (1, bs*nangles, H, W)
    poses = F.softmax(rearrange(poses, '() (b o) h w -> b (o h w)', b=bs), dim=1)
    poses = rearrange(poses, 'b (o h w) -> b o h w', h=H, w=W)

    return poses

def project_to_ground_plane(img_feats, spatial_locs, valid_inputs, local_shape, K, eps=-1e16):
    """
    Project image features to locations in ground-plane given by spatial_locs.
    Inputs:
        img_feats       - (bs, f, H/K, W/K) image features to project to ground plane
        spatial_locs    - (bs, 2, H, W)
                          for each pixel in each batch, the (x, y) ground-plane locations are given.
        valid_inputs    - (bs, 1, H, W) ByteTensor
        local_shape     - (outh, outw) tuple indicating size of output projection
        K               - image_size / map_shape ratio (needed for sampling values from spatial_locs)
        eps             - fill_value
    Outputs:
        proj_feats      - (bs, f, s, s)
    """
    device = img_feats.device
    outh, outw = local_shape
    bs, f, HbyK, WbyK = img_feats.shape

    # Sub-sample spatial_locs, valid_inputs according to img_feats resolution.
    idxes_ss = ((torch.arange(0, HbyK, 1)*K).long().to(device), \
                (torch.arange(0, WbyK, 1)*K).long().to(device))

    spatial_locs_ss = spatial_locs[:, :, idxes_ss[0][:, None], idxes_ss[1]] # (bs, 2, HbyK, WbyK)
    valid_inputs_ss = valid_inputs[:, :, idxes_ss[0][:, None], idxes_ss[1]] # (bs, 1, HbyK, WbyK)
    valid_inputs_ss = valid_inputs_ss.squeeze(1) # (bs, HbyK, WbyK)
    invalid_inputs_ss = ~valid_inputs_ss

    # Filter out invalid spatial locations
    invalid_spatial_locs = (spatial_locs_ss[:, 1] >= outh) | (spatial_locs_ss[:, 1] < 0 ) | \
                           (spatial_locs_ss[:, 0] >= outw) | (spatial_locs_ss[:, 0] < 0 ) # (bs, H, W)

    invalid_writes = invalid_spatial_locs | invalid_inputs_ss

    # Set the idxes for all invalid locations to (0, 0)
    spatial_locs_ss[:, 0][invalid_writes] = 0
    spatial_locs_ss[:, 1][invalid_writes] = 0

    # Weird hack to account for max-pooling negative feature values
    invalid_writes_f = rearrange(invalid_writes, 'b h w -> b () h w').float()
    img_feats_masked = img_feats * (1 - invalid_writes_f) + eps * invalid_writes_f
    img_feats_masked = rearrange(img_feats_masked, 'b e h w -> b e (h w)')

    # Linearize ground-plane indices (linear idx = y * W + x)
    linear_locs_ss = spatial_locs_ss[:, 1] * outw + spatial_locs_ss[:, 0] # (bs, H, W)
    linear_locs_ss = rearrange(linear_locs_ss, 'b h w -> b () (h w)')
    linear_locs_ss = linear_locs_ss.expand(-1, f, -1) # .contiguous()

    proj_feats, _ = torch_scatter.scatter_max(
                        img_feats_masked,
                        linear_locs_ss,
                        dim=2,
                        dim_size=outh*outw,
                        fill_value=eps,
                    )
    proj_feats = rearrange(proj_feats, 'b e (h w) -> b e h w', h=outh)

    # Replace invalid features with zeros
    eps_mask = (proj_feats == eps).float()
    proj_feats = proj_feats * (1 - eps_mask) + eps_mask * (proj_feats - eps)

    return proj_feats

def compute_spatial_locs(depth_inputs, local_shape, local_scale, camera_params):
    """
    Compute locations on the ground-plane for each pixel.
    Inputs:
        depth_inputs  - (bs, 1, imh, imw) depth values per pixel in `units`. 
        local_shape   - (s, s) tuple of ground projection size
        local_scale   - cell size of ground projection in `units`
        camera_params - (fx, fy, cx, cy) tuple
    Outputs:
        spatial_locs  - (bs, 2, imh, imw) x,y locations of projection per pixel
        valid_inputs  - (bs, 1, imh, imw) ByteTensor (all locations where depth measurements are available)
    """
    fx, fy, cx, cy  = camera_params
    bs, _, imh, imw = depth_inputs.shape
    s               = local_shape[1]
    device          = depth_inputs.device

    # 2D image coordinates
    x               = rearrange(torch.arange(0, imw), 'w -> () () () w')
    y               = rearrange(torch.arange(imh, 0, step=-1), 'h -> () () h ()')
    x, y            = x.float().to(device), y.float().to(device)
    xx              = (x - cx) / fx
    yy              = (y - cy) / fy

    # 3D real-world coordinates (in meters)
    Z               = depth_inputs
    X               = xx * Z
    Y               = yy * Z
    valid_inputs    = (depth_inputs != 0)
    # 2D ground projection coordinates (in meters)
    # Note: map_scale - dimension of each grid in meters
    # - depth/scale + (s-1)/2 since image convention is image y downward
    # and agent is facing upwards.
    x_gp            = ( (X / local_scale) + (s-1)/2).round().long() # (bs, 1, imh, imw)
    y_gp            = (-(Z / local_scale) + (s-1)/2).round().long() # (bs, 1, imh, imw)

    return torch.cat([x_gp, y_gp], dim=1), valid_inputs
