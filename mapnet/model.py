import csv
import sys
import pdb
import copy
import math
import torch
import torch_scatter
import torch.nn as nn
import torch.nn.functional as F

from mapnet.model_utils import (
    localize, 
    rotation_resample, 
    compute_spatial_locs
    project_to_ground_plane,
)
from mapnet.model_aggregate import GRUAggregate
from mapnet.resnet_parts import *

from einops import rearrange, reduce, asnumpy, parse_shape
from einops.layers.torch import Rearrange, Reduce

class MapNet(nn.Module):
    def __init__(self, config, cnn):
        super().__init__()

        # map_shape: (F, height, width)
        self._map_shape       = config['map_shape']
        self._out_shape       = config['map_shape']
        self.angles           = config['angles']
        self.nangles          = self.angles.shape[0]
        self._local_map_shape = config['local_map_shape']
        self.map_scale        = config['map_scale']
        # Aggregator
        self.aggregate        = GRUAggregate(self._map_shape[0])

        f, H, W               = self._map_shape
        self._map_center      = (H//2, W//2)
        self.zero_angle_idx   = None
        for i in range(self.angles.shape[0]):
            if self.angles[i].item() == 0:
                self.zero_angle_idx = i
                break

        self.top_down_inputs  = config['top_down_inputs']

        if not self.top_down_inputs:
            hs, mh, mw = config['map_shape']
            self.fx = config['camera_params']['fx']
            self.fy = config['camera_params']['fy']
            self.cx = config['camera_params']['cx']
            self.cy = config['camera_params']['cy']
            self.K = config['camera_params']['K']
            self.fov = config['camera_params']['fov']

        self.main = cnn

        self.train()

    def forward(self, inputs, debug=False):
        """
        inputs - dictionary with the following keys
            rgb   - (L, bs, C, H, W)
            depth - (L, bs, 1, H, W) --- only for 3D settings
        """
        L, bs, _, H, W = inputs['rgb'].shape
        device = inputs['rgb'].device
        map_size = self._map_shape[1]

        # Compute image features
        encoded_outputs = self.encode_and_project(inputs)
        x_gp_all = encoded_outputs['x_gp'] # (L, bs, f, local_map_size, local_map_size)
        x_gp_rot = rotation_resample(x_gp_all[0], self.angles)

        # Initializations
        maps = torch.zeros(bs, *self._map_shape, device=device) # (bs, f, map_size, map_size)
        poses = torch.zeros(bs, self.nangles, map_size, map_size, device=device) # (bs, 3)
        poses[:, self.zero_angle_idx, (map_size-1)//2, (map_size-1)//2] = 1.0
        maps_all = []
        poses_all = []

        for l in range(L-1):
            # Register current observation at the estimated pose
            maps = self._write_memory(x_gp_rot, maps, poses)

            # Localize next observation
            x_gp = x_gp_all[l+1]
            x_gp_rot = rotation_resample(x_gp, self.angles)
            poses = self._localize(x_gp_rot, maps)

            maps_all.append(maps)
            poses_all.append(poses)

        maps_all = torch.stack(maps_all, dim=0) # (L-1, bs, f, map_size, map_size)
        poses_all = torch.stack(poses_all, dim=0) # (L-1, bs, nangles, map_size, map_size)

        outputs = {'x_gp': x_gp_all, 'poses': poses_all, 'maps': maps_all}

        return outputs

    def get_feats(self, inputs):
        x = self.main(inputs)
        return x

    def encode_and_project(self, inputs):
        L, bs = inputs['rgb'].shape[:2]
        rgb = rearrange(inputs['rgb'], 't b c h w -> (t b) c h w')
        x = self.get_feats(rgb) # (L*bs, F, H/K, W/K)
        if self.top_down_inputs:
            x_gp = rearrange(x, '(t b) e h w -> t b e h w', t=L)
            outputs = {'x_gp': x_gp}
        else:
            depth = rearrange(inputs['depth'], 't b () h w -> (t b) () h w')
            spatial_locs, valid_inputs = self._compute_spatial_locs(depth) # (L*bs, 2, H, W)
            x_gp = self._project_to_ground_plane(x, spatial_locs, valid_inputs) # (L*bs, F, s, s)
            outputs = {
                'x_gp': rearrange(x_gp, '(t b) e h w -> t b e h w', t=L),
                'spatial_locs': rearrange(spatial_locs, '(t b) c h w -> t b c h w', t=L),
                'valid_inputs': rearrange(valid_inputs, '(t b) c h w -> t b c h w', t=L),
            }

        return outputs

    def _project_to_ground_plane(self, img_feats, spatial_locs, valid_inputs, eps=-1e16):
        """
        Inputs:
            img_feats       - (bs, f, H/K, W/K)
            spatial_locs    - (bs, 2, H, W)
                              for each pixel in each batch, the (x, y) ground-plane locations are given.
            valid_inputs    - (bs, 1, H, W) ByteTensor
            eps             - fill_value
        Outputs:
            proj_feats      - (bs, f, s, s)
        """
        proj_feats = project_to_ground_plane(
                         img_feats,
                         spatial_locs,
                         valid_inputs,
                         self._local_map_shape[1:],
                         self.K,
                         eps=eps
                     )
        return proj_feats

    def _compute_spatial_locs(self, depth_inputs):
        """
        Inputs:
            depth_inputs - (bs, 1, imh, imw) depth values per pixel in meters. 
        Outputs:
            spatial_locs - (bs, 2, imh, imw) x,y locations of projection per pixel
            valid_inputs - (bs, 1, imh, imw) ByteTensor (all locations where depth measurements are available)
        """
        camera_params = (self.fx, self.fy, self.cx, self.cy)
        local_scale = self.map_scale
        local_shape = self._local_map_shape[1:]
        spatial_locs, valid_inputs = compute_spatial_locs(depth_inputs, local_shape, local_scale, camera_params)

        return spatial_locs, valid_inputs

    def _write_memory(self, o, m, p):
        """
        Inputs:
            o  - (bs, nangles, f, s, s) Tensor of ground plane projection of current image
            m  - (bs, f, H, W) Tensor of overall map
            p  - (bs, nangles, H, W) probabilities over (orientations, y, x)
        Outputs:
            m  - (bs, f, H, W) Tensor of overall map after update
        """
        # assume s is odd
        bs, nangles, H, W  = p.shape
        f = m.shape[1]
        view_range = (o.shape[-1] - 1) // 2

        p_    = rearrange(p, 'b o h w -> () (b o) h w')
        o_    = rearrange(o, 'b o e h w -> (b o) e h w')
        o_reg = F.conv_transpose2d(p_, o_, groups=bs, padding=view_range) # (1, bs*f, H, W)
        o_reg = rearrange(o_reg, '() (b e) h w -> b e h w', b=bs)
        m     = self._update_map(o_reg, m)

        return m

    def _update_map(self, o_, m):
        """
        Inputs:
            o_  - (bs, f, H, W)
            m   - (bs, f, H, W)
        """
        bs, f, H, W = o_.size()

        # Update feature map
        o_    = rearrange(o_, 'b e h w -> (b h w) e')
        m     = rearrange(m, 'b e h w -> (b h w) e')
        m     = self.aggregate(o_, m) # (bs*H*W, f)
        m     = rearrange(m, '(b h w) e -> b e h w', b=bs, h=H, w=W)

        return m

    def _localize(self, x_rot, maps):
        """ 
        Inputs: 
            x_rot  - (bs, nangles, f, s, s) ground projection of image 
            maps   - (bs, f, H, W) full map 
        Outputs:
            poses  - (bs, nangles, H, W) softmax over all poses
        """ 
        poses = localize(x_rot, maps)

        return poses
