import pdb
import math
import numpy as np
import torch
import torch.nn as nn

from einops import rearrange, reduce, asnumpy, parse_shape
from einops.layers.torch import Rearrange, Reduce

from mapnet.utils import (
    flatten_two,
    unflatten_two,
    convert_map2world,
    convert_world2map,
    norm_angle,
    compute_relative_pose,
    process_image,
    process_maze_batch,
)

def compute_metrics(p, gt_world_pos, map_shape, map_scale, angles):
    """
    Inputs: 
        p - (L, bs, nangles, H, W) numpy array of probabilities 
            at each time step for each test element
        gt_world_pos - (L, bs, 3) numpy array of (x, y, theta)
        angles - (nangles, ) torch Tensor

    Outputs:
        metrics     - a dictionary containing the different metrics measured
    """
    metrics = {}

    L, bs, nangles, H, W = p.shape
    # Compute localization loss
    gt_pos = convert_world2map(
                 rearrange(torch.Tensor(gt_world_pos), 't b n -> (t b) n'),
                 map_shape,
                 map_scale,
                 angles,
             ).long() # (L*bs, 3) ---> (x, y, dir)

    logp = rearrange(np.log(p + 1e-12), 't b o h w -> (t b) o h w')
    logp_at_gtpos = logp[range(gt_pos.shape[0]), gt_pos[:, 2], gt_pos[:, 1], gt_pos[:, 0]] # (L*bs, )
    
    metrics['loc_loss'] = - logp_at_gtpos.mean().item()

    pred_pos = np.unravel_index(np.argmax(logp.reshape(L*bs, -1), axis=1), logp.shape[1:])
    pred_pos = np.stack(pred_pos, axis=1) # (L*bs, 3) shape with (theta_idx, y, x)
    pred_pos = np.ascontiguousarray(np.flip(pred_pos, axis=1)) # Convert to (x, y, theta_idx)
    pred_pos = torch.Tensor(pred_pos).long() # (L*bs, 3) --> (x, y, dir)
    pred_world_pos = convert_map2world(pred_pos, map_shape, map_scale, angles) # (L*bs, 3) --> (x_world, y_world, theta_world)
    pred_world_pos = asnumpy(pred_world_pos)
    gt_world_pos = rearrange(gt_world_pos, 't b n -> (t b) n')

    # Compute APE - average position error
    ape_all = np.linalg.norm(pred_world_pos[:, :2] - gt_world_pos[:, :2], axis=1)
    ape_all = rearrange(ape_all, '(t b) -> t b', t=L)
    metrics['median/ape'] = np.median(ape_all, axis=1).mean().item()
    metrics['mean/ape'] = np.mean(ape_all).item()
    # Compute DE - direction error
    pred_angle = torch.Tensor(pred_world_pos[:, 2])
    gt_angle = torch.Tensor(gt_world_pos[:, 2])
    de_all = torch.abs(norm_angle(pred_angle - gt_angle))
    de_all = asnumpy(rearrange(de_all, '(t b) -> t b', t=L))
    metrics['median/de'] = np.median(de_all, axis=1).mean().item()
    metrics['mean/de'] = np.mean(de_all).item()

    print("\nEvaluation using {} episodes\n=========== metrics =============".format(bs))
    for k, v in metrics.items():
        print(f'{k:<20s}: {v:^10.3f}')
    return metrics

def evaluate_avd(model, eval_data_loader, config, device):

    batch_size = config['batch_size']
    num_steps = config['num_steps']
    split = config['split']
    seed = config['seed']
    map_shape = config['map_shape']
    map_scale = config['map_scale']
    angles = config['angles']
    env_name = config['env_name']
    max_batches = config['max_batches']

    mapnet = model['mapnet']

    eval_data_loader.reset()
    # Set to evaluation mode
    mapnet.eval()

    # Gather evaluation information
    eval_p_all = []
    eval_gt_poses_all = []
    if max_batches == -1:
        num_batches = eval_data_loader.nsplit // batch_size
    else:
        num_batches = max_batches

    for eval_ep in range(0, num_batches):
        episode_iter = eval_data_loader.sample()
        # Storing evaluation information per process.

        for episode_batch in episode_iter:
            obs = episode_batch
            L, bs = obs['rgb'].shape[:2]
            obs['rgb'] = unflatten_two(process_image(flatten_two(obs['rgb'])), L, bs)
            obs_poses = obs['poses']
            start_pose = obs_poses[0].clone()
            # Transform the poses relative to the starting pose
            for l in range(L):
                obs_poses[l] = compute_relative_pose(start_pose, obs_poses[l]) # (x, y, theta)

            with torch.no_grad():
                # (L-1, bs, nangles, H, W)
                p_all = mapnet(obs)['poses']

            gt_world_poses = obs_poses[1:]
            eval_p_all.append(asnumpy(p_all))
            eval_gt_poses_all.append(asnumpy(gt_world_poses))

    eval_p_all = np.concatenate(eval_p_all, axis=1) # (L-1, bs, nangles, H, W)
    eval_gt_poses_all = np.concatenate(eval_gt_poses_all, axis=1) # (L-1, bs, 3)

    eval_data_loader.close()
    mapnet.train()

    metrics = compute_metrics(eval_p_all, eval_gt_poses_all, map_shape, map_scale, angles.cpu()) 

    return metrics

def evaluate_maze(model, eval_data_loader, config, device):

    batch_size = config['batch_size']
    num_steps = config['num_steps']
    split = config['split']
    seed = config['seed']
    map_shape = config['map_shape']
    map_scale = config['map_scale']
    angles = config['angles']
    env_name = config['env_name']

    mapnet = model['mapnet']

    # Set to evaluation mode
    mapnet.eval()

    # Gather evaluation information
    eval_p_all = []
    eval_gt_poses_all = []
    count = 0

    for batch in eval_data_loader:
        obs = process_maze_batch(batch, device)
        L, bs = obs['rgb'].shape[:2]
        obs_poses = obs['poses']
        start_pose = obs_poses[0].clone()
        # Transform the poses relative to the starting pose
        for l in range(L):
            obs_poses[l] = compute_relative_pose(start_pose, obs_poses[l]) # (x, y, theta)

        with torch.no_grad():
            # (L-1, bs, nangles, H, W)
            p_all = mapnet(obs)['poses']

        gt_world_poses = obs_poses[1:]
        eval_p_all.append(asnumpy(p_all))
        eval_gt_poses_all.append(asnumpy(gt_world_poses))

    eval_p_all = np.concatenate(eval_p_all, axis=1) # (L-1, bs, nangles, H, W)
    eval_gt_poses_all = np.concatenate(eval_gt_poses_all, axis=1) # (L-1, bs, 3)

    mapnet.train()

    metrics = compute_metrics(eval_p_all, eval_gt_poses_all, map_shape, map_scale, angles.cpu())

    return metrics
