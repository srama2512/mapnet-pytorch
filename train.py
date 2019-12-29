import os
import sys
import copy
import glob
import time

import cv2
import pdb
import gym
import math
import numpy as np
import mapnet.algo as algo

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils

from collections import deque
from mapnet.arguments import get_args
from mapnet.model import MapNet
from mapnet.utils import (
    flatten_two,
    unflatten_two,
    get_camera_parameters,
    convert_world2map,
    convert_map2world,
    compute_relative_pose,
    process_image,
    process_maze_batch,
)
from mapnet.eval_utils import evaluate_avd, evaluate_maze
from mapnet.avd_data_loader import DataLoaderAVD
from torch.utils.data.dataloader import DataLoader

from gym.spaces import Discrete
from tensorboardX import SummaryWriter
from mapnet.resnet_parts import AVDResNet
from mapnet.cnn_models import get_two_layers_cnn
from mapnet.mazes import Mazes

from einops import rearrange, reduce, asnumpy, parse_shape
from einops.layers.torch import Rearrange, Reduce

args = get_args()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

try:
    os.makedirs(args.log_dir)
except OSError:
    pass

eval_log_dir = args.log_dir + "_eval"

try:
    os.makedirs(eval_log_dir)
except OSError:
    pass

def main():
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    if args.env_name == 'avd':
        args.feat_dim = 32
        args.map_shape = (args.feat_dim, args.map_size, args.map_size)
        args.local_map_shape = (args.feat_dim, args.local_map_size, args.local_map_size)
        train_loader = DataLoaderAVD(
             args.data_path,
             args.batch_size,
             args.num_steps,
             'train',
             device,
             args.seed,
             args.env_name,
             max_steps=None,
             randomize_start_time=True,
        )
        val_loader = DataLoaderAVD(
             args.data_path,
             args.batch_size,
             args.num_steps,
             'val',
             device,
             args.seed,
             args.env_name,
             max_steps=None,
             randomize_start_time=False,
        )
        args.obs_shape = train_loader.observation_space['im']
        args.camera_params = get_camera_parameters(args.env_name, args.obs_shape)
        args.angles = torch.Tensor(np.radians(np.linspace(0, 360, 13)[:-1])).to(device)
        args.top_down_inputs = False
    elif args.env_name == 'maze':
        args.feat_dim = 16
        args.map_shape = (args.feat_dim, args.map_size, args.map_size)
        args.local_map_shape = (args.feat_dim, args.local_map_size, args.local_map_size)
        env_size = (21, 21)
        full_set = Mazes(
            args.data_path,
            env_size,
            seq_length=args.num_steps,
            max_speed=0
        )
        (train_set, val_set) = torch.utils.data.random_split(full_set, (len(full_set)-5000, 5000))
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=16)
        val_loader = DataLoader(val_set, batch_size=10*args.batch_size, shuffle=False, num_workers=16)
        args.camera_params = None
        args.angles = torch.Tensor(np.radians(np.array([0.0, 90.0, 180.0, 270.0]))).to(device)
        args.top_down_inputs = True
    else:
        raise ValueError(f'env_name {args.env_name} is not defined!')

    save_path = os.path.join(args.save_dir, 'supervised')
    tbwriter = SummaryWriter(log_dir=args.log_dir)

    print('Map Size      : {}'.format(args.map_shape))
    print('Local Map Size: {}'.format(args.local_map_shape))

    print('==============> Preparing models')
    #================================================================================
    # =================== Encoder ====================
    mapnet_config = {
        'map_shape': args.map_shape,
        'local_map_shape': args.local_map_shape,
        'angles': args.angles,
        'map_scale': args.map_scale,
        'camera_params': args.camera_params,
        'device': device,
        'top_down_inputs': args.top_down_inputs,
    }
    if args.env_name == 'avd':
        cnn = AVDResNet()
    else:
        import argparse
        cnn_args = argparse.Namespace(bn=True)
        cnn = get_two_layers_cnn(cnn_args)

    mapnet = MapNet(mapnet_config, cnn)

    j_start = 0

    if os.path.isfile(os.path.join(save_path, args.env_name + ".pt")):
        print("Resuming from old model!")
        # Load models
        loaded_models = torch.load(os.path.join(save_path, args.env_name + ".pt"))
        mapnet_loaded = loaded_models[0] 
        # Load state_dicts
        mapnet.load_state_dict(mapnet_loaded.state_dict())
        # Resume settings
        j_start = loaded_models[1]

    mapnet.to(device)
    mapnet.train()

    print('==============> Preparing training algorithm')
    #================================================================================
    # =================== Define training algorithm ====================
    algo_config = {
        'mapnet': mapnet,
        'lr': args.lr,
        'max_grad_norm': args.max_grad_norm,
        'angles': args.angles,
    }

    supervised_agent = algo.SupervisedMapNet(algo_config)

    # =================== Eval metrics ================
    all_losses_deque     = None
    eval_val_metrics_deque = None
    eval_train_metrics_deque = None
    evaluate = evaluate_avd if args.env_name == 'avd' else evaluate_maze

    print('==============> Starting training')
    #================================================================================
    # =================== Training ====================
    start = time.time()
    if args.env_name == 'avd':
        episode_iter = None
    else:
        data_iter = None

    for j in range(j_start+1, args.num_updates):
        
        # Initialize things
        num_batches  = 0

        # Sample data
        if args.env_name == 'avd':
            sampled_data = False
            while not sampled_data:
                if episode_iter is None:
                    episode_iter = train_loader.sample()
                try:
                    batch = next(episode_iter)
                    sampled_data = True
                except StopIteration:
                    episode_iter = None
        else:
            sampled_data = False
            while not sampled_data:
                if data_iter is None:
                    data_iter = iter(train_loader)
                try:
                    batch = next(data_iter)
                    batch = process_maze_batch(batch, device)
                    sampled_data = True
                except StopIteration:
                    data_iter = None

        num_batches += 1
        obs = batch
        L, bs = obs['rgb'].shape[:2]
        if args.env_name == 'avd':
            obs['rgb'] = unflatten_two(process_image(flatten_two(obs['rgb'])), L, bs)
        obs_poses = obs['poses']
        start_pose = obs_poses[0].clone()

        # Transform the poses relative to the starting pose
        for l in range(L):
            obs_poses[l] = compute_relative_pose(start_pose, obs_poses[l]) # (x, y, theta)

        # Compute gt_pos
        gt_poses = convert_world2map(
            rearrange(obs_poses, 't b n -> (t b) n'),
            args.map_shape,
            args.map_scale, 
            args.angles,
        )
        gt_poses = rearrange(gt_poses, '(t b) n -> t b n', t=L) # (x, y, angle_idx)
        obs['gt_poses'] = gt_poses[1:]

        # Perform update
        all_losses = supervised_agent.update(obs)

        if all_losses_deque is None:
            all_losses_deque = {}
        for k, v in all_losses.items():
            if k not in all_losses_deque:
                all_losses_deque[k] = deque(maxlen=10)
            all_losses_deque[k].append(v)

        # =================== Save model ====================
        if j % args.save_interval == 0 and args.save_dir != "":
            save_path = os.path.join(args.save_dir, 'supervised')
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            # A really ugly way to save a model to CPU
            save_model          = mapnet
            if args.cuda:
                save_model      = copy.deepcopy(save_model).cpu()
            save_model_final    = [save_model, j]
            torch.save(save_model_final, os.path.join(save_path, args.env_name + ".pt"))

        # =================== Logging data ====================
        total_num_steps = (j + 1 - j_start) * args.batch_size * args.num_steps * num_batches

        if j % args.log_interval == 0:
            end                 = time.time()
            print_string        = '===> Updates {}, #steps {}, FPS {} \n'.format(j, total_num_steps, int(total_num_steps / (end - start))) 
            for loss_type, loss_deque in all_losses_deque.items():
                loss = np.mean(loss_deque).item()
                print_string += ', {}: {:.3f}'.format(loss_type, loss)
                tbwriter.add_scalar(loss_type, loss, j)
            print(print_string)

        # =================== Evaluate models ====================
        if (args.eval_interval is not None and j % args.eval_interval == 0):

            # Evaluating on val split
            eval_config = {
                'data_path': args.data_path,
                'batch_size': args.batch_size,
                'num_steps': args.num_steps,
                'split': 'val',
                'seed': args.seed,
                'map_shape': args.map_shape,
                'map_scale': args.map_scale,
                'angles': args.angles,
                'env_name': args.env_name,
                'max_batches': -1,
            }

            eval_model = {'mapnet': mapnet}

            print('============ Evaluating on val split ============')
            val_metrics = evaluate(eval_model, val_loader, eval_config, device)

            if eval_val_metrics_deque is None:
                eval_val_metrics_deque = {}
            for k, v in val_metrics.items():
                if k not in eval_val_metrics_deque:
                    eval_val_metrics_deque[k] = deque(maxlen=10)
                eval_val_metrics_deque[k].append(v)

            for key, value_deque in eval_val_metrics_deque.items():
                value = np.mean(value_deque).item()
                tbwriter.add_scalar('evaluation/val_' + key, value, j)

    tbwriter.close()

if __name__ == "__main__":
    main()
