import argparse

import torch

def str2bool(v):
    if v.lower() in ['y', 'yes', 't', 'true']:
        return True
    return False

def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--lr', type=float, default=7e-4,
                        help='learning rate (default: 7e-4)')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument('--num-steps', type=int, default=5,
                        help='number of forward steps in A2C (default: 5)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='log interval, one log per n updates (default: 10)')
    parser.add_argument('--save-interval', type=int, default=100,
                        help='save interval, one save per n updates (default: 100)')
    parser.add_argument('--eval-interval', type=int, default=None,
                        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument('--env-name', default='avd',
                        help='environment to train on [ avd | maze ]')
    parser.add_argument('--log-dir', default='./logs',
                        help='directory to save agent logs (default: ./logs/')
    parser.add_argument('--save-dir', default='./trained_models/',
                        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--map-size', type=int, default=21,
                        help='dimension of memory')
    parser.add_argument('--local-map-size', type=int, default=7,
                        help='dimension of local ground projection')
    parser.add_argument('--map-scale', type=float, default=1,
                        help='number of pixels per grid length')
    parser.add_argument('--num-updates', type=int, default=100000)
    parser.add_argument('--data-path', type=str, default='data.h5',
                        help='Path to file containing source dataset')

    ####################### Evaluation arguments ##########################
    parser.add_argument('--load-path', type=str, default='model.pt')
    parser.add_argument('--eval-episodes', type=int, default=10000,
                        help='number of random episodes to evaluate over')
    parser.add_argument('--eval-split', type=str, default='val', choices=['train', 'val', 'test'])

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.num_refs = 1

    return args
