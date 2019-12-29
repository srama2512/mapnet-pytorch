import pdb
import h5py
import torch
import numpy as np

from mapnet.utils import compute_relative_pose, convert_polar2xyt

class DataLoaderAVD:
    def __init__(
        self,
        data_path,
        batch_size,
        num_steps,
        split,
        device,
        seed,
        env_name,
        max_steps=None,
        randomize_start_time=False,
    ):
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.split = split
        self.device = device
        self.rng = np.random.RandomState(seed)
        self.data_file = h5py.File(self.data_path, 'r')
        self.obs_keys = list(self.data_file[self.split].keys())
        self.nsplit = self.data_file[self.split][self.obs_keys[0]].shape[1]
        if max_steps is None:
            self.max_steps = self.data_file[self.split][self.obs_keys[0]].shape[0]
        else:
            self.max_steps = max_steps
        self.randomize_start_time = randomize_start_time
        self.observation_space = {
            key: self.data_file[self.split][key].shape[2:]
            for key in self.obs_keys
        }
        self.data_idx = 0
        self.env_name = env_name

    def sample(self):
        """
        Samples a random set of batch_size episodes. 

        Outputs:
            episode - generator with each output as 
                      dictionary, values are torch Tensor observations
        """
        if self.data_idx + self.batch_size >= self.nsplit:
            self.data_idx = 0
        device = self.device

        episodes = {}
        for key_ in self.obs_keys:
            key = 'rgb' if key_ == 'im' else key_
            episodes[key] = np.array(self.data_file[self.split][key_][:, self.data_idx:(self.data_idx + self.batch_size), ...])
            episodes[key] = episodes[key].astype(np.float32)
            episodes[key] = torch.Tensor(episodes[key])

        self.data_idx += self.batch_size

        n_time_batches = self.max_steps // self.num_steps
        poses = torch.zeros(self.max_steps, self.batch_size, 3)
        for t in range(1, self.max_steps):
            poses[t] = poses[t-1] + convert_polar2xyt(episodes['delta'][t])

        episodes['poses'] = poses

        if self.randomize_start_time:
            starts = np.random.randint(0, self.max_steps-self.num_steps, size=(1, ))
        else:
            starts = np.arange(0, self.max_steps-self.num_steps, self.num_steps).astype(np.int32)

        for t in starts:
            episodes_split = {
                key: val[t:(t+self.num_steps)].to(device)
                for key, val in episodes.items()
            } # (num_steps, batch_size, ...)
            yield episodes_split

    def close(self):
        self.data_file.close()

    def reset(self):
        self.data_file = h5py.File(self.data_path, 'r')
        self.data_idx = 0
