from random import randint, seed
from math import pi, cos, sin, ceil
from time import time

import pdb
import torch as t
from torch.utils.data.dataset import Dataset

def rotate90(tensor, times):
  """Rotate tensor clockwise by 90 degrees a number of times.
  Assumes spatial dimensions are the last ones."""
  if times == 1:  # 90 deg
    return tensor.transpose(-2, -1).flip(-1)
  elif times == 2:  # 180 deg
    return tensor.flip(-2).flip(-1)
  elif times == 3:  # 270 deg
    return tensor.transpose(-2, -1).flip(-2)
  else:  # 0 deg, no change
    assert times == 0
    return tensor


# also see: https://github.com/ActiveState/code/blob/3b27230f418b714bc9a0f897cb8ea189c3515e99/recipes/Python/578112_Bresenhams_line_algorithm/recipe-578112.py
def bresenham_line(x, y, x2, y2):
  """Brensenham line algorithm"""
  steep = 0
  coords = []
  dx = abs(x2 - x)
  if (x2 - x) > 0: sx = 1
  else: sx = -1
  dy = abs(y2 - y)
  if (y2 - y) > 0: sy = 1
  else: sy = -1
  if dy > dx:
    steep = 1
    x,y = y,x
    dx,dy = dy,dx
    sx,sy = sy,sx
  d = (2 * dy) - dx
  for i in range(0,dx):
    if steep: coords.append((y,x))
    else: coords.append((x,y))
    while d >= 0:
      y = y + sy
      d = d - (2 * dx)
    x = x + sx
    d = d + (2 * dy)
  coords.append((x2,y2))
  return coords

class Mazes(Dataset):
  """Dataset of 2D mazes, simulating local view with occlusions. The view is returned as a Bx2xHxW byte tensor."""
  def __init__(self, filename, env_size, view_range=5, seq_length=5, visible_threshold=3, no_rotation=False, max_speed=0):
    # read all text at once
    with open(filename, 'r') as f: text = f.read()
    
    # convert to a 1D tensor of bytes
    data = t.tensor(list(text.encode()), dtype=t.uint8)

    # reshape into a stack of lines
    data = data.reshape(-1, env_size[0] + 1)

    # remove line break (character 10) at the end of each line
    assert (data[:,-1] == 10).all()
    data = data[:,:-1]

    # reshape to split lines into list of environments along first dim
    self.envs = (data.reshape(-1, env_size[1], env_size[0]) == 35)

    self.view_range = view_range
    self.seq_length = seq_length
    self.visible_threshold = visible_threshold
    self.max_speed = max_speed

    # raycast and store resulting lines, starting from the origin, for all 4 directions
    r = self.view_range  # radius
    perimeter = 2 * pi * r
    self.rays = []
    for ang in range(1 if no_rotation else 4):
      if no_rotation:  # fixed, full 360 deg view
        angles = t.linspace(0, 2 * pi, int(ceil(perimeter)))
      else:  # oriented, 180 FOV
        base_angle = ang * pi / 2
        angles = t.linspace(base_angle - pi / 2, base_angle + pi / 2, int(ceil(perimeter)))

      rays = [tuple(bresenham_line(0, 0, round(r * cos(a)), round(r * sin(a)))) for a in angles]
      rays = list(set(rays))  # remove duplicates
      #rays = [t.tensor(r, dtype=t.long) for r in rays]
      self.rays.append(rays)

  def raycast(self, env, x, y, ang):
    """Raycast in an environment to black-out any non-visible tiles"""
    image = t.zeros((2, env.shape[0], env.shape[1]))
    for ray in self.rays[ang]:
      #assert isinstance(ray, tuple)
      for (rx, ry) in ray:
        v = int(env[y + ry, x + rx].item())
        image[v, y + ry, x + rx] = 1
        if v: break  # hit a wall
    return image

  def __len__(self):
    """Return number of environments"""
    return self.envs.shape[0]

  def __getitem__(self, index):
    """Return a random sequence of images from the environment with given index"""
    start = time()
    # get environment
    env = self.envs[index, :, :]

    # possible initial positions: free space
    pos = (~env).nonzero()  #(1 - env).nonzero()

    # output tensor with image sequence
    images = t.zeros((self.seq_length, 2, self.view_range * 2 + 1, self.view_range * 2 + 1), dtype=t.uint8)
    poses = []
    frame = 0

    for tries in range(10 * self.seq_length):
      # choose random position (restricted to available choices)
      (y, x) = pos[randint(0, pos.shape[0] - 1), :].tolist()
      assert env[y, x] == 0  # sanity check, tile shouldn't be a wall

      # choose random orientation
      ang = randint(0, len(self.rays) - 1)

      # project rays that position/orientation
      image = self.raycast(env, x, y, ang)
      
      # accept it if there are enough visible tiles
      visible = image[0, :, :]
      if visible.sum() >= self.visible_threshold:
        # jump to a visible position next
        new_pos = visible.nonzero()

        # avoid this position if there are no valid tiles reachable within the max speed (distance from one frame to the next)
        if self.max_speed:
          is_close = ((new_pos - t.tensor([[y, x]])) ** 2).sum(dim=1) <= self.max_speed ** 2
          if is_close.sum() == 0: continue
          new_pos = new_pos[is_close, :]
        
        pos = new_pos

        # store results for this frame
        images[frame, :, :, :] = extract_view(image, x, y, ang, self.view_range)
        poses.append((x, y, pi / 2 * ang))

        frame += 1
        if frame >= self.seq_length: break  # done

    poses = t.tensor(poses, dtype=t.float)
    return {'rgb': images, 'poses': poses}#, 'time': time() - start}


def extract_view(env, x, y, ang, view_range):
  """Extract a local view from an environment at the given pose"""
  # get coordinates of window to extract
  xs = t.arange(x - view_range, x + view_range + 1, dtype=t.long)
  ys = t.arange(y - view_range, y + view_range + 1, dtype=t.long)

  # get coordinate 0 instead of going out of bounds
  (h, w) = (env.shape[-2], env.shape[-1])
  (invalid_xs, invalid_ys) = ((xs < 0) | (xs >= w), (ys < 0) | (ys >= h))  # coords outside the env
  xs[invalid_xs] = 0
  ys[invalid_ys] = 0

  # extract view, and set to 0 observations that were out of bounds
  #view = env[..., ys, xs]  # not equivalent to view = env[..., y1:y2, x1:x2]
  view = env.index_select(dim=-2, index=ys).index_select(dim=-1, index=xs)

  view[..., :, invalid_xs] = 0
  view[..., invalid_ys, :] = 0

  # rotate back. note only 90 degrees rotations are allowed
  return rotate90(view, (-ang) % 4)


if __name__ == '__main__':
  from overboard import tshow
  seed(0)  # repeatable random sequence

  mazes = Mazes('data/maze/mazes-10-10-100000.txt', env_size=(21, 21))
  
  tshow(mazes.envs[0:6, :, :])

  images = [mazes[0]['rgb'] for _ in range(10)]
  images = t.stack(images, dim=0)
  images = images[:,:,0,:,:] - images[:,:,1,:,:]  # difference between wall and ground

  tshow(images)
  input()
