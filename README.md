# mapnet-pytorch
This is an unofficial PyTorch re-implementation of MapNet: An Allocentric Spatial Memory for Mapping Environments (CVPR 2018, oral).
It reproduces some preliminary results on Maze and AVD environments. Some parts of the code were borrowed from the original implementation
provided by the author at https://github.com/jotaf98/mapnet. 

# Requirements
The following are the primary requirements:
- PyTorch 1.2
- Python 3.6
- PyTorch scatter (https://github.com/rusty1s/pytorch_scatter)
- Einops (https://github.com/arogozhnikov/einops)
- TensorboardX 

# Datasets
- Maze dataset: https://github.com/jotaf98/mapnet/tree/master/data/maze
- [Active Vision Dataset (processed)](https://drive.google.com/file/d/145Qe2wHglKYB0burJchWcbruuBQWckrc/view?usp=sharing)

# Setup
Add the following to ~/.bashrc
```
export MAPNET_PATH=<PATH TO mapnet-pytorch>
export PYTHONPATH=$PYTHONPATH:$MAPNET_PATH
```
# Usage
## Training on Maze environment
```
cd $MAPNET_PATH
mkdir -p trained_models/maze_models

python train.py --lr 3e-3 --seed 1 --batch-size 100 --num-steps 5 \
                --save-interval 1000 --eval-interval 1000 --num-updates 100000 \
                --env-name maze --data-path <PATH TO mazes-10-10-100000.txt> \
                --save-dir trained_models/maze_models --log-dir trained_models/maze_models/ \ 
                --map-size 15 --local-map-size 11 --map-scale 1.0 --log-interval 100
```
![Maze training results](https://github.com/srama2512/mapnet-pytorch/blob/master/results/mapnet_maze_results.png)

## Training on Active Vision Dataset
```
cd $MAPNET_PATH
mkdir -p trained_models/avd_models

python train.py --lr 1e-3 --seed 1 --batch-size 64 --num-steps 5 \
                --save-interval 1000 --eval-interval 1000 --num-updates 100000 \
                --env-name avd --data-path <PATH TO avd_trajectories_T20.h5> \
                --save-dir trained_models/avd_models --log-dir trained_models/avd_models \
                --map-size 31 --local-map-size 21 --map-scale 300.0 --log-interval 100
```
![AVD training results](https://github.com/srama2512/mapnet-pytorch/blob/master/results/mapnet_avd_results.png)
