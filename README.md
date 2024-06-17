# Static Data Collection for World Models

This repository contains code to collect random trajectories in the Crafter game and visualize the resulting dataset.
At the moment, the random trajectories follow a fixed hand-crafted policy, leading to a static dataset.

The code is originally from a private git branch leading to the [DIAMOND](https://github.com/eloialonso/diamond) project, made by [Eloi Alonso](https://eloialonso.github.io)\* and [Vincent Micheli](https://vmicheli.github.io)\* and was modified by [Stéphane Nguyen](https://zenchiyu.github.io).

Modifications include
- Removed RL agent and world model. PyTorch models are no longer used in the data collection.
- Renamed:
  - `trainer` to `collector`, as there are no PyTorch models to train.
  - `play` to `explore` and game to `visualizer`, as we can no longer play inside the world model.
- Can now retrieve the Crafter terrain. For a vectorized gymnasium environment `env`, we need to use `env.unwrapped.call("get_map", episode=ep)`.
- Added  `trajectories.py` to visualize some agent trajectories (WIP).

## BibTeX

If you find this code useful, please use the following reference, as it is a work resulting from the [DIAMOND](https://github.com/eloialonso/iris) project:

```
@article{iris2022,
  title={Transformers are Sample Efficient World Models},
  author={Micheli, Vincent and Alonso, Eloi and Fleuret, François},
  journal={arXiv preprint arXiv:2209.00588},
  year={2022}
}
```

## Setup

- Install [PyTorch](https://pytorch.org/get-started/locally/) (torch and torchvision). Code developed with torch==1.11.0 and torchvision==0.12.0.
- Install [other dependencies](requirements.txt): `pip install -r requirements.txt`
- Warning: Atari ROMs will be downloaded with the dependencies, which means that you acknowledge that you have the license to use them.

## Launch a collection run

- For the Crafter game:
```bash
python src/main.py common.device=cuda:0
```

By default, the logs are synced to [weights & biases](https://wandb.ai), set `wandb.mode=disabled` to turn it off.

## Configuration

- All configuration files are located in `config/`, the main configuration file is `config/collector.yaml`.
- The simplest way to customize the configuration is to edit these files directly.
- Please refer to [Hydra](https://github.com/facebookresearch/hydra) for more details regarding configuration management.

## Run folder

Each new run is located at `outputs/YYYY-MM-DD/hh-mm-ss/`. This folder is structured as:

```txt
outputs/YYYY-MM-DD/hh-mm-ss/
├── config
│   └── collector.yaml
├── dataset
│   └── recent
├── main.log
├── src
│   ├── collector.py
│   ├── collect.py
│   ├── data
│   ├── env_loop.py
│   ├── envs
│   ├── explore.py
│   ├── __init__.py
│   ├── main.py
│   ├── utils.py
│   └── visualizer
```
- `dataset`: contains the collected dataset in the format described in the `src/data` directory.
- **From the run folder**:
    - Running `python ./src/explore.py` to visualize the episodes contained in `dataset/recent` (`src/explore.py` was previously called `src/play.py`).
      - Adding `--no-header` to the command will ignore the header.
      - Adding `--fps <number>` to the command will change the PyGame frame-rate.
      - In the visualizer, you can perform the following actions:
        ```
        Actions (macro):

        ESC/q : quit
        ⏎ : reset env
        m : next dataset
        ↑ : next episode
        ↓ : previous episode

        Actions (micro):

        → : next frame
        ← : previous frame
        page up: skip 10 frames
        page down: go back 10 frames
        ```

## Credits

- [https://github.com/eloialonso/iris](https://github.com/eloialonso/iris)
- [https://github.com/eloialonso/diamond](https://github.com/eloialonso/diamond)
- [https://github.com/pytorch/pytorch](https://github.com/pytorch/pytorch)
- [https://github.com/CompVis/taming-transformers](https://github.com/CompVis/taming-transformers)
- [https://github.com/karpathy/minGPT](https://github.com/karpathy/minGPT)
- [https://github.com/google-research/rliable](https://github.com/google-research/rliable)
