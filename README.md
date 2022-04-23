# MegCup 2022 Team Feedforward

- [MegCup 2022 Team Feedforward](#megcup-2022-team-feedforward)
  - [Environment](#environment)
    - [Conda](#conda)
    - [Docker](#docker)
  - [Usage](#usage)
  - [Contributor](#contributor)


## Environment

### Conda

```shell
$ conda create -f ./env.yaml
```

### Docker

<font color=red>TBD</font>

## Usage

```shell
$ python test.py --data-path DATA_PATH      # The test input data path.
                 --checkpoint CHEKPOINT     # The checkpoint need to be loaded.
                [--batch-size BATCH_SIZE]   # OPTIONAL: Batch size for the dataloader,            DEFAULT: 1
                [--num-workers NUM_WORKERS] # OPTIONAL: Number of workers for the dataloader,     DEFAULT: 0
                [--output PATH]             # OPTIONAL: The path to output the final binary file, DEFAULT: '.'
```

Example:
```shell
$ cp PATH/DATA .
$ python test.py --data-path ./DATA --checkpoint ./feedback_restormer.mge
```

## Contributor

- [Zhen Li](https://github.com/Paper99)
- [Xin Jin](https://github.com/Srameo)
- [Rui-Qi Wu](https://github.com/RQ-Wu)
