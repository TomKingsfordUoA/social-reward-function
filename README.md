# Social Robotics Reward

## Introduction

The intent of this library is to provide a standardised framework and reference implementation for online, dense
recognition of human emotions through multiple sensor modalities, and mapping these onto a real-valued
number to form a reward function.

This reward function may then be used for one or both of:
1. Evaluating and comparing the behviour of social robotics
2. Performing reinforcement learning in social robotics

For more information, please refer to **(insert citation)**

## Getting Started

To get started, run:

`git clone git@github.com:TomKingsfordUoA/social-robotics-reward.git`
`cd social-robotics-reward`
`pip install .`
`srr.py -h`

## To Do

* Audio has a reward even when nothing's being said.
* Write tests (unit and system)
* Expose reward externally (AMQP? ROS?)
* Introduce touch modality

## Citation

If you use this work, please cite:
**(insert citation)**