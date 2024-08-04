# Weapon Target Assignment using Reinforcement Learning and Graph Neural Networks

This project implements a novel approach to the Weapon Target Assignment (WTA) problem using Reinforcement Learning (RL) and Graph Neural Networks (GNN).

## Project Overview

This repository contains the implementation of the research presented in:

**Artificial Intelligence in Combat Decision-making: Weapon Target Assignment via Reinforcement Learning and Graph Neural Networks**

- Authors: Oh Seung Heon, Gun Woong Byueon, Young In Cho, Seungmin Kwon, Jong Hun Woo
- Published: March 19, 2024
- Journal: Authorea Preprints
- Publisher: Authorea

## Repository Structure

- `/agent`: Contains agent-related code
- `/assignment`: Assignment logic implementation
- `/data`: Data files and datasets
- `/environment`: Environment setup and configurations
- `/image`: Image resources (if any)
- `cfg.py`: Configuration file
- `GA.py`: Genetic Algorithm implementation
- `heuristic.py`: Heuristic methods
- `LICENSE.md`: License information
- `README.md`: This file
- `test.py`: Test scripts
- `train.py`: Training scripts
- `utils.py`: Utility functions

## Getting Started
Execute `train.py` for training RL agent

Execute `GA.py` for optimizing heuristic parameters by employing genetic algorithm-based optimizer

Execute `heuristic.py` for optimizing heuristic parameters by employing heuristics


## License

MIT license

## Citation
@article{heon2024artificial,
  title={Artificial Intelligence in Combat Decision-making: Weapon Target Assignment via Reinforcement Learning and Graph Neural Networks},
  author={Heon, Oh Seung and Byueon, Gun Woong and Cho, Young In and Kwon, Seungmin and Woo, Jong Hun},
  journal={Authorea Preprints},
  year={2024},
  publisher={Authorea}
}