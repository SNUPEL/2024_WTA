# 2024 WTA Project

## Project Overview
This project appears to be focused on data analysis and simulation for the 2024 World Tennis Association (WTA) events. It likely includes prediction models for tennis match outcomes, player performance analysis, and related tennis analytics.

## Project Structure# 2024_WTA

2024_WTA/
│
├── agent/
│   ├── CGRN.py
│   └── GPO.py
│
├── assignment/
│
├── data/
│   ├── test/
│   └── train/
│       └── input_data.xlsx
│
├── environment/
│   ├── Adapter_Component.py
│   ├── Modeler_Component.py
│   ├── Modeler_Component_test.py
│   ├── Modeler_Component_visualize.py
│   ├── Policy.py
│   ├── Reporter_Component.py
│   ├── Simulation_Component.py
│   └── Simulation_Component_test.py
│
├── output_susceptibility/
│   ├── cfg.py
│   ├── genetic_algorithm_plotting.py
│   ├── genetic_algorithm_plotting_fule5.py
│   ├── genetic_algorithm_plotting_rule7.py
│   ├── greedy_policy.py
│   ├── greedy_policy_wo_openfire_distance.py
│   ├── heuristic.py
│   ├── interval_policy.py
│   ├── interval_policy_wo_openfire_distance.py
│   ├── main_ppo.py
│   ├── random_policy.py
│   ├── setup.py
│   ├── test.py
│   └── utils.py
│
└── .gitignore

## Installation
[Include instructions on how to set up the project environment, including any dependencies]

## Usage
[Provide information on how to run the main scripts and use the project]

## Components
- **Agent**: Contains CGRN and GPO implementations.
- **Environment**: Includes components for modeling, simulation, and reporting.
- **Data**: Stores training and testing datasets.
- **Output Susceptibility**: Contains various policy implementations and analysis tools.

## Contributing
[Include guidelines for contributing to the project, if applicable]

## License
[Specify the license under which this project is released]

## Contact
[Provide contact information for the project maintainer or team]
