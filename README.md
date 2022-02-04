# Heated Gridworlds

<img src="/img/heatmap.gif" style="width: 500px; display: block; margin-left: auto; margin-right: auto;"/>

This supplementary code allows to reproduce all results presented in the paper 
["A Study of First-Passage Time Minimization via Q-Learning in Heated Gridworlds"](https://ieeexplore.ieee.org/document/9622239), IEEE Access 2021.


- Main2D script runs simulations for 2D case with L-shaped obstacle
- Main1D script runs simulations for 1D case with a drift

Call `python3 Main2D --help` or `python3 Main1D --help` to see a full list of arguments.
`Ctrl+C` cancels the execution.

Algorithms available for 2D are simple tabular Q-learning, SARSA, Expected SARSA and Double Q-learning.
Only Q- and Double Q-learning are implemented for 1D case.

Additionally, running Main1D with MC option will test <img src="https://render.githubusercontent.com/render/math?math=\pi_R"> policy

     python3 Main1D.py --algorithm MC --agents 1000


Execution time of scripts usually does not exceed 2-3 min for default settings.

Please, note that main results can be obtained with small number of agents (10-100), but
mean scores will fluctuate strongly due to stochastic nature of the environment. We use 10^3 and 10^4 agents in 2D and 1D simulations respectively.

### Output example

<img src="/img/main1d_output_example.png" style="width: 600px; display: block; margin-left: auto; margin-right: auto;"/>