<!-- <h2 align="center"> State Estimation in Structural Dynamics through RNN Transfer Learning </h2> -->
<!--
<div align="center"> Shuo Hao<sup>1</sup>, Hong-Wei Li<sup>1,2</sup>, Yi-Qing Ni<sup>1,2</sup>, Wei-Jia Zhang<sup>1</sup>, Lei Yuan<sup>1,2</sup> </div>
<div align="center"> <sup>1</sup>Department of Civil and Environmental Engineering, The Hong Kong Polytechnic University </div>
<div align="center"> <sup>2</sup>National Rail Transit Electrification and Automation Engineering Technology Research Center (Hong Kong Branch) </div> -->

# TL-RNN-StatePrediction-StructDyn

This repository contains the code for the paper "State Estimation in Structural Dynamics through RNN Transfer Learning". Feel free to contact us at [cee-shuo.hao@connect.polyu.hk](mailto:ceecee-shuo.hao@connect.polyu.hk) for any questions or comments.

## Installation

Clone the repository and navigate to the directory.

```bash
git clone https://github.com/shuohaopolyu/TR-RNN-StatePrediction-StructDyn.git
cd TR-RNN-StatePrediction-StructDyn
```

Install the conda package manager from [here](https://docs.conda.io/en/latest/miniconda.html).

```bash
conda create --name trrnn python=3.10.14
conda activate trrnn
```

Install the required packages.

```bash
conda install --yes --file requirements.txt
```

## Usage

The code is organized in the following way:

- `main.py`: Contains the main program for running the experiments.
- `models/`: Contains the RNN models used in the paper.
- `experiments/`: Contains experiments for the state estimation.
- `systems/`: Contains finite element models of the structures and solvers for response simulation.
- `excitations/`: Contains the excitation models for the finite element simulations.
- `dataset/`: Contains the data used in the paper.
- `figures/`: Contains the program for generating the figures in the paper.
- `utils.py`: Contains utility functions for data analysis, processing, and generation.

To run the experiments, use the following command:

```bash
python main.py
```

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Citation

- Shuo Hao, Hong-Wei Li, Yi-Qing Ni, Su-Mei Wang, Wei-Jia Zhang, Lei Yuan. "State Estimation in Structural Dynamics through RNN Transfer Learning".
