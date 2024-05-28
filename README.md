# TR-RNN-StatePrediction-StructDyn

## State Estimation in Structural Dynamics through RNN Transfer Learning

This is the code repository for the paper "State Estimation in Structural Dynamics through RNN Transfer Learning" by Shuo HAO. It contains the code necessary to reproduce the numerical and experimental results presented in the paper. Now the code is under development and will be released soon.

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

- `dataset/`: Contains the data used in the paper.
- `excitations/`: Contains the excitation models for the finite element simulations.
- `exps/`: Contains experiments for the state estimation.

- `models/`: Contains the RNN models used in the paper.
- `systems/`: Contains finite element models of the structures.
