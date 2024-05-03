# Transfer learning of recurrent neural networks for state estimation in structural dynamics

This is the code repository for the paper "Transfer learning of recurrent neural networks for state estimation in structural dynamics" by Shuo HAO. It contains the code necessary to reproduce the numerical and experimental results presented in the paper. Now the code is under development and will be released soon.

## Install

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
pip install -r requirements.txt
```

## Usage

The code is organized in the following way:

- `data/`: Contains the data used in the paper.
