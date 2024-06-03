<h2 align="center"> State Estimation in Structural Dynamics through RNN Transfer Learning </h2>

<div align="center"> Shuo Hao<sup>1</sup>, Hong-Wei Li<sup>1,2</sup>, Yi-Qing Ni<sup>1,2</sup>, Wei-Jia Zhang<sup>1</sup>, Lei Yuan<sup>1,2</sup> </div>
<div align="center"> <sup>1</sup>Department of Civil and Environmental Engineering, The Hong Kong Polytechnic University </div>
<div align="center"> <sup>2</sup>National Rail Transit Electrification and Automation Engineering Technology Research Center (Hong Kong Branch) </div>

## Introduction

Model construction for state estimation is a pivotal concern in structural dynamics, driven by the need for effective control and health monitoring of structures. Traditional research has relied heavily on developing finite element models. However, the limited capability of finite element models to simulate actual structures and the complexity of the environments in which these structures operate pose significant challenges to the accuracy of model-based state estimations. This paper introduces a novel approach that leverages recurrent neural network (RNN)-based transfer learning to construct state estimation models, aiming to enhance the accuracy of state estimations for actual structures. A calibrated finite element model is used to generate extensive response data under synthetic excitation. This data is then processed and integrated to train an RNN model specifically designed for state estimation. Considering the diverse sensors involved in real-world structure monitoring, this study innovatively utilizes the collected data in a dual-purpose manner. A portion of this data serves as input for the RNN model, while the complete dataset facilitates the transfer learning process for the RNN model. This strategy enables the RNN model to adapt to real-structure state prediction. Unlike traditional deep learning transfer learning approaches that typically adjust parameters targeting the output layers, the proposed method fine-tunes parameters within the RNN cells at the network’s front end, ensuring the training converges effectively. Numerical and experimental studies demonstrate that the RNN, trained via transfer learning and integrating both model-generated and actual measurement data, achieves significantly higher accuracy under comparable data acquisition conditions than state estimation models based solely on finite element models.

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
