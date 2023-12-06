# Meta Graph Learning for Long-tail Recommendation

This repository is an implementation of Meta Graph Learning (MGL) for long-tail recommendation systems, forked from [the official MGL repository](https://github.com/weicy15/MGL).

## Requirements

Before you begin, ensure you have the following requirements installed:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

If you want CUDA support, please follow the instruction of PyTorch documentation: [Start Locally \| PyTorch](https://pytorch.org/get-started/locally/)

After installing the above requirements, please install pytorch_sparse by following the documentation provided here: [rusty1s/pytorch_sparse](https://github.com/rusty1s/pytorch_sparse). This library is essential for optimized autograd sparse matrix operations used in MGL.

Example:

```bash
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.1+cpu.html
```

## Data Processing

To prepare the data for model training, run the following command:

```bash
python data_process.py
```

This script processes the input data to make it suitable for training the MGL model. It formats and structures the data, ensuring compatibility with the model's requirements.

## Training

To train the model as described in the associated paper, use the following command:

```bash
python train.py
```

Output: The training process will generate a file named "model.tar", which contains the trained model weights and configuration settings. This file can be used for deploying the model in a recommendation system.
