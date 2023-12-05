# Meta Graph Learning for Long-tail Recommendation

This repository is the official implementation of MGL.

## Requirements

To install requirements:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Then, instrall `pytorch_sparse` following the documentation: [rusty1s/pytorch\_sparse: PyTorch Extension Library of Optimized Autograd Sparse Matrix Operations](https://github.com/rusty1s/pytorch_sparse)

## Data Process

To prepare the data for the model training:

```bash
python data_process.py
```

## Training

To train the model(s) in the paper:

```bash
python train.py
```

> Output: the file "model.tar"
