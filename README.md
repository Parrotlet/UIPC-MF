# UIPC-MF: User-Item Prototype Connection Matrix Factorization for Explainable Collaborative Filtering

This repository hosts the codefor the paper "UIPC-MF: User-Item Prototype Connection Matrix Factorization for Explainable Collaborative Filterings" by Lei Pan and Von-Wun Soo.

```latex
@article{pan2023uipc,
  title={UIPC-MF: User-Item Prototype Connection Matrix Factorization for Explainable Collaborative Filtering},
  author={Pan, Lei and Soo, Von-Wun},
  journal={arXiv preprint arXiv:2308.07048},
  year={2023}
}
```

## Repository Structure
The code is derived from https://github.com/karapostK/ProtoMF. The structure of the code is the same as https://github.com/karapostK/ProtoMF.

The code is written in Python and relies on [Pytorch](https://pytorch.org/) to compute the model's gradients. We further
use [ray tune](https://www.ray.io/ray-tune) for hyperparameter optimization and [weight and biases](https://wandb.ai/)
for logging, among many cool packages.


## Installation and configuration

### Environment

- Install the environment with
  `conda env create -f uipc_mf.yml`
- Activate the environment with `conda activate uipc_mf`

### Data

All preprocessed data is in /data.

### Configuration

In `utilities/consts.py` you need to set:

- `DATA_PATH`: absolute path to the `./data` folder
- `WANDB_API_KEY`: API key of Weight and Biases, currently the results are only printed there.

## Run

Running start.py will initiate the training process.


By default, `python start.py` runs the hyperparameter optimization for a single seed (check `utilities/consts.py`).


Results and progress can be checked on the Weight&Biases webpage.

## License 
The code in this repository is licensed under the Apache 2.0 License.
