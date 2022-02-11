
# Edge-augmented Graph Transformer

## News

* 02/10/2022 - Updated results on the GNN benchmarking datasets by [Dwivedi et al.](https://arxiv.org/abs/2003.00982).
* 02/08/2022 - A new simplified tensorflow implementation which supports random masking, attention dropout and centrality scalers. For PCQM4M datasets please view the PyTorch implementation at: https://github.com/shamim-hussain/egt_pytorch.
* 01/26/2022 - The **PyTorch** implementation of EGT is now available at: https://github.com/shamim-hussain/egt_pytorch. The results and hyperparameters for PCQM4M, PCQM4Mv2, OGBG-MolPCBA, OGBG-MolHIV are also available.

## Introduction

This is the official implementation of the **Edge-augmented Graph Transformer (EGT)** as described in https://arxiv.org/abs/2108.03348, which augments the Transformer architecture with residual edge channels. The resultant architecture can directly process graph-structured data and acheives good results on supervised graph-learning tasks as presented by [Dwivedi et al.](https://arxiv.org/abs/2003.00982). It also achieves good performance on the large-scale [PCQM4M-LSC](https://arxiv.org/abs/2103.09430) (`0.1224 MAE` on val) and [PCQM4Mv2-LSC](https://arxiv.org/abs/2103.09430) (`0.0872 MAE` on test-dev) dataset. EGT beats convolutional/message-passing graph neural networks on a wide range of supervised tasks and thus demonstrates that convolutional aggregation is not an essential inductive bias for graphs.

## Results

Dataset       | #params | Metric         | Test Result    |
--------------|---------|----------------|----------------|
PATTERN       | ≈100K   | Accuracy (%)   | 86.816 ± 0.027 |
PATTERN       | ≈500K   | Accuracy (%)   | 86.821 ± 0.020 |
CLUSTER       | ≈500K   | Accuracy (%)   | 79.232 ± 0.348 |
MNIST         | ≈500K   | Accuracy (%)   | 98.173 ± 0.087 |
CIFAR10       | ≈500K   | Accuracy (%)   | 68.702 ± 0.409 |
TSP           | ≈100K   | F1             |  0.822 ± 0.000 |
TSP           | ≈500K   | F1             |  0.853 ± 0.001 |
ZINC          | ≈100K   | MAE            |  0.143 ± 0.011 |
ZINC          | ≈500K   | MAE            |  0.108 ± 0.009 |

## Requirements

* `python >= 3.7`
* `tensorflow >= 2.1.0`
* `h5py >= 2.8.0`
* `numpy >= 1.18.4`
* `scikit-learn >= 0.22.1`

## Download the Datasets

For our experiments, we converted the datasets to HDF5 format for the convenience of using them without any specific library. Only the `h5py` library is required. The GNN Benchmarking Datasets by Dwivedi et al.can be downloaded from https://zenodo.org/record/5500978. Or you can simply run the provided bash script `download_medium_scale_datasets.sh`. The default location of the datasets is the *datasets* directory.

## Run Training and Evaluations

You must create a `JSON` config file containing the configuration of a model, its training and evaluation configs (configurations). The same config file is used to do both training and evaluations.

* To run training: ```python run_training.py <config_file.json>```
* To end training (prematurely): ```python end_training.py <config_file.json>```
* To perform evaluations: ```python do_evaluations.py <config_file.json>```

Config files for the main results presented in the paper are contained in the *configs/main* directory, whereas configurations for the ablation study are contained in the *configs/ablation* directory. The paths and names of the files are self-explanatory.

### More About Training and Evaluations

Once the training is started a model folder will be created in the *models* directory, under the specified dataset name. This folder will contain a copy of the input config file, for the convenience of resuming training/evaluation. Also, it will contain a config.json which will contain all configs, including unspecified default values, used for the training. Training will be checkpointed per epoch. In case of any interruption you can resume training by running the *run_training.py* with the config.json file again.

In case you wish to finalize training midway, just stop training and run `end_training.py` script with the config.json file to save the model weights.

After training, you can run the `do_evaluations.py` script with the same config file to perform evaluations. Alongside being printed to stdout, results will be saved in the *predictions* directory, under the model directory.

### Config File

The config file can contain many different configurations, however, the only **required** configuration is `scheme`, which specifies the training scheme. If the other configurations are not specified, a default value will be assumed for them. Here are some of the commonly used configurations:

`scheme`: Used to specify the training scheme. It has a format <dataset_name>.<positional_encoding>. For example: *cifar10.svd* or *zinc.eig*. If no encoding is to be used it can be something like *pcqm4m.mat*. For a full list you can explore the *lib/training/schemes* directory.

`dataset_path`: If the datasets are contained in the default location in the *datasets* directory, this config need not be specified. Otherwise you have to point it towards the *<dataset_name>.h5* file.

`model_name`: Serves as an identifier for the model, also specifies default path of the model directory, weight files etc.

`save_path`: The training process will create a model directory containing the logs, checkpoints, configs, model summary and predictions/evaluations. By default it creates a folder at *models/<dataset_name>/<model_name>* but it can be changed via this config.

`cache_dir`: During first time of training/evaluation the data will be cached to a tensorflow cache format. Default path is *data_cache/<dataset_name>/<positional_encoding>*. But it can be changed via this config.

`distributed`: In a multi-gpu setting you can set it to True, for distributed training.

`batch_size`: Batch size.

`num_epochs`: Maximum Number of epochs.

`initial_lr`: Initial learning rate. In case of warmup it is the maximum learning rate.

`rlr_factor`: Reduce LR on plateau factor. Setting it to a value >= 1.0 turns off Reduce LR.

`rlr_patience`: Reduce LR patience, i.e. the number of epochs after which LR is reduced if validation loss doesn't improve.

`min_lr_factor`: The factor by which the minimum LR is smaller, of the initial LR. Default is 0.01.

`model_height`: The number of layers *L*.

`model_width`: The dimensionality of the node channels *d_h*.

`edge_width`: The dimensionality of the edge channels *d_e*.

`num_heads`: The number of attention heads. Default is 8.

`ffn_multiplier`: FFN multiplier for both channels. Default is 2.0 .

`virtual_nodes`: number of virtual nodes. 0 (default) would result in global average pooling being used instead of virtual nodes.

`upto_hop`: Clipping value of the input distance matrix. A value of 1 (default) would result in adjacency matrix being used as input structural matrix.

`mlp_layers`: Dimensionality of the final MLP layers, specified as a list of factors with respect to *d_h*. Default is [0.5, 0.25].

`gate_attention`: Set this to False to get the ungated EGT variant (EGT-U).

`dropout`: Dropout rate for both channels. Default is 0.

`edge_dropout`: If specified, applies a different dropout rate to the edge channels.

`edge_channel_type`: Used to create ablated variants of EGT. A value of "residual" (default) implies pure/full EGT. "constrained" implies EGT-constrained. "bias" implies EGT-simple.

`warmup_steps`: If specified, performs a linear learning rate warmup for the specified number of gradient update steps.

`total_steps`: If specified, performs a cosine annealing after warmup, so that the model is trained for the specified number of steps.

**[For SVD-based encodings]:**

`use_svd`: Turning this off (False) would result in no positional encoding being used.

`sel_svd_features`: Rank of the SVD encodings *r*.

`random_neg`: Augment SVD encodings by random negation.

**[For Eigenvectors encodings]:**

`use_eig`: Turning this off (False) would result in no positional encoding being used.

`sel_eig_features`: Number of eigen vectors.

**[For Distance prediction Objective (DO)]:**

`distance_target`: Predict distance up to the specified hop, *nu*.

`distance_loss`: Factor by which to multiply the distance prediction loss, *kappa*.


## Creation of the HDF5 Datasets from Scratch

We included a Jupyter notebook to demonstrate how the HDF5 datasets are created - `create_hdf_benchmarking_datasets.ipynb`. You will need `pytorch`, `ogb==1.1.1` and `dgl==0.4.2` libraries to run the notebook. The notebook is also runnable on Google Colaboratory.

## Python Environment

The Anaconda environment in which our experiments were conducted is specified in the `environment.yml` file.


## Citation

Please cite the following paper if you find the code useful:
```
@article{hussain2021edge,
  title={Edge-augmented Graph Transformers: Global Self-attention is Enough for Graphs},
  author={Hussain, Md Shamim and Zaki, Mohammed J and Subramanian, Dharmashankar},
  journal={arXiv preprint arXiv:2108.03348},
  year={2021}
}
```
