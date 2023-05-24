# Link Prediction with attention applied on multiple knowledge graph embedding models

This code is the official PyTorch implementation of [Link Prediction with attention applied on multiple knowledge graph embedding models](https://dl.acm.org/doi/pdf/10.1145/3543507.3583358) [1] .
This implementation lies on the KGEmb framework developed by [2] 

## Datasets

Download and pre-process the datasets:

```bash
source datasets/download.sh
python datasets/process.py
```
## Installation

First, create a python 3.7 environment and install dependencies:

```bash
virtualenv -p python3.7 hyp_kg_env
source hyp_kg_env/bin/activate
pip install -r requirements.txt
```

Then, set environment variables and activate your environment:

```bash
source set_env.sh
```

```

## Usage

To train and evaluate a KG embedding model for the link prediction task, use the run.py script:

```bash
usage: run.py [-h] [--dataset {FB15K,WN,WN18RR,FB237,YAGO3-10}]
              [--model {TransE,CP,MurE,RotE,RefE,AttE,RotH,RefH,AttH,ComplEx,RotatE}]
              [--regularizer {N3,N2}] [--reg REG]
              [--optimizer {Adagrad,Adam,SGD,SparseAdam,RSGD,RAdam}]
              [--max_epochs MAX_EPOCHS] [--patience PATIENCE] [--valid VALID]
              [--rank RANK] [--batch_size BATCH_SIZE]
              [--neg_sample_size NEG_SAMPLE_SIZE] [--dropout DROPOUT]
              [--init_size INIT_SIZE] [--learning_rate LEARNING_RATE]
              [--gamma GAMMA] [--bias {constant,learn,none}]
              [--dtype {single,double}] [--double_neg] [--debug] [--multi_c]

Knowledge Graph Embedding

optional arguments:
  -h, --help            show this help message and exit
  --dataset {FB15K,WN,WN18RR,FB237,YAGO3-10}
                        Knowledge Graph dataset
  --model {TransE,CP,MurE,RotE,RefE,AttE,RotH,RefH,AttH,ComplEx,RotatE}
                        Knowledge Graph embedding model
  --regularizer {N3,N2}
                        Regularizer
  --reg REG             Regularization weight
  --optimizer {Adagrad,Adam,SparseAdam}
                        Optimizer
  --max_epochs MAX_EPOCHS
                        Maximum number of epochs to train for
  --patience PATIENCE   Number of epochs before early stopping
  --valid VALID         Number of epochs before validation
  --rank RANK           Embedding dimension
  --batch_size BATCH_SIZE
                        Batch size
  --neg_sample_size NEG_SAMPLE_SIZE
                        Negative sample size, -1 to not use negative sampling
  --dropout DROPOUT     Dropout rate
  --init_size INIT_SIZE
                        Initial embeddings scale
  --learning_rate LEARNING_RATE
                        Learning rate
  --gamma GAMMA         Margin for distance-based losses
  --bias {constant,learn,none}
                        Bias type (none for no bias)
  --dtype {single,double}
                        Machine precision
  --double_neg          Whether to negative sample both head and tail entities
  --debug               Only use 1000 examples for debugging
  --multi_c             Multiple curvatures per relation
```



## Citation

If you use this implementation, please cite the following paper [1]:

```
@inproceedings{10.1145/3543507.3583358, author = {Gregucci, Cosimo and Nayyeri, Mojtaba and Hern\'{a}ndez, Daniel and Staab, Steffen}, title = {Link Prediction with Attention Applied on Multiple Knowledge Graph Embedding Models}, year = {2023}, isbn = {9781450394161}, publisher = {Association for Computing Machinery}, address = {New York, NY, USA}, url = {https://doi.org/10.1145/3543507.3583358}, doi = {10.1145/3543507.3583358}, booktitle = {Proceedings of the ACM Web Conference 2023}, pages = {2600–2610}, numpages = {11}, keywords = {link prediction, geometric integration, ensemble, Knowledge graph embedding}, location = {Austin, TX, USA}, series = {WWW '23} }
```

## References

[1] Cosimo Gregucci, Mojtaba Nayyeri, Daniel Hernández, and Steffen Staab. 2023. Link Prediction with Attention Applied on Multiple Knowledge Graph Embedding Models. In Proceedings of the ACM Web Conference 2023 (WWW '23). Association for Computing Machinery, New York, NY, USA, 2600–2610. https://doi.org/10.1145/3543507.3583358


[2] Chami, Ines, et al. "Low-Dimensional Hyperbolic Knowledge Graph Embeddings."
Annual Meeting of the Association for Computational Linguistics. 2020.


