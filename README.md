notes
```
python cli.py vuln-evaluate --model-path e3e17252.ep350 --model-name

python cli.py evaluate --model-path e3e17252.ep350 --model-name GraphConvInstanceGlobalMaxSmallSoftMaxAggrEdge --dataset-path eu4 --eval-prefix eu4_eval --requires-edge-feats
```


# Know your neighborhood (KYN)

This repo contains the training and evaluation code for the paper titled "Know Your Neighborhood: General and Zero-Shot Capable Binary Function Search Powered by Call Graphlets".

>Binary code similarity detection is an important
problem with applications in areas such as malware analysis,
vulnerability research and license violation detection. This
paper proposes a novel graph neural network architecture
combined with a novel graph data representation called call
graphlets. A call graphlet encodes the neighborhood around
each function in a binary executable, capturing the local
and global context through a series of statistical features. A
specialized graph neural network model operates on this graph
representation, learning to map it to a feature vector that
encodes semantic binary code similarities using deep-metric
learning.
>
> The proposed approach is evaluated across five distinct
datasets covering different architectures, compiler tool chains,
and optimization levels. Experimental results show that the
combination of call graphlets and the novel graph neural
network architecture achieves comparable or state-of-the-art
performance compared to baseline techniques across cross- architecture, mono-architecture and zero shot tasks. In addition,
our proposed approach also performs well when evaluated
against an out-of-domain function inlining task. The work
provides a general and effective graph neural network-based
solution for conducting binary code similarity detection.

# Environment Setup

```bash
uv venv --python 3.10
uv sync
```

# Reproducing Results

Download and extract the datasets which can downloaded from Google Drive [here](https://drive.google.com/file/d/1zcTsj_HIwQGmFBAx5s65PizbnPRCqDxJ/view?usp=sharing)

## Search Pool Eval

The `run_searhpool_eval.py` script has been provided to make this straightforward.

Using the `test/cisco-d1-test-callers-edge-between` evaluation as an example, the results can re-produced by running:
```bash
python run_searchpool_eval.py --model paper-artefacts/models/best-model/8c913a81.ep350 \
  --dataset datasets/test/cisco-d1-test-callers-edge-between \
  --prefix cisco-d1-test \
  --pools 500 \
  --pool-sizes 100 \
  --device cuda
```

Change the `--dataset` to any of the evaluation dataset within `datasets/` to re-create the other evals

## Vuln Search Eval

The `run_vuln_eval.py` script has been provided to make this too straightforward. Download the data from Google Drive [here](https://drive.google.com/file/d/1Te1ESPT6dgUJ-otU0RUCyhuW6Z7K-2_N/view?usp=drive_link)
and unzip.

The results for the NETGEAR device can then be reproduced with the following command:
```bash
python run_vuln_eval.py --model paper-artefacts/models/best-model/8c913a81.ep350 \
  --data-root cgs/ \
  --target netgear
```

The results for the TP-LINK device can then be reproduced with the following command:
```bash
python run_vuln_eval.py --model paper-artefacts/models/best-model/8c913a81.ep350 \
  --data-root cgs/ \
  --target tplink
```

# Training and Testing on custom data

The `cli.py` script has been provided to demonstrate how you can train and evaluator
your own models using the KYN model and associated code base. In addition to this,
the `generate_datasets.py` script has been provided to generate new datasets. 
> It is worth noting that the `generate_datasets.py` script will not create identical splits
> to those that are downloadable. This is due to the seed being set *after* the provided 
> datasets were made.

