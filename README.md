# PSWE: Pooling by Sliced-Wasserstein Embedding (NeurIPS 2021)

PSWE is a permutation-invariant feature aggregation/pooling method based on sliced-Wasserstein distances that generates an output embedding from a set of input features, whose dimensionality does not depend on the input set size.

## Run ModelNet40 experiments

Use the following command to run the complete experiments (for PSWE and other pooling methods) on the `ModelNet40` point cloud dataset:

```shell
python3 ModelNet40_train_test.py
```
