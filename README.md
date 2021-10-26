# PSWE: Pooling by Sliced-Wasserstein Embedding (NeurIPS 2021)

PSWE is a permutation-invariant feature aggregation/pooling method based on sliced-Wasserstein distances that generates an output embedding from a set of input features, whose dimensionality does not depend on the input set size.

## Run ModelNet40 experiments

Use the following command to run the complete experiments (for PSWE and other pooling methods) on the `ModelNet40` point cloud dataset:

```shell
python3 ModelNet40_train_test.py
```

## Dependencies

* [PyTorch](https://pytorch.org/)
* [NumPy](https://numpy.org/)
* [torchinterp1d](https://github.com/aliutkus/torchinterp1d)
* [h5py](https://www.h5py.org/)
* [tqdm](https://tqdm.github.io/)

## Citation

Please use the following BibTeX citation if you use this repository in your work:

```
@article{naderializadeh2021_PSWE,
  title={Pooling by Sliced-{Wasserstein} Embedding},
  author={Naderializadeh, Navid and Comer, Joseph F and Andrews, Reed and Hoffmann, Heiko and Kolouri, Soheil},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```
