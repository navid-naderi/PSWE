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
@inproceedings{naderializadeh2021PSWE,
  title={Pooling by Sliced-{Wasserstein} Embedding},
  author={Navid Naderializadeh and Joseph F. Comer and Reed W Andrews and Heiko Hoffmann and Soheil Kolouri},
  booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
  year={2021},
  url={https://openreview.net/forum?id=1z2T01DKEaE}
}
```
