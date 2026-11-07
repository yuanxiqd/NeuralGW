# NeuralGW

[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-blue)](https://neurips.cc/virtual/2026/loc/sydney/poster/151656)

Code for **Bridging Graph Worlds: Neural Approximation of Gromov-Wasserstein Distances**
(Dong Qiao, Chris Ding, Jicong Fan; NeurIPS 2026).

![Architecture of NeuralGW. The GIN and the Transformer are shared by the two input graphs; the Transformer attends over the nodes of both graphs jointly.](assets/neuralgw.png)

NeuralGW predicts the Gromov-Wasserstein (GW) distance between two graphs. A shared GIN-Transformer
encoder with SVD positional encodings embeds the nodes of both graphs. A multi-kernel Maximum Mean
Discrepancy (MMD) layer compares the two sets of node embeddings, and a multilayer perceptron maps the
MMD values to the predicted distance. Once trained, it needs no iterative optimization at test time.

## Setup

```bash
pip install -r requirements.txt
```

`requirements.txt` pins the versions the code was tested with (Python 3.12). Install the PyTorch and
PyTorch Geometric builds that match your CUDA version; without a GPU, the code runs on the CPU.
`grakel` (graph-kernel baselines) and `matplotlib` (plots) are needed only by the scripts that use them.

## Data and reference GW values

The datasets are TU datasets, which PyTorch Geometric downloads into `$NEURALGW_DATA_ROOT/TUDataset`
(default `./data`). The cross-domain and clustering runs on the 300–350-node subsets use a separate
root, `./data_300_350`, which their scripts set unless `NEURALGW_DATA_ROOT` is already set.

The reference GW distances are not stored in the repository. On first use,
`models/_utils_Processing.py` computes the missing ones and caches them under
`$NEURALGW_DATA_ROOT/precomputed/`. It uses POT's conditional-gradient solver
(`ot.gromov.gromov_wasserstein2` on the cost A + I, uniform marginals, `tol_rel = tol_abs = 1e-4`).
This takes seconds for MUTAG and hours for the larger REDDIT-BINARY subsets. `scripts/prep_collab_bal.py`,
`scripts/prep_bio_bal.py` and `scripts/prep_synth_gw.py` precompute the values for the clustering
subsets and the synthetic graphs. The values depend on the POT version, so pin POT when you need the
paper's exact reference values.

## Running

**In-domain:** train and test on one dataset (an 8:2 split of its graphs by default).

```bash
python main_indomain.py --ds_name MUTAG --seed 0
```

`python main_indomain.py --help` lists the other options: test ratio, positional encoding, cost
matrix and GPU.

**Cross-domain:** train on some datasets and test on another. `test.sh` wraps `main_cdomain.py` and
runs the job in the background. It logs to `logs_cdomain_test/` and saves the predicted distance
matrix of the test set to `results/clustering/{trainsets}_to_{testset}_seed{seed}_distance_matrix.npy`.

```bash
bash test.sh --trainsets PTC_FM BZR --testset MUTAG --seed 0 --cuda 0
```

`main_cdomain.py` treats its last `--ds_name*` argument as the test set; `test.sh` orders them for you.

**Experiment launchers:** `indomain/` and `cdomain/` hold the launchers for the experiment variants
(backbones, splits, transfer settings, large graphs). Run them from the repository root, for example
`bash indomain/main_indomain.sh`.

**Clustering:**

- `python scripts/spectral_cluster_cross_domain.py --trainsets PTC_FM BZR --testset MUTAG` runs spectral
  clustering on the cross-domain matrices in `results/clustering/` and reports ACC, NMI, ARI and RI,
  averaged over the seeds.
- `bash scripts/run_clustering_table.sh` queues the cross-domain runs for the clustering table.
  `bash scripts/run_cdomain_social.sh SRC TGT` trains on one dataset and predicts the distances on another.
- `scripts/run_indomain_target.py` trains in-domain and saves a target's full predicted matrix.
- `scripts/cluster_eval.py` scores distance matrices by spectral clustering, and
  `scripts/compute_kernels.py` computes the graph-kernel baselines with GraKeL.

## Configuration

`configs/main.json` holds the model and training defaults of the code used for the paper:
- 9 MMD kernels;
- learning rate 1e-4, batch size 4, no weight decay;
- up to 200 epochs, evaluating every 10 epochs and stopping after 3 evaluations without improvement.

The code also contains options added after the paper's experiments. They are all off by default and
were not used for its results:
- post-hoc calibration of the predictions (`NEURALGW_POSTHOC_CALIB`);
- positional-encoding normalization (`NEURALGW_PE_NORM`);
- adaptive MMD bandwidths (`NEURALGW_MMD_BW_ESTIMATOR`, `NEURALGW_MMD_BW_SCALE`);
- graph meta-features (`NEURALGW_META_INJECTION`);
- MLP-head variants (`head_use_layernorm` and `head_dropout_p` in the config, `NEURALGW_HEAD_OUTPUT_RELU`).

## Other tools

- `test_generating_large_graphs.py` generates random graphs with thousands of nodes and computes their
  GW distances (`--num_graphs`, `--n_min`, `--n_max`, `--avg_degree`, `--tol_rel`, `--tol_abs`,
  `--max_iter`).
- `test_generating_directed_graphs.py` builds directed variants of MUTAG and REDDIT-BINARY
  (`--source_name`, `--orientation`, `--tu_root`, `--verify`). To use them, pass `MUTAG_DIR` or
  `REDDIT-BINARY_DIR` as a dataset name.
- `check_triangle_inequality.py` counts the triangle-inequality violations in a saved distance matrix.
- `test_time.py` times the GW solver, its entropic and sampled variants, and NeuralGW's inference.
- `test_instability.py` compares GW values from different initializations on MUTAG.
- `plot_mmd_curve.py` plots MMD values across datasets and model configurations.

## Citation

```bibtex
@inproceedings{qiao2026bridging,
  title     = {Bridging Graph Worlds: Neural Approximation of {Gromov-Wasserstein} Distances},
  author    = {Qiao, Dong and Ding, Chris and Fan, Jicong},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

MIT (see `LICENSE`).
