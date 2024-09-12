# **TimeMixer-TFM**: Tunning the Hyperparameters of the Multivariate Time Series Forecasting Deep Neural Network TimeMixer

[TimeMixer-TFM](https://github.com/antmartinez47/TimeMixer-TFM) is a forked version of the [official PyTorch implementation](https://github.com/kwuking/TimeMixer) of the Deep Neural Network [TimeMixer](https://arxiv.org/abs/2405.14616), a fully MLP-based architecture **Past-Decomposable-Mixing (PDM)** and **Future-Multipredictor-Mixing (FMM)** blocks that achieves state-of-the-art performance on both long and short-term time series forecasting benchmarks (first published on arXiv on 23/05/2024). The primary motivation behind this repository is to build a robust Hyperparameter Optimization (HPO) pipeline for tuning the [TimeMixer](https://arxiv.org/abs/2405.14616)'s hyperparameters, aiming to improve the performance reported by the original authors for the task of (Multivariate) Time Series Forecasting (TSF). This initiative arose as part of the objectives set during the development of the Final Master's Thesis (TFM) for the Master's Degree in Data Science at the University of Valencia (*Máster en Ciencia de Datos, Escuela Técnica Superior de Ingeniería, Universidad de Valencia*).

The original source code has been modified, extended, and enhanced to improve efficiency and compatibility with the selected hyperparameter tuning utilities. [TimeMixer-TFM](https://github.com/antmartinez47/TimeMixer-TFM) provides several Python scripts and packages that implement the HPO routine using the [`raytune`](https://docs.ray.io/en/master/tune/index.html) and [`smac`](https://automl.github.io/SMAC3/main/) libraries, supporting a wide range of search algorithms (random search, Bayesian optimization, TPE, BOHB, etc.). The modifications maintain the coding style of the official implementation, utilizing `argparse` for user interaction via the Linux shell. This feature offers greater flexibility in designing customized HPO experiments and facilitates integration with similar architectures and tasks, as it allows users to specify the search space, search algorithm, tuning process variables, and other global parameters of the training process.

[TimeMixer-TFM](https://github.com/antmartinez47/TimeMixer-TFM) was used during the development of the TFM to generate the experimental results related to the [TimeMixer](https://arxiv.org/abs/2405.14616) architecture, including the reproduction of the results reported by the authors in the original paper and the HPO experiment with ETTh1 dataset. By using `argparse` and the command line to execute MTSF training and tuning experiments, [TimeMixer-TFM](https://github.com/antmartinez47/TimeMixer-TFM) allows for direct optimization of any search space for [TimeMixer](https://arxiv.org/abs/2405.14616) hyperparameters, with support for probability distributions and conditional dependencies. Furthermore, thanks to its modular coding style, which preserves the syntax of the original implementation, [TimeMixer-TFM](https://github.com/antmartinez47/TimeMixer-TFM) can be easily adapted to work with different datasets, architectures, and tasks.

![timemixer](tfm_imgs/timemixer_overview.png)
![components](tfm_imgs/timemixer_components.png)
**Figure 1**: Overview of [TimeMixer](https://arxiv.org/abs/2405.14616) Deep Neural Network. [TimeMixer](https://arxiv.org/abs/2405.14616) utilizes an MLP-based encoder-decoder architecture, where PDM blocks are applied sequentially over the multiscale time series in order to form the input for the decoder (mixing block), which is subsequently used to generate the forecasts.

## **Table of Contents**

- [Requirements](#requirements)
- [Structure](#structure)
- [How to Use](#how-to-use)
- [TimeMixer-ETTh1 Experiment](#timemixer-etth1-experiment)
- [References](#references)

## **Requirements**

* Linux (Ubuntu 22.04+)
* NVIDIA-compatible GPU (and drivers)
* miniconda3/anaconda3
* GCC compiler

## **Structure**

TimeMixer-TFM is built on top of a fork of the official repository. The original files are preserved, with some minor alterations. Additionally, the following files and folders were specifically added to run and analyze training and tuning experiments for Time Series Forecasting (TSF) problems:

- `checkpoints`: CSV files summarizing the progress of the experiments.
- `conda_config_files`: YAML configuration files specifying the conda environments.
- `tfm_imgs`: Images used in the thesis to summarize the results obtained.
- `tfm_scripts`: Shell scripts and text logs of the experiments.
- `tfm_utils`: A Python package containing various modules specifically implemented for the TFM. These modules address different needs encountered during development, such as model checkpointing, early stopping callbacks, utilities for plotting the results of HPO experiments, and enhanced training and evaluation utilities extracted from the original implementation.
- `build_summary_imgs.ipynb`: A Jupyter notebook used to create dataframes for comparing default and optimized configurations, and to verify the correct reproduction of the paper’s results.
- `download_data.py`: Downloads and extracts datasets used in the paper’s benchmarks and TFM experiments.
- **`plot_tune_results.py`**: Generates plots representing the progress and summary of the HPO experiments.
- **`smac_softs.py`**: Executes an HPO experiment on SOFTS using the SMAC package.
- **`train_softs.py`**: Trains a SOFTS model.
- **`tune_softs.py`**: Runs an HPO experiment on SOFTS using the Ray Tune package.
- `TimeMixer-ETTh1.sh`: Runs all scripts used to generate the results presented in the TFM, enabling full reproduction of the SOFTS-ETTh2 experiment.
- `search_space.md`: Explanation of the process of constructing the search space for SOFTS-ETTh2 HPO experiment.

The files marked in **bold** utilize `argparse`, allowing users to adapt their behavior to specific problems. Furthermore, all functions and modules of these new files are well-documented to help users easily adapt and expand the code for their own use cases.

## **How to use**

Clone [TimeMixer-TFM](https://github.com/antmartinez47/TimeMixer-TFM) repository in the current working directory:

```bash
git clone https://github.com/antmartinez47/TimeMixer-TFM
cd TimeMixer-TFM
```

Install conda environments:

```bash
conda create -f conda_config_files/environment-raytune.yml
conda create -f conda_config_files/environment-smac.yml
```

TimeMixer-TFM is designed to be used as an extension of the original repository, maintaining the same `argparse` style for user interaction. Scripts are invoked through a shell, with arguments passed using the `--argument value` syntax. 

To train a TimeMixer model, the dataset must meet several requirements. In particular, it should be a single CSV file where each column contains a time series, along with an additional column named *date* for the timestamps.

The following is an example of using TimeMixer-TFM to train a TimeMixer model for the task of Multivariate Time Series Forecasting on the ETTh1 dataset (7 time series):

```bash
conda activate py3.11-timemixer-raytune
python3 download_data.py # Download and extract datasets
python3 train_timemixer \
    --data ETTh1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --features M \
    --save_dir ./checkpoints/baseline/ETTh1_96_192 --seq_len 96 --pred_len 192 \
    --enc_in 7 ---c_out 7 --d_model 16 --d_ff 32 --e_layers 2 \
    --down_sampling_layers 3 --down_sampling_window 2 --down_sampling_method avg \
    --dropout 0.0 --num_workers 4 --train_epochs 10 --batch_size 128 --train_prop 1.0 \
    --patience 0 --delta 0.0 --learning_rate 0.01 --loss MSE --seed 123;
```

* Dataset location: ./dataset/ETT-small/ETTh1.csv `--root_path ./dataset/ETT-small/` `--data_path ETTh2.csv`
* Type of forecasting: M (Multivariate) `--features M`
* Checkpoints directory: ./checkpoints/baseline/ETTh2_96_192 `--save_dir ./checkpoints/baseline/ETTh1_96_192`
* Num timesteps input window (lookback length): 96 `--seq_len 96`
* Num timesteps output window (horizon length): 192 `--pred_len 192`
* Num input channels encoder (the number of input timeseries): 7 `--enc_in 7`
* Num output channels (the number of output timeseries; for multivariate forecasting should be equal to the number of input series): 7 `--c_out 7`
* Model dimension (number of channels of each multiscale time series): 16 `--d_model 16`
* Hidden size of MLP submodules (feed forward dimension): 32 `--d_ff 32`
* Number of PDM blocks in the encoder: 2 `--e_layers 2`
* Number of scales (multiscale time series): 3 `--down_sampling_layers 3`
* Window size of downsampling operation: 2 `--down_sampling_window` 3
* Downsampling method: moving average `--down_sampling_method avg`
* Dropout rate: 0.0 `--dropout 0.0`
* Number of workers for multithreading during data loading: 4 `--num_workers 4`
* Number of maximum training epochs: 10 `--training_epochs 10`
* Batch size: 128 `--batch_size 128`
* Proportion of training set that is used to optimize model weights: 1.0 `--train_prop 1.0`
* Patience for Early Stopping callback: 0 (no early stopping) `--patience 0`
* Delta for Early Stopping callback: 0.0 `--delta 0.0`
* Initial learning rate of Adam optimizer: 0.01 `--learning_rate 0.01`
* Loss function: MSE (Mean Squared Error) `--loss MSE`
* Random seed: 123 `--seed 123`

And the following block of code tunes the hyperparameters of TimeMixer for the task of MTSF on the ETTh1 dataset:

```bash
conda activate py3.10-timemixer-smac
python3 smac_timemixer \
    --data ETTh1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --features M \
    --seq_len 96 --pred_len 192 --enc_in 7 ---c_out 7 \
    --smac_min_budget 1 --smac_eta 3 --smac_incumbent_selection "highest_budget" --smac_n_init_configs 150 \
    --smac_storage_path ./checkpoints/hptunning/smac/ --smac_experiment_name ETTh1_96_${horizon} \
    --smac_n_trials 1500 --smac_trial_walltime_limit 100 --smac_time_budget_s 14400 \
    --smac_n_workers 1 \
    --smac_param_space "{
        \"batch_size\": [\"choice\", [16, 32, 64, 128]], \
        \"learning_rate\": [\"loguniform\", [0.0005, 0.012]], \
        \"down_sampling_method\": [\"choice\", [\"avg\", \"conv\"]], \
        \"d_model\": [\"choice\", [8, 16, 32, 64, 128, 256, 512]], \
        \"alpha_d_ff\": [\"choice\", [2, 3, 4]], \
        \"decomp_method\": [\"choice\", [[\"moving_avg\", \"moving_avg\", [15, 25, 35, 55, 75]], [\"dft_decomp\"]]], \
        \"e_layers\": [\"choice\", [1, 2, 3, 4]], \
        \"dropout\": [\"normal\", [0.05, 0.15, 0.1, 0.025]]
    }" \
   --seed 123;
```
* Search Algorithm: SMAC3 (defined by script)
* Configuration of the search algorithm: `--smac_min_budget 1` `--smac_eta 3` `--smac_incumbent_selection "highest_budget"` `--smac_n_init_configs 150`
* Checkpoints and logs save directory: ./checkpoints/hptunning/smac/ETTh1_96_192 `--smac_storage_path ./checkpoints/hptunning/smac` `--smac_experiment_name ETTh2_96_192`
* Maximum number of configurations: 1500 (training stops after reaching this value) `--smac_n_trials 1500`
* Stopping conditions:
  * Global condition: Tuning stops after reaching 14400 seconds (6 hours) `smac_time_budget_s 14400`
  * Local condition: Trial evaluation stops after reaching 100 seconds `smac_trial_walltime_limit 100`
* Maximum concurrent trials (single GPU setting is assumed): 1 `--smac_n_workers 1`
* A Search space (`--smac_search_space`) with conditional dependencies consisting on the following hyperparameters (every argparse argument of `train_timemixer.py` can be tuned):
  * Batch size: categorical distribution with values [16, 32, 64, 128]
  * Learning rate: Loguniform distribution in the range [0.0005, 0.012]
  * Downsampling method: categorical distribution with values [*avg*, *conv*]
  * *d_model*: categorical distribution with values [8, 16, 32, 64, 128, 256, 512]
  * MLP hidden size as a multiple of *d_model* (*alpha_d_ff*): categorical distribution with values [2, 3, 4]
  * Series decomposition method (*decomp_method*): categorical distribution with values [*moving_avg*, *dft_decomp*] 
    * If sampled method is *moving_avg* another hyperparameter named *moving_avg* must be sampled from a categorical distribution with values [15,25,35,55,75] (i.e *moving_avg* exists conditionally on *decomp_method*)
  * Number of PDM blocks (encoder layers): categorical distribution with values [1, 2, 3, 4]
  * Dropout rate: Normal distribution with min=0.5, max=0.15, mean=0.1, std=0.025
* Random Seed (acts as initial seed for both the global tunning process and the local config evaluation process): 123 `seed 123`

This command tunes TimeMixer's hyperparameters by using the Hyperopt TPE search algorithm. The script allows for specifying key factors related to the HPO experiment configuration like the search space, the number of samples, resource allocation (GPU and CPU), and a time budget for tuning, enabling efficient exploration of various model configurations. 

In both cases, the script arguments that are not specified take the default value as given by their documentation (which you can print by running the command `python3 tune_timemixer.py --help`)

## **TimeMixer-ETTh1 experiment**

To replicate the results of the `TimeMixer-ETTh1` experiment (TimeMixer architecture with the ETTh1 dataset) presented in the TFM, run the following command. 

```{bash}
. TimeMixer-ETTh1.sh
```

The script `TimeMixer-ETTh1` contains all the instructions required to obtain the results reported in the Final Master's Project for the `TimeMixer-ETTh1` experiment (full reproducibility), from baseline evaluation to hyperparameter tunning, plot generation and comparative analysis. Runtime is approximately 4 GPU days on an Nvidia RTX 3090, with the following parallelization scheme for a single GPU: 4 concurrent trials for random search, TPE, and BOHB algorithms, and 1 concurrent trial for the SMAC algorithm.


![baseline](tfm_imgs/baseline.png)
**Figure 2**: Error metrics of TimeMixer on the ETTh1, ETTh2, ETTm1, ETTm2 and Weather datasets. The source code was built on top of the official TimeMixer official implementation, with modifications aimed at improving efficiency and flexibility. Hyperparameters (including the random seed) were set according to the values specified in the TimeMixer paper to reproduce the reported results.

![evolution](tfm_imgs/time_vs_cost/summary.png)
**Figure 3**: Evolution of each hyperparameter tuning session conducted on TimeMixer's hyperparameters with ETTh1 dataset. The horizontal axis represents wall clock time in seconds and the number of configurations evaluated, while the vertical axis represents the objective cost of the tuning session, in this case, the best validation loss observed across the entire learning curve (seed=123)

![comparison_96](tfm_imgs/comparison_h96.png)
![comparison_192](tfm_imgs/comparison_h192.png)
![comparison_336](tfm_imgs/comparison_h336.png)
![comparison_720](tfm_imgs/comparison_h720.png)
**Figure 4**: Comparison between the default hyperparameters and the best configuration found by the set of algorithms considered, for each forecast horizon. The lowest values are highlighted in yellow, indicating the best configuration for each particular metric (seed=123)

## References

```
@inproceedings{wang2023timemixer,
  title={TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting},
  author={Wang, Shiyu and Wu, Haixu and Shi, Xiaoming and Hu, Tengge and Luo, Huakun and Ma, Lintao and Zhang, James Y and ZHOU, JUN},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024}
}
@article{liaw2018tune,
  title={Tune: A Research Platform for Distributed Model Selection and Training},
  author={Liaw, Richard and Liang, Eric and Nishihara, Robert and Moritz, Philipp and Gonzalez, Joseph E and Stoica, Ion},
  journal={arXiv preprint arXiv:1807.05118},
  year={2018}
}
@article{lindauer2021smac3,
  title={SMAC3: A Versatile Bayesian Optimization Package for Hyperparameter Optimization},
  author={Lindauer, Marius and Eggensperger, Katharina and Feurer, Matthias and Biedenkapp, André and Deng, Difan and Benjamins, Carolin and Ruhkopf, Tim and Sass, René and Hutter, Frank},
  journal={arXiv preprint arXiv:2109.09831},
  year={2021}
}
```