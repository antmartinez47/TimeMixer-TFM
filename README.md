System requirements:

* Ubuntu 18.04+
* NVIDIA GPU with latest stable drivers
* conda
* gcc

Assuming current working directory is TimeMixer-TFM repository:

### Install conda environments (linux system required; tested on ubuntu 22.04)

```{bash}
conda create -f conda_config_files/environment-raytune.yml
conda create -f conda_config_files/environment-smac.yml
```

### Download datasets

```{bash}
conda activate py3.11-timemixer-raytune
python3 download_data.py
conda deactivate
```
### Run training scripts for default configurations (original implementation)

```{bash}
. runOriginalScripts.sh
```

### Run training scripts for default configurations (modified original implementation)

```{bash}
. runReproducedScripts.sh
```

### Run hyperaprameter tunning experiment for ETTh1 dataset

```{bash}
. runHPTunningScriptsETTh1.sh
```