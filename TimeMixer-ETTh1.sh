export CUDA_VISIBLE_DEVICES=0 # Enforce single-GPU training

conda activate py3.11-timemixer-raytune

# Download and extract datasets (link in TimeMixer official repository)
python3 download_data.py

# Create log directories if not exists
mkdir -p scripts/long_term_forecast/ETT_script/logs
mkdir -p scripts/long_term_forecast/Weather_script/logs
mkdir -p tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_96/logs
mkdir -p tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_192/logs
mkdir -p tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_336/logs
mkdir -p tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_720/logs

### BASELINE (default hyperparameters; extracted from the unified hyperparameter as specified in paper)
### Input size is set to 96 samples. Output size is within {96, 192, 336, 720} samples

## Run original scripts for ETTh1, ETTh2, ETTm1, ETTm2 and Weather datasets

cat scripts/long_term_forecast/ETT_script/TimeMixer_ETTh1_unify.sh > scripts/long_term_forecast/ETT_script/logs/TimeMixer_ETTh1_unify.txt
. scripts/long_term_forecast/ETT_script/TimeMixer_ETTh1_unify.sh >> scripts/long_term_forecast/ETT_script/logs/TimeMixer_ETTh1_unify.txt 2>&1
cat scripts/long_term_forecast/ETT_script/TimeMixer_ETTh2_unify.sh > scripts/long_term_forecast/ETT_script/logs/TimeMixer_ETTh2_unify.txt
. scripts/long_term_forecast/ETT_script/TimeMixer_ETTh2_unify.sh >> scripts/long_term_forecast/ETT_script/logs/TimeMixer_ETTh2_unify.txt 2>&1
cat scripts/long_term_forecast/ETT_script/TimeMixer_ETTm1_unify.sh > scripts/long_term_forecast/ETT_script/logs/TimeMixer_ETTm1_unify.txt
. scripts/long_term_forecast/ETT_script/TimeMixer_ETTm1_unify.sh >> scripts/long_term_forecast/ETT_script/logs/TimeMixer_ETTm1_unify.txt 2>&1
cat scripts/long_term_forecast/ETT_script/TimeMixer_ETTm2_unify.sh > scripts/long_term_forecast/ETT_script/logs/TimeMixer_ETTm2_unify.txt
. scripts/long_term_forecast/ETT_script/TimeMixer_ETTm2_unify.sh >> scripts/long_term_forecast/ETT_script/logs/TimeMixer_ETTm2_unify.txt 2>&1
cat scripts/long_term_forecast/Weather_script/TimeMixer_unify.sh > scripts/long_term_forecast/Weather_script/logs/TimeMixer_unify.txt
. scripts/long_term_forecast/Weather_script/TimeMixer_unify.sh >> scripts/long_term_forecast/Weather_script/logs/TimeMixer_unify.txt 2>&1

## Run modified scripts for ETTh1, ETTh2, ETTm1, ETTm2 and Weather datasets

mkdir -p tfm_scripts/long_term_forecast/ETT_script/logs
mkdir -p tfm_scripts/long_term_forecast/Weather_script/logs

cat tfm_scripts/long_term_forecast/ETT_script/TimeMixer_ETTh1_unify.sh > tfm_scripts/long_term_forecast/ETT_script/logs/TimeMixer_ETTh1_unify.txt
. tfm_scripts/long_term_forecast/ETT_script/TimeMixer_ETTh1_unify.sh >> tfm_scripts/long_term_forecast/ETT_script/logs/TimeMixer_ETTh1_unify.txt 2>&1
cat tfm_scripts/long_term_forecast/ETT_script/TimeMixer_ETTh2_unify.sh > tfm_scripts/long_term_forecast/ETT_script/logs/TimeMixer_ETTh2_unify.txt
. tfm_scripts/long_term_forecast/ETT_script/TimeMixer_ETTh2_unify.sh >> tfm_scripts/long_term_forecast/ETT_script/logs/TimeMixer_ETTh2_unify.txt 2>&1
cat tfm_scripts/long_term_forecast/ETT_script/TimeMixer_ETTm1_unify.sh > tfm_scripts/long_term_forecast/ETT_script/logs/TimeMixer_ETTm1_unify.txt
. tfm_scripts/long_term_forecast/ETT_script/TimeMixer_ETTm1_unify.sh >> tfm_scripts/long_term_forecast/ETT_script/logs/TimeMixer_ETTm1_unify.txt 2>&1
cat tfm_scripts/long_term_forecast/ETT_script/TimeMixer_ETTm2_unify.sh > tfm_scripts/long_term_forecast/ETT_script/logs/TimeMixer_ETTm2_unify.txt
. tfm_scripts/long_term_forecast/ETT_script/TimeMixer_ETTm2_unify.sh >> tfm_scripts/long_term_forecast/ETT_script/logs/TimeMixer_ETTm2_unify.txt 2>&1
cat tfm_scripts/long_term_forecast/Weather_script/TimeMixer_unify.sh > tfm_scripts/long_term_forecast/Weather_script/logs/TimeMixer_unify.txt
. tfm_scripts/long_term_forecast/Weather_script/TimeMixer_unify.sh >> tfm_scripts/long_term_forecast/Weather_script/logs/TimeMixer_unify.txt 2>&1

### HPTUNNING: TimeMixer-ETTh1 (TFM)
### Input size is set to 96 samples. Output size is within {96, 192, 336, 720} samples

# Random Search
# ETTh1_96_96
cat tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_96/random_search.sh > tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_96/logs/random_search.txt
. tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_96/random_search.sh >> tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_96/logs/random_search.txt 2>&1
# ETTh1_96_192
cat tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_192/random_search.sh > tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_192/logs/random_search.txt
. tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_192/random_search.sh >> tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_192/logs/random_search.txt 2>&1
# ETTh1_96_336
cat tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_336/random_search.sh > tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_336/logs/random_search.txt
. tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_336/random_search.sh >> tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_336/logs/random_search.txt 2>&1
# ETTh1_96_720
cat tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_720/random_search.sh > tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_720/logs/random_search.txt
. tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_720/random_search.sh >> tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_720/logs/random_search.txt 2>&1

# Hyperopt TPE
# ETTh1_96_96
cat tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_96/hyperopt_tpe.sh > tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_96/logs/hyperopt_tpe.txt
. tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_96/hyperopt_tpe.sh >> tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_96/logs/hyperopt_tpe.txt 2>&1
# ETTh1_96_192
cat tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_192/hyperopt_tpe.sh > tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_192/logs/hyperopt_tpe.txt
. tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_192/hyperopt_tpe.sh >> tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_192/logs/hyperopt_tpe.txt 2>&1
# ETTh1_96_336
cat tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_336/hyperopt_tpe.sh > tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_336/logs/hyperopt_tpe.txt
. tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_336/hyperopt_tpe.sh >> tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_336/logs/hyperopt_tpe.txt 2>&1
# ETTh1_96_720
cat tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_720/hyperopt_tpe.sh > tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_720/logs/hyperopt_tpe.txt
. tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_720/hyperopt_tpe.sh >> tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_720/logs/hyperopt_tpe.txt 2>&1¡

# BOHB
# ETTh1_96_96
cat tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_96/bohb.sh > tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_96/logs/bohb.txt
. tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_96/bohb.sh >> tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_96/logs/bohb.txt 2>&1
# ETTh1_96_192
cat tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_192/bohb.sh > tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_192/logs/bohb.txt
. tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_192/bohb.sh >> tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_192/logs/bohb.txt 2>&1
# ETTh1_96_336
cat tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_336/bohb.sh > tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_336/logs/bohb.txt
. tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_336/bohb.sh >> tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_336/logs/bohb.txt 2>&1
# ETTh1_96_720
cat tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_720/bohb.sh > tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_720/logs/bohb.txt
. tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_720/bohb.sh >> tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_720/logs/bohb.txt 2>&1

conda deactivate
conda activate py3.10-timemixer-smac

# SMAC
# ETTh1_96_96
cat tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_96/smac.sh > tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_96/logs/smac.txt
. tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_96/smac.sh >> tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_96/logs/smac.txt 2>&1
# ETTh1_96_192
cat tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_192/smac.sh > tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_192/logs/smac.txt
. tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_192/smac.sh >> tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_192/logs/smac.txt 2>&1
# ETTh1_96_336
cat tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_336/smac.sh > tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_336/logs/smac.txt
. tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_336/smac.sh >> tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_336/logs/smac.txt 2>&1
# ETTh1_96_720
cat tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_720/smac.sh > tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_720/logs/smac.txt
. tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_720/smac.sh >> tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_720/logs/smac.txt 2>&1

conda deactivate
conda activate py3.11-timemixer-raytune

# Train and evaluate default for each horizon setting and with the same initial seed as the one utilized during HPO
cat tfm_scripts/tune_long_term_forecast/ETT_script/evaluate_default_configs/train.sh > tfm_scripts/tune_long_term_forecast/ETT_script/evaluate_default_configs/train.txt
. tfm_scripts/tune_long_term_forecast/ETT_script/evaluate_default_configs/train.sh >> tfm_scripts/tune_long_term_forecast/ETT_script/evaluate_default_configs/train.txt 2>&1
# Train and evaluate best configuration found by each algorithm for each setting and with the same initial seed
cat tfm_scripts/tune_long_term_forecast/ETT_script/evaluate_best_configs/train.sh > tfm_scripts/tune_long_term_forecast/ETT_script/evaluate_best_configs/train.txt
. tfm_scripts/tune_long_term_forecast/ETT_script/evaluate_best_configs/train.sh >> tfm_scripts/tune_long_term_forecast/ETT_script/evaluate_best_configs/train.txt 2>&1

# Plot Generation

# ETTh1_96_96
python3 plot_tune_results.py --title ETTh1_96_96 \
    --keys random_search hyperopt_tpe bohb smac3 \
    --csv_paths checkpoints/hptunning/random_search/ETTh1_96_96/tune_results.csv \
                checkpoints/hptunning/hyperopt_tpe/ETTh1_96_96/tune_results.csv \
                checkpoints/hptunning/bohb/ETTh1_96_96/tune_results.csv \
                checkpoints/hptunning/smac/ETTh1_96_96/results.csv \
    --out_dir tfm_imgs;
# ETTh1_96_192
python3 plot_tune_results.py --title ETTh1_96_192 \
    --keys random_search hyperopt_tpe bohb smac3 \
    --csv_paths checkpoints/hptunning/random_search/ETTh1_96_192/tune_results.csv \
                checkpoints/hptunning/hyperopt_tpe/ETTh1_96_192/tune_results.csv \
                checkpoints/hptunning/bohb/ETTh1_96_192/tune_results.csv \
                checkpoints/hptunning/smac/ETTh1_96_192/results.csv \
    --out_dir tfm_imgs;
# ETTh1_96_336
python3 plot_tune_results.py --title ETTh1_96_336 \
    --keys random_search hyperopt_tpe bohb smac3 \
    --csv_paths checkpoints/hptunning/random_search/ETTh1_96_336/tune_results.csv \
                checkpoints/hptunning/hyperopt_tpe/ETTh1_96_336/tune_results.csv \
                checkpoints/hptunning/bohb/ETTh1_96_336/tune_results.csv \
                checkpoints/hptunning/smac/ETTh1_96_336/results.csv \
    --out_dir tfm_imgs;
# ETTh1_96_720
python3 plot_tune_results.py --title ETTh1_96_720 \
    --keys random_search hyperopt_tpe bohb smac3 \
    --csv_paths checkpoints/hptunning/random_search/ETTh1_96_720/tune_results.csv \
                checkpoints/hptunning/hyperopt_tpe/ETTh1_96_720/tune_results.csv \
                checkpoints/hptunning/bohb/ETTh1_96_720/tune_results.csv \
                checkpoints/hptunning/smac/ETTh1_96_720/results.csv \
    --out_dir tfm_imgs;

