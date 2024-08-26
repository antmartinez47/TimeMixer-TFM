conda activate py3.11-timemixer-raytune

mkdir -p tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_96/logs
mkdir -p tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_192/logs
mkdir -p tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_336/logs
mkdir -p tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_720/logs

# Random Search

cat tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_96/random_search.sh > tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_96/logs/random_search.txt
. tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_96/random_search.sh >> tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_96/logs/random_search.txt 2>&1

cat tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_192/random_search.sh > tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_192/logs/random_search.txt
. tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_192/random_search.sh >> tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_192/logs/random_search.txt 2>&1

cat tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_336/random_search.sh > tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_336/logs/random_search.txt
. tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_336/random_search.sh >> tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_336/logs/random_search.txt 2>&1

cat tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_720/random_search.sh > tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_720/logs/random_search.txt
. tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_720/random_search.sh >> tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_720/logs/random_search.txt 2>&1

# Hyperopt TPE

cat tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_96/hyperopt_tpe.sh > tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_96/logs/hyperopt_tpe.txt
. tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_96/hyperopt_tpe.sh >> tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_96/logs/hyperopt_tpe.txt 2>&1

cat tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_192/hyperopt_tpe.sh > tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_192/logs/hyperopt_tpe.txt
. tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_192/hyperopt_tpe.sh >> tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_192/logs/hyperopt_tpe.txt 2>&1

cat tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_336/hyperopt_tpe.sh > tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_336/logs/hyperopt_tpe.txt
. tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_336/hyperopt_tpe.sh >> tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_336/logs/hyperopt_tpe.txt 2>&1

cat tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_720/hyperopt_tpe.sh > tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_720/logs/hyperopt_tpe.txt
. tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_720/hyperopt_tpe.sh >> tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_720/logs/hyperopt_tpe.txt 2>&1


# BOHB

cat tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_96/bohb.sh > tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_96/logs/bohb.txt
. tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_96/bohb.sh >> tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_96/logs/bohb.txt 2>&1

cat tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_192/bohb.sh > tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_192/logs/bohb.txt
. tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_192/bohb.sh >> tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_192/logs/bohb.txt 2>&1

cat tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_336/bohb.sh > tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_336/logs/bohb.txt
. tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_336/bohb.sh >> tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_336/logs/bohb.txt 2>&1

cat tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_720/bohb.sh > tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_720/logs/bohb.txt
. tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_720/bohb.sh >> tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_720/logs/bohb.txt 2>&1

# SMAC

cat tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_96/smac.sh > tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_96/logs/smac.txt
. tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_96/smac.sh >> tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_96/logs/smac.txt 2>&1

cat tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_192/smac.sh > tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_192/logs/smac.txt
. tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_192/smac.sh >> tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_192/logs/smac.txt 2>&1

cat tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_336/smac.sh > tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_336/logs/smac.txt
. tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_336/smac.sh >> tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_336/logs/smac.txt 2>&1

cat tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_720/smac.sh > tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_720/logs/smac.txt
. tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_720/smac.sh >> tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_720/logs/smac.txt 2>&1

donda deactivate

conda activate py3.10-timemixer-smac

# SMAC

cat tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_96/smac.sh > tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_96/logs/smac.txt
. tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_96/smac.sh >> tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_96/logs/smac.txt 2>&1

cat tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_192/smac.sh > tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_192/logs/smac.txt
. tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_192/smac.sh >> tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_192/logs/smac.txt 2>&1

cat tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_336/smac.sh > tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_336/logs/smac.txt
. tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_336/smac.sh >> tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_336/logs/smac.txt 2>&1

cat tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_720/smac.sh > tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_720/logs/smac.txt
. tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_720/smac.sh >> tfm_scripts/tune_long_term_forecast/ETT_script/ETTh1_96_720/logs/smac.txt 2>&1

