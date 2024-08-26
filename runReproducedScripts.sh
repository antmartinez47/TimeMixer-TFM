conda activate py3.11-timemixer-raytune

mkdir -p tfm_scripts/long_term_forecast/ETT_script/logs
mkdir -p tfm_scripts/long_term_forecast/Traffic_script/logs
mkdir -p tfm_scripts/long_term_forecast/Weather_script/logs

. tfm_scripts/long_term_forecast/ETT_script/ETTh1.sh > tfm_scripts/long_term_forecast/ETT_script/logs/ETTh1.txt
. tfm_scripts/long_term_forecast/ETT_script/ETTh2.sh > tfm_scripts/long_term_forecast/ETT_script/logs/ETTh2.txt
. tfm_scripts/long_term_forecast/ETT_script/ETTm1.sh > tfm_scripts/long_term_forecast/ETT_script/logs/ETTm1.txt
. tfm_scripts/long_term_forecast/ETT_script/ETTm2.sh > tfm_scripts/long_term_forecast/ETT_script/logs/ETTm2.txt

. tfm_scripts/long_term_forecast/Traffic_script/Traffic.sh > tfm_scripts/long_term_forecast/Traffic_script/logs/Traffic.txt
. tfm_scripts/long_term_forecast/Weather_script/Weather.sh > tfm_scripts/long_term_forecast/Weather_script/logs/Weather.txt