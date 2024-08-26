conda activate py3.11-timemixer-raytune

mkdir -p scripts/long_term_forecast/ETT_script/logs
mkdir -p scripts/long_term_forecast/Weather_script/logs

. scripts/long_term_forecast/ETT_scripts/TimeMixer_ETTh1_unify.sh > scripts/long_term_forecast/ETT_scripts/logs/TimeMixer_ETTh1_unify.txt 2>&1
. scripts/long_term_forecast/Weather_scripts/TimeMixer_unify.sh > scripts/long_term_forecast/Weather_scripts/logs/TimeMixer_unify.txt 2>&1