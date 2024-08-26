horizon=336
maxconcurrent=1
start_time=$(date +%s)  # Get the current time in seconds
python3 smac_timemixer.py \
    --model TimeMixer \
    --data ETTh1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --features M \
    --seq_len 96 \
    --label_len 0 \
    --pred_len $horizon \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_model 16 \
    --d_ff 32 \
    --down_sampling_layers 3 \
    --down_sampling_window 2 \
    --down_sampling_method avg \
    --decomp_method moving_avg \
    --moving_avg 25 \
    --train_epochs 8 \
    --patience 0 \
    --num_workers 1 \
    --gpu 0 \
    --smac_storage_path ./checkpoints/hptunning/smac/ \
    --smac_experiment_name ETTh1_96_${horizon} \
    --smac_n_trials 1500 \
    --smac_trial_walltime_limit 100 \
    --smac_time_budget_s 14400 \
    --smac_n_workers $maxconcurrent \
    --smac_default_config "{
        \"batch_size\": 128, \
        \"learning_rate\": 0.01, \
        \"down_sampling_method\": \"avg\", \
        \"d_model\": 16, \
        \"alpha_d_ff\": 2, \
        \"decomp_method\": \"moving_avg\", \
        \"moving_avg\": 25, \
        \"e_layers\": 2, \
        \"dropout\": 0.1
    }" \
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
    --smac_min_budget 1 \
    --smac_eta 3 \
    --smac_incumbent_selection "highest_budget" \
    --smac_n_init_configs 150 \
    --seed 123;
end_time=$(date +%s)  # Get the current time in seconds
elapsed_time=$((end_time - start_time))  # Calculate the elapsed time
echo ""
echo ""
echo "Time taken ($maxconcurrent parallel trials): $elapsed_time seconds"
echo ""
echo ""