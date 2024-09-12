horizon=720
maxconcurrent=1
gpu_fraction=$(echo "scale=2; 1/$maxconcurrent" | bc)  # Calculate GPU fraction with 2 decimal places
start_time=$(date +%s)  # Get the current time in seconds
python3 tune_timemixer.py \
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
    --tune_search_algorithm bohb \
    --tune_storage_path ./checkpoints/hptunning/bohb/ \
    --tune_experiment_name ETTh1_96_${horizon} \
    --tune_objective best_valid_loss \
    --tune_num_samples 1500 \
    --tune_max_trial_time_s 100 \
    --tune_time_budget_s 14400 \
    --tune_max_concurrent $maxconcurrent \
    --tune_gpu_resources $gpu_fraction \
    --tune_cpu_resources 1 \
    --tune_default_config "{
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
    --tune_param_space "{
        \"batch_size\": [\"choice\", [16, 32, 64, 128]], \
        \"learning_rate\": [\"loguniform\", [0.0005, 0.012]], \
        \"down_sampling_method\": [\"choice\", [\"avg\", \"conv\"]], \
        \"d_model\": [\"choice\", [8, 16, 32, 64, 128, 256, 512]], \
        \"alpha_d_ff\": [\"choice\", [2, 3, 4]], \
        \"decomp_method\": [\"choice\", [[\"moving_avg\", \"moving_avg\", [15, 25, 35, 55, 75]], [\"dft_decomp\"]]], \
        \"e_layers\": [\"choice\", [1, 2, 3, 4]], \
        \"dropout\": [\"normal\", [0.1, 0.025]]
    }" \
    --tune_hb_eta 3 \
    --tune_bohb_min_points_in_model 150 \
    --tune_bohb_top_n_percent 15 \
    --tune_bohb_num_samples 64 \
    --tune_bohb_random_fraction 0.333 \
    --tune_bohb_bandwidth_factor 3 \
    --tune_bohb_min_bandwidth 0.001 \
    --seed 123;
end_time=$(date +%s)  # Get the current time in seconds
elapsed_time=$((end_time - start_time))  # Calculate the elapsed time
echo ""
echo ""
echo "Time taken ($maxconcurrent parallel trials): $elapsed_time seconds"
echo ""
echo ""