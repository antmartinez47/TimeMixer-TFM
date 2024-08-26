python3 train_timemixer.py \
    --model TimeMixer \
    --data custom --root_path ./dataset/traffic/ --data_path traffic.csv \
    --features M --save_dir ./checkpoints/baseline/TimeMixer/traffic_96_96 \
    --seq_len 96 --label_len 0 --pred_len 96 --down_sampling_layers 3 --down_sampling_window 2 --down_sampling_method avg \
    --enc_in 862 --dec_in 862 --c_out 862 --d_model 32 --d_ff 64 --e_layers 3 --d_layers 1 --factor 3 \
    --learning_rate 0.01 --train_epochs 10 --patience 10 --batch_size 8 --itr 1;

python3 train_timemixer.py \
    --model TimeMixer \
    --data custom --root_path ./dataset/traffic/ --data_path traffic.csv \
    --features M --save_dir ./checkpoints/baseline/TimeMixer/traffic_96_192 \
    --seq_len 96 --label_len 0 --pred_len 192 --down_sampling_layers 3 --down_sampling_window 2 --down_sampling_method avg \
    --enc_in 862 --dec_in 862 --c_out 862 --d_model 32 --d_ff 64 --e_layers 3 --d_layers 1 --factor 3 \
    --learning_rate 0.01 --train_epochs 10 --patience 10 --batch_size 8 --itr 1;

python3 train_timemixer.py \
    --model TimeMixer \
    --data custom --root_path ./dataset/traffic/ --data_path traffic.csv \
    --features M --save_dir ./checkpoints/baseline/TimeMixer/traffic_96_336 \
    --seq_len 96 --label_len 0 --pred_len 336 --down_sampling_layers 3 --down_sampling_window 2 --down_sampling_method avg \
    --enc_in 862 --dec_in 862 --c_out 862 --d_model 32 --d_ff 64 --e_layers 3 --d_layers 1 --factor 3 \
    --learning_rate 0.01 --train_epochs 10 --patience 10 --batch_size 8 --itr 1;

python3 train_timemixer.py \
    --model TimeMixer \
    --data custom --root_path ./dataset/traffic/ --data_path traffic.csv \
    --features M --save_dir ./checkpoints/baseline/TimeMixer/traffic_96_720 \
    --seq_len 96 --label_len 0 --pred_len 720 --down_sampling_layers 3 --down_sampling_window 2 --down_sampling_method avg \
    --enc_in 862 --dec_in 862 --c_out 862 --d_model 32 --d_ff 64 --e_layers 3 --d_layers 1 --factor 3 \
    --learning_rate 0.01 --train_epochs 10 --patience 10 --batch_size 8 --itr 1;