python3 train_timemixer.py \
    --model TimeMixer \
    --data ETTh1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv \
    --features M --save_dir ./checkpoints/baseline/reproduction/ETTh1_96_96 \
    --seq_len 96 --label_len 0 --pred_len 96 --down_sampling_layers 3 --down_sampling_window 2 --down_sampling_method avg \
    --enc_in 7 --dec_in 7 --c_out 7 --d_model 16 --d_ff 32 --learning_rate 0.01 --train_epochs 10 --patience 10 --batch_size 128;

python3 train_timemixer.py \
    --model TimeMixer \
    --data ETTh1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv \
    --features M --save_dir ./checkpoints/baseline/reproduction/ETTh1_96_192 \
    --seq_len 96 --label_len 0 --pred_len 192 --down_sampling_layers 3 --down_sampling_window 2 --down_sampling_method avg \
    --enc_in 7 --dec_in 7 --c_out 7 --d_model 16 --d_ff 32 --learning_rate 0.01 --train_epochs 10 --patience 10 --batch_size 128;

python3 train_timemixer.py \
    --model TimeMixer \
    --data ETTh1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv \
    --features M --save_dir ./checkpoints/baseline/reproduction/ETTh1_96_336 \
    --seq_len 96 --label_len 0 --pred_len 336 --down_sampling_layers 3 --down_sampling_window 2 --down_sampling_method avg \
    --enc_in 7 --dec_in 7 --c_out 7 --d_model 16 --d_ff 32 --learning_rate 0.01 --train_epochs 10 --patience 10 --batch_size 128;

python3 train_timemixer.py \
    --model TimeMixer \
    --data ETTh1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv \
    --features M --save_dir ./checkpoints/baseline/reproduction/ETTh1_96_720 \
    --seq_len 96 --label_len 0 --pred_len 720 --down_sampling_layers 3 --down_sampling_window 2 --down_sampling_method avg \
    --enc_in 7 --dec_in 7 --c_out 7 --d_model 16 --d_ff 32 --learning_rate 0.01 --train_epochs 10 --patience 10 --batch_size 128;
