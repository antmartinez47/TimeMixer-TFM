python3 train_timemixer.py \
    --model TimeMixer \
    --data custom --root_path ./dataset/weather/ --data_path weather.csv \
    --features M --save_dir ./checkpoints/baseline/weather_96_96 \
    --seq_len 96 --label_len 0 --pred_len 96 --down_sampling_layers 3 --down_sampling_window 2 --down_sampling_method avg \
    --enc_in 21 --dec_in 21 --c_out 21 --d_model 16 --d_ff 32 --e_layers 3 \
    --learning_rate 0.01 --train_epochs 20 --patience 10 --batch_size 128;

python3 train_timemixer.py \
    --model TimeMixer \
    --data custom --root_path ./dataset/weather/ --data_path weather.csv \
    --features M --save_dir ./checkpoints/baseline/weather_96_192 \
    --seq_len 96 --label_len 0 --pred_len 192 --down_sampling_layers 3 --down_sampling_window 2 --down_sampling_method avg \
    --enc_in 21 --dec_in 21 --c_out 21 --d_model 16 --d_ff 32 --e_layers 3 \
    --learning_rate 0.01 --train_epochs 20 --patience 10 --batch_size 128;

python3 train_timemixer.py \
    --model TimeMixer \
    --data custom --root_path ./dataset/weather/ --data_path weather.csv \
    --features M --save_dir ./checkpoints/baseline/weather_96_336 \
    --seq_len 96 --label_len 0 --pred_len 336 --down_sampling_layers 3 --down_sampling_window 2 --down_sampling_method avg \
    --enc_in 21 --dec_in 21 --c_out 21 --d_model 16 --d_ff 32 --e_layers 3 \
    --learning_rate 0.01 --train_epochs 20 --patience 10 --batch_size 128;

python3 train_timemixer.py \
    --model TimeMixer \
    --data custom --root_path ./dataset/weather/ --data_path weather.csv \
    --features M --save_dir ./checkpoints/baseline/weather_96_720 \
    --seq_len 96 --label_len 0 --pred_len 720 --down_sampling_layers 3 --down_sampling_window 2 --down_sampling_method avg \
    --enc_in 21 --dec_in 21 --c_out 21 --d_model 16 --d_ff 32 --e_layers 3 \
    --learning_rate 0.01 --train_epochs 20 --patience 10 --batch_size 128;