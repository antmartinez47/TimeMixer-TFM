
# Train and evaluate the best configuration found by each algorithm with the same initial seed

# ETTh1_96_96

# random_search

# Trial status: 394 TERMINATED | 1 ERROR
# Current time: 2024-08-23 04:42:36. Total running time: 4hr 0min 0s
# Logical resource usage: 2.0/32 CPUs, 1.0/1 GPUs (0.0/1.0 accelerator_type:G)
# Current best trial: d31f1_00256 with best_valid_loss=0.6711121400625542 and 
# params={
#     'batch_size': 16, 
#     'learning_rate': 0.0026127872930535134, 
#     'down_sampling_method': 'conv', 
#     'd_model': 16, 
#     'decomp_method': 'moving_avg', 
#     'moving_avg': 55, 
#     'e_layers': 3, 
#     'dropout': 0.08646655843366237, 
#     'd_ff': 48
#     }

python3 train_timemixer.py \
    --model TimeMixer \
    --data ETTh1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --features M \
    --save_dir ./checkpoints/hptunning/best_configs/random_search/ETTh1_96_96 \
    --seq_len 96 \
    --label_len 0 \
    --pred_len 96 \
    --down_sampling_layers 3 \
    --down_sampling_window 2 \
    --down_sampling_method conv \
    --decomp_method moving_avg \
    --moving_avg 55 \
    --e_layers 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_model 16 \
    --d_ff 48 \
    --learning_rate 0.0026127872930535134 \
    --train_epochs 10 \
    --patience 10 \
    --batch_size 16 \
    --train_prop 1.0 \
    --dropout 0.08646655843366237 \
    --seed 123;

# hyperopt_tpe


# Trial status: 541 TERMINATED | 1 ERROR
# Current time: 2024-08-23 23:20:43. Total running time: 4hr 0min 0s
# Logical resource usage: 2.0/32 CPUs, 1.0/1 GPUs (0.0/1.0 accelerator_type:G)
# Current best trial: 2a6d8827 with best_valid_loss=0.6645102063300966 and 
# params={
#     'alpha_d_ff': 4, 
#     'batch_size': 32, 
#     'd_model': 64, 
#     'decomp_method': {'decomp_method': 'dft_decomp'}, 
#     'down_sampling_method': 'conv', 
#     'dropout': 0.06403254931163539, 
#     'e_layers': 1, 
#     'learning_rate': 0.008535115341011767
#     }

python3 train_timemixer.py \
    --model TimeMixer \
    --data ETTh1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --features M \
    --save_dir ./checkpoints/hptunning/best_configs/hyperopt_tpe/ETTh1_96_96 \
    --seq_len 96 \
    --label_len 0 \
    --pred_len 96 \
    --down_sampling_layers 3 \
    --down_sampling_window 2 \
    --down_sampling_method conv \
    --decomp_method dft_decomp \
    --e_layers 1 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_model 64 \
    --d_ff 256 \
    --learning_rate 0.008535115341011767 \
    --train_epochs 10 \
    --patience 10 \
    --batch_size 32 \
    --train_prop 1.0 \
    --dropout 0.06403254931163539 \
    --seed 123;

# bohb

# Trial status: 437 TERMINATED | 8 ERROR
# Current time: 2024-08-24 15:46:08. Total running time: 4hr 0min 0s
# Logical resource usage: 2.0/32 CPUs, 1.0/1 GPUs (0.0/1.0 accelerator_type:G)
# Current best trial: 11a9d5c0 with best_valid_loss=0.6727460483196138 and 
# params={
#     'batch_size': 32, 
#     'd_model': 128, 'decomp_method': 'moving_avg', 
#     'down_sampling_method': 'conv', 
#     'dropout': 0.10977791287466174, 
#     'e_layers': 4, 
#     'learning_rate': 0.0019597614256863247, 
#     'moving_avg': 75, 
#     'd_ff': 384
#     }

python3 train_timemixer.py \
    --model TimeMixer \
    --data ETTh1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --features M \
    --save_dir ./checkpoints/hptunning/best_configs/bohb/ETTh1_96_96 \
    --seq_len 96 \
    --label_len 0 \
    --pred_len 96 \
    --down_sampling_layers 3 \
    --down_sampling_window 2 \
    --down_sampling_method conv \
    --decomp_method moving_avg \
    --moving_avg 75 \
    --e_layers 4 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_model 32 \
    --d_ff 384 \
    --learning_rate 0.0019597614256863247 \
    --train_epochs 10 \
    --patience 10 \
    --batch_size 32 \
    --train_prop 1.0 \
    --dropout 0.10977791287466174 \
    --seed 123;

# smac

# Configuration(values={
#   'alpha_d_ff': 2,
#   'batch_size': 16,
#   'd_model': 128,
#   'decomp_method': 'moving_avg',
#   'down_sampling_method': 'conv',
#   'dropout': 0.0680883074135,
#   'e_layers': 2,
#   'learning_rate': 0.001112923223,
#   'moving_avg': 55,
# })
# Incumbent cost: 0.6698800189227894

python3 train_timemixer.py \
    --model TimeMixer \
    --data ETTh1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --features M \
    --save_dir ./checkpoints/hptunning/best_configs/smac/ETTh1_96_96 \
    --seq_len 96 \
    --label_len 0 \
    --pred_len 96 \
    --down_sampling_layers 3 \
    --down_sampling_window 2 \
    --down_sampling_method conv \
    --decomp_method moving_avg \
    --moving_avg 55 \
    --e_layers 2 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_model 128 \
    --d_ff 256 \
    --learning_rate 0.001112923223 \
    --train_epochs 10 \
    --patience 10 \
    --batch_size 16 \
    --train_prop 1.0 \
    --dropout 0.0680883074135 \
    --seed 123;


# ETTh1_96_192

# random_search

# Trial status: 399 TERMINATED | 1 ERROR
# Current time: 2024-08-23 08:42:51. Total running time: 4hr 0min 0s
# Logical resource usage: 2.0/32 CPUs, 1.0/1 GPUs (0.0/1.0 accelerator_type:G)
# Current best trial: 63b86_00339 with best_valid_loss=0.9778934390771956 and 
# params={
#     'batch_size': 128, 
#     'learning_rate': 0.002950723989351509, 
#     'down_sampling_method': 'avg', 
#     'd_model': 32, 
#     'decomp_method': 'dft_decomp', 
#     'moving_avg': 25, 
#     'e_layers': 4, 
#     'dropout': 0.07079383271536857, 
#     'd_ff': 128
#     }


python3 train_timemixer.py \
    --model TimeMixer \
    --data ETTh1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --features M \
    --save_dir ./checkpoints/hptunning/best_configs/random_search/ETTh1_96_192 \
    --seq_len 96 \
    --label_len 0 \
    --pred_len 192 \
    --down_sampling_layers 3 \
    --down_sampling_window 2 \
    --down_sampling_method avg \
    --decomp_method dft_decomp \
    --e_layers 4 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_model 32 \
    --d_ff 128 \
    --learning_rate 0.002950723989351509 \
    --train_epochs 10 \
    --patience 10 \
    --batch_size 128 \
    --train_prop 1.0 \
    --dropout 0.07079383271536857 \
    --seed 123;

# hyperopt_tpe

# Current time: 2024-08-24 03:20:50. Total running time: 4hr 0min 0s
# Logical resource usage: 2.0/32 CPUs, 1.0/1 GPUs (0.0/1.0 accelerator_type:G)
# Current best trial: dc7da481 with best_valid_loss=0.9722486881627923 and 
# params={
#     'alpha_d_ff': 3, 
#     'batch_size': 32, 
#     'd_model': 128, 
#     'decomp_method': {'decomp_method': 'moving_avg', 'moving_avg': 55}, 
#     'down_sampling_method': 'avg', 
#     'dropout': 0.15727343859406534, 
#     'e_layers': 3, 
#     'learning_rate': 0.0010996702835260287
#     }


python3 train_timemixer.py \
    --model TimeMixer \
    --data ETTh1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --features M \
    --save_dir ./checkpoints/hptunning/best_configs/hyperopt_tpe/ETTh1_96_192 \
    --seq_len 96 \
    --label_len 0 \
    --pred_len 192 \
    --down_sampling_layers 3 \
    --down_sampling_window 2 \
    --down_sampling_method avg \
    --decomp_method moving_avg \
    --moving_avg 55 \
    --e_layers 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_model 32 \
    --d_ff 384 \
    --learning_rate 0.0010996702835260287 \
    --train_epochs 10 \
    --patience 10 \
    --batch_size 32 \
    --train_prop 1.0 \
    --dropout 0.15727343859406534 \
    --seed 123;

# bohb

# Current time: 2024-08-24 19:46:23. Total running time: 4hr 0min 0s
# Logical resource usage: 2.0/32 CPUs, 1.0/1 GPUs (0.0/1.0 accelerator_type:G)
# Current best trial: f5aa88b4 with best_valid_loss=0.97842238062904 and 
# params={
#     'batch_size': 128, 
#     'd_model': 32, 
#     'decomp_method': 'dft_decomp', 
#     'down_sampling_method': 'conv', 
#     'dropout': 0.08999125277492026, 
#     'e_layers': 4, 
#     'learning_rate': 0.007069001999520033, 
#     'd_ff': 96
#     }


python3 train_timemixer.py \
    --model TimeMixer \
    --data ETTh1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --features M \
    --save_dir ./checkpoints/hptunning/best_configs/bohb/ETTh1_96_192 \
    --seq_len 96 \
    --label_len 0 \
    --pred_len 192 \
    --down_sampling_layers 3 \
    --down_sampling_window 2 \
    --down_sampling_method conv \
    --decomp_method dft_decomp \
    --e_layers 4 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_model 32 \
    --d_ff 96 \
    --learning_rate 0.007069001999520033 \
    --train_epochs 10 \
    --patience 10 \
    --batch_size 128 \
    --train_prop 1.0 \
    --dropout 0.08999125277492026 \
    --seed 123;

# smac

# Configuration(values={
#   'alpha_d_ff': 2,
#   'batch_size': 16,
#   'd_model': 128,
#   'decomp_method': 'dft_decomp',
#   'down_sampling_method': 'conv',
#   'dropout': 0.1271247625548,
#   'e_layers': 4,
#   'learning_rate': 0.0010341554496,
# })
# Incumbent cost: 0.9759241329239947

python3 train_timemixer.py \
    --model TimeMixer \
    --data ETTh1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --features M \
    --save_dir ./checkpoints/hptunning/best_configs/smac/ETTh1_96_192 \
    --seq_len 96 \
    --label_len 0 \
    --pred_len 192 \
    --down_sampling_layers 3 \
    --down_sampling_window 2 \
    --down_sampling_method conv \
    --decomp_method dft_decomp \
    --e_layers 4 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_model 128 \
    --d_ff 256 \
    --learning_rate 0.0010341554496 \
    --train_epochs 10 \
    --patience 10 \
    --batch_size 16 \
    --train_prop 1.0 \
    --dropout 0.1271247625548 \
    --seed 123;

# ETTh1_96_336

# Current time: 2024-08-23 12:43:07. Total running time: 4hr 0min 0s
# Logical resource usage: 2.0/32 CPUs, 1.0/1 GPUs (0.0/1.0 accelerator_type:G)
# Current best trial: f3e58_00419 with best_valid_loss=1.2782753614134759 and 
# params={
#     'batch_size': 16, 
#     'learning_rate': 0.000817790812520881, 
#     'down_sampling_method': 'avg', 
#     'd_model': 256, 
#     'decomp_method': 'dft_decomp', 
#     'moving_avg': 35, 
#     'e_layers': 4, 
#     'dropout': 0.11683482976236154, 
#     'd_ff': 768
#     }


python3 train_timemixer.py \
    --model TimeMixer \
    --data ETTh1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --features M \
    --save_dir ./checkpoints/hptunning/best_configs/random_search/ETTh1_96_336 \
    --seq_len 96 \
    --label_len 0 \
    --pred_len 336 \
    --down_sampling_layers 3 \
    --down_sampling_window 2 \
    --down_sampling_method avg \
    --decomp_method dft_decomp \
    --e_layers 4 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_model 256 \
    --d_ff 768 \
    --learning_rate 0.000817790812520881 \
    --train_epochs 10 \
    --patience 10 \
    --batch_size 16 \
    --train_prop 1.0 \
    --dropout 0.11683482976236154 \
    --seed 123;


# hyperopt_tpe

# Trial status: 322 TERMINATED | 31 ERROR
# Current time: 2024-08-24 07:20:57. Total running time: 4hr 0min 0s
# Logical resource usage: 2.0/32 CPUs, 1.0/1 GPUs (0.0/1.0 accelerator_type:G)
# Current best trial: 7b0a02c9 with best_valid_loss=1.2693479460365367 and 
# params={
#     'alpha_d_ff': 4, 
#     'batch_size': 16, 
#     'd_model': 256, 
#     'decomp_method': {'decomp_method': 'dft_decomp'}, 
#     'down_sampling_method': 'avg', 
#     'dropout': 0.11374290594454176, 
#     'e_layers': 4, 
#     'learning_rate': 0.0006048636262373894
#     }


python3 train_timemixer.py \
    --model TimeMixer \
    --data ETTh1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --features M \
    --save_dir ./checkpoints/hptunning/best_configs/hyperopt_tpe/ETTh1_96_336 \
    --seq_len 96 \
    --label_len 0 \
    --pred_len 336 \
    --down_sampling_layers 3 \
    --down_sampling_window 2 \
    --down_sampling_method avg \
    --decomp_method dft_decomp \
    --e_layers 4 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_model 256 \
    --d_ff 768 \
    --learning_rate 0.0006048636262373894 \
    --train_epochs 10 \
    --patience 10 \
    --batch_size 16 \
    --train_prop 1.0 \
    --dropout 0.11374290594454176 \
    --seed 123;


# bohb

# Trial status: 463 TERMINATED | 7 ERROR
# Current time: 2024-08-24 23:46:39. Total running time: 4hr 0min 0s
# Logical resource usage: 2.0/32 CPUs, 1.0/1 GPUs (0.0/1.0 accelerator_type:G)
# Current best trial: 168ccbf9 with best_valid_loss=1.2811715484790083 and 
# params={
#     'batch_size': 16, 
#     'd_model': 128, 
#     'decomp_method': 'moving_avg', 
#     'down_sampling_method': 'conv', 
#     'dropout': 0.11450445956529968, 
#     'e_layers': 1, 
#     'learning_rate': 0.0012097657749695666, 
#     'moving_avg': 55, 
#     'd_ff': 256
#     }

python3 train_timemixer.py \
    --model TimeMixer \
    --data ETTh1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --features M \
    --save_dir ./checkpoints/hptunning/best_configs/bohb/ETTh1_96_336 \
    --seq_len 96 \
    --label_len 0 \
    --pred_len 336 \
    --down_sampling_layers 3 \
    --down_sampling_window 2 \
    --down_sampling_method conv \
    --decomp_method moving_avg \
    --moving_avg 55 \
    --e_layers 1 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_model 128 \
    --d_ff 256 \
    --learning_rate 0.0012097657749695666 \
    --train_epochs 10 \
    --patience 10 \
    --batch_size 16 \
    --train_prop 1.0 \
    --dropout 0.11450445956529968 \
    --seed 123;


# smac
# Configuration(values={
#   'alpha_d_ff': 3,
#   'batch_size': 16,
#   'd_model': 256,
#   'decomp_method': 'dft_decomp',
#   'down_sampling_method': 'avg',
#   'dropout': 0.0636436683646,
#   'e_layers': 3,
#   'learning_rate': 0.0006530326064,
# })
# Incumbent cost: 1.2724536153130561

python3 train_timemixer.py \
    --model TimeMixer \
    --data ETTh1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --features M \
    --save_dir ./checkpoints/hptunning/best_configs/smac/ETTh1_96_336 \
    --seq_len 96 \
    --label_len 0 \
    --pred_len 336 \
    --down_sampling_layers 3 \
    --down_sampling_window 2 \
    --down_sampling_method avg \
    --decomp_method dft_decomp \
    --e_layers 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_model 256 \
    --d_ff 768 \
    --learning_rate 0.0006530326064 \
    --train_epochs 10 \
    --patience 10 \
    --batch_size 16 \
    --train_prop 1.0 \
    --dropout 0.0636436683646 \
    --seed 123;

# ETTh1_96_720

# random_search

# Trial status: 422 TERMINATED | 11 ERROR
# Current time: 2024-08-23 16:43:17. Total running time: 4hr 0min 0s
# Logical resource usage: 2.0/32 CPUs, 1.0/1 GPUs (0.0/1.0 accelerator_type:G)
# Current best trial: 80f1d_00237 with best_valid_loss=1.5340036047829522 and 
# params={
#     'batch_size': 16, 
#     'learning_rate': 0.004019030012326274, 
#     'down_sampling_method': 'avg', 
#     'd_model': 64, 
#     'decomp_method': 'dft_decomp', 
#     'moving_avg': 15, 
#     'e_layers': 4, 
#     'dropout': 0.1225631105600971, 
#     'd_ff': 192
#     }


python3 train_timemixer.py \
    --model TimeMixer \
    --data ETTh1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --features M \
    --save_dir ./checkpoints/hptunning/best_configs/random_search/ETTh1_96_720 \
    --seq_len 96 \
    --label_len 0 \
    --pred_len 720 \
    --down_sampling_layers 3 \
    --down_sampling_window 2 \
    --down_sampling_method avg \
    --decomp_method dft_decomp \
    --e_layers 4 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_model 64 \
    --d_ff 192 \
    --learning_rate 0.004019030012326274 \
    --train_epochs 10 \
    --patience 10 \
    --batch_size 16 \
    --train_prop 1.0 \
    --dropout 0.1225631105600971 \
    --seed 123;

# hyperopt_tpe

# Trial status: 222 TERMINATED
# Current time: 2024-08-24 11:21:09. Total running time: 4hr 0min 0s
# Logical resource usage: 1.0/32 CPUs, 1.0/1 GPUs (0.0/1.0 accelerator_type:G)
# Current best trial: fbb451de with best_valid_loss=1.5601323318125597 and 
# params={
#     'alpha_d_ff': 3, 
#     'batch_size': 32, 
#     'd_model': 512, 
#     'decomp_method': {'decomp_method': 'dft_decomp'}, 
#     'down_sampling_method': 'avg', 
#     'dropout': 0.19237397793931257, 
#     'e_layers': 1, 
#     'learning_rate': 0.0008450451542293993
#     }


python3 train_timemixer.py \
    --model TimeMixer \
    --data ETTh1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --features M \
    --save_dir ./checkpoints/hptunning/best_configs/hyperopt_tpe/ETTh1_96_720 \
    --seq_len 96 \
    --label_len 0 \
    --pred_len 720 \
    --down_sampling_layers 3 \
    --down_sampling_window 2 \
    --down_sampling_method avg \
    --decomp_method dft_decomp \
    --e_layers 1 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_model 512 \
    --d_ff 1536 \
    --learning_rate 0.0008450451542293993 \
    --train_epochs 10 \
    --patience 10 \
    --batch_size 32 \
    --train_prop 1.0 \
    --dropout 0.19237397793931257 \
    --seed 123;

# bohb
# Trial status: 307 TERMINATED | 3 ERROR
# Current time: 2024-08-25 03:46:54. Total running time: 4hr 0min 0s
# Logical resource usage: 1.0/32 CPUs, 1.0/1 GPUs (0.0/1.0 accelerator_type:G)
# Current best trial: 5027d55f with best_valid_loss=1.5497846497429741 and 
# params={
#     'batch_size': 16, 
#     'd_model': 128, 
#     'decomp_method': 'dft_decomp', 
#     'down_sampling_method': 'conv', 
#     'dropout': 0.07732190337378124, 
#     'e_layers': 3, 
#     'learning_rate': 0.004088098275896603, 
#     'd_ff': 384
#     }


python3 train_timemixer.py \
    --model TimeMixer \
    --data ETTh1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --features M \
    --save_dir ./checkpoints/hptunning/best_configs/bohb/ETTh1_96_720 \
    --seq_len 96 \
    --label_len 0 \
    --pred_len 720 \
    --down_sampling_layers 3 \
    --down_sampling_window 2 \
    --down_sampling_method conv \
    --decomp_method dft_decomp \
    --e_layers 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_model 128 \
    --d_ff 384 \
    --learning_rate 0.004088098275896603 \
    --train_epochs 10 \
    --patience 10 \
    --batch_size 16 \
    --train_prop 1.0 \
    --dropout 0.07732190337378124 \
    --seed 123;

# smac


# Configuration(values={
#   'alpha_d_ff': 3,
#   'batch_size': 16,
#   'd_model': 256,
#   'decomp_method': 'dft_decomp',
#   'down_sampling_method': 'avg',
#   'dropout': 0.0636436683646,
#   'e_layers': 3,
#   'learning_rate': 0.0006530326064,
# })
# Incumbent cost: 1.5548050324122111

python3 train_timemixer.py \
    --model TimeMixer \
    --data ETTh1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --features M \
    --save_dir ./checkpoints/hptunning/best_configs/smac/ETTh1_96_720 \
    --seq_len 96 \
    --label_len 0 \
    --pred_len 720 \
    --down_sampling_layers 3 \
    --down_sampling_window 2 \
    --down_sampling_method avg \
    --decomp_method dft_decomp \
    --e_layers 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_model 256 \
    --d_ff 768 \
    --learning_rate 0.0006530326064 \
    --train_epochs 10 \
    --patience 10 \
    --batch_size 16 \
    --train_prop 1.0 \
    --dropout 0.0636436683646 \
    --seed 123;