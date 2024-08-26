# SMAC needs four core components to run an optimization process: configuration space, target function, scenario and facade

# Global imports
import os
import argparse
import random
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

# TimeMixer utilities
from utils.tools import adjust_learning_rate
from utils.metrics import metric
from tfm_utils.callbacks import ModelCheckpointCallback, EarlyStoppingCallback
from tfm_utils.train_utils import model_dict, _get_data, _select_criterion, _select_optimizer, vali

# ConfigSpace utilities
from ConfigSpace import (
    Configuration,
    ConfigurationSpace,
    Categorical,
    Float,
    Normal,
    EqualsCondition
)

# SMAC utilities
from smac import Scenario
# Select a SMAC facade
# A facade is the entry point to SMAC, which constructs a default optimization pipeline. 
# SMAC offers various facades, which satisfy many common use cases and are crucial to achieving peak performance. 
# Most of the available facades subclass smac.facade.abstract_facade.AbstractFacade
from smac import MultiFidelityFacade as MFFacade
# Select a SMAC intensifier (e.g SucessiveHalving or Hyperband, required by MultifidelityFacade)
from smac.intensifier.hyperband import Hyperband

# Build parser
parser = argparse.ArgumentParser(description='TimeMixer-SMAC')
# Basic config
parser.add_argument('--model', type=str, required=True, default='TimeMixer', help='model name, options: [TimeMixer]')
parser.add_argument('--task_name', type=str, required=False, default='long_term_forecast', help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--seed', type=int, default=2021)
# Data loader
parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/ETT-small/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--train_prop', type=float, default=1.0)
# Forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
# Model
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--channel_independence', type=int, default=1, help='0: channel dependence 1: channel independence for FreTS model')
parser.add_argument('--decomp_method', type=str, default='moving_avg', help='method of series decompsition, only support moving_avg or dft_decomp')
parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
parser.add_argument('--down_sampling_method', type=str, default='avg', help='down sampling method, only support avg, max, conv')
parser.add_argument('--use_future_temporal_feature', type=int, default=0, help='whether to use future_temporal_feature; True 1 False 0')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
# Optimization
parser.add_argument('--num_workers', type=int, default=4, help='data loader num workers')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--delta', type=float, default=0.0, help='early stopping min delta')
parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--comment', type=str, default='none', help='com')
# GPU
parser.add_argument('--use_gpu', action="store_true", default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
# SMAC configuration
parser.add_argument('--smac_storage_path', type=str, default="./ray_results")
parser.add_argument('--smac_experiment_name', type=str, default='SOFTS_ETTh1_96_96_exp01')
parser.add_argument('--smac_n_trials', type=int, default=10, help='Max number of hyperparmaeter configurations to be evaluated')
parser.add_argument('--smac_time_budget_s', type=int, default=4*60*60, help='Max running time in seconds')
parser.add_argument('--smac_n_workers', type=int, default=1)
parser.add_argument('--smac_deterministic', action="store_true", default=False)
parser.add_argument('--smac_trial_walltime_limit', type=int, default=None)
parser.add_argument('--smac_trial_memory_limit', type=int, default=None)
parser.add_argument('--smac_min_budget', type=int, default=1)
# Configuration for SMAC intensifier
parser.add_argument('--smac_eta', type=int, default=3)
parser.add_argument('--smac_incumbent_selection', type=str, default="highest_budget")
# Single argument to pass the entire search space as a JSON string
parser.add_argument("--smac_param_space", type=str, required=True, help="JSON string of parameter space")
parser.add_argument('--smac_default_config', type=str, default=None)
parser.add_argument('--smac_n_init_configs', type=int, default=5)

def build_param_space(args):
    hps = []
    # Parse the JSON string into a dictionary
    param_space_dict = json.loads(args.smac_param_space)
    # Build search space for BOHB algorithm using ConfigSpace utilities
    param_space = ConfigurationSpace(seed=args.seed)
    hps_list = []
    for key, value in param_space_dict.items():
        dist_type, dist_values = value[0], value[1]
        if dist_type == "choice":
            if isinstance(dist_values[0], list):
                hps_list.append(Categorical(key, [sublist[0] for sublist in dist_values]))
            else:
                hps_list.append(Categorical(key, dist_values))
        elif dist_type == "loguniform":
            hps_list.append(Float(key, (dist_values[0], dist_values[1]), log=True))
        elif dist_type == "uniform":
            hps_list.append(Float(key, (dist_values[0], dist_values[1]), log=False))
        elif dist_type == "normal":
            hps_list.append(Float(key, (dist_values[0], dist_values[1]), distribution=Normal(dist_values[2], dist_values[3]), log=False))
        else:
            raise ValueError(f"Unsupported distribution type: {dist_type}")
        hps.append(key)
    # Add all hyperparameters at once
    param_space.add(hps_list)
    # Add condition for hyperparameters 'decomp_method' and 'moving_avg'
    key = "decomp_method"
    if key in param_space_dict:
        value = param_space_dict[key]
        dist_type, dist_values = value[0], value[1]
        if isinstance(dist_values[0], list):
            for i in range(len(dist_values)):
                if dist_values[i][0] == "moving_avg":
                    break
            param_space.add(Categorical(dist_values[i][1], dist_values[i][2]))
            param_space.add(EqualsCondition(child=param_space.get_hyperparameter(dist_values[i][1]), parent=param_space.get_hyperparameter(key), value=dist_values[i][0]))
            hps.append('moving_avg')
    return param_space, hps

def config2dict(config: Configuration) -> dict:
    # Convert configuration to dict
    config = dict(config)
    # Convert numpy.integers to base integers to avoid runtime errors
    _config = config.copy()
    config = {
        k: int(v) if isinstance(v, (np.integer)) else v
        for k, v in _config.items()
    }
    return config

# Define target function
# The target function takes a configuration from the configuration space and returns a performance value. 
def train(config: Configuration, seed: int = 0, budget: int = 5) -> float:
    # parse args
    args = parser.parse_args()

    # Set seeds to ensure full reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # Parse Configuration object
    config = config2dict(config)
    
    config2id_path = os.path.join(os.path.join(args.smac_storage_path, args.smac_experiment_name, "config2id.json"))
    if not os.path.exists(config2id_path):
        config2id = {}
    else:
        with open(config2id_path, "r") as f:
            config2id = json.load(f)

    config_str = json.dumps(config)
    if config_str not in config2id:
        config2id[config_str] = len(config2id)+1

    # Save config2id dict
    with open(config2id_path, "w") as f:
        json.dump(config2id, f, indent=4)

    config_id = config2id[config_str]

    checkpoint_dir = os.path.join(args.smac_storage_path, args.smac_experiment_name, "checkpoints", f"trial-{config_id}")
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Clean configuration dict
    if "alpha_d_ff" in config:
        config["d_ff"] = int(config["alpha_d_ff"]*config["d_model"])
        del config["alpha_d_ff"]
    
    print(config)

    # Convert argparse Namespace to dictionary
    args = vars(args)
    args.update(config)
    # Convert back to Namespace
    args = argparse.Namespace(**args)

    # Set device configuration
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device = torch.device('cuda:{}'.format(args.gpu))
        print('Use GPU: cuda:{}'.format(args.gpu))
    else:
        device = torch.device('cpu')
        print('Use CPU')

    # Build model
    model = model_dict[args.model].Model(args).float()
    if args.use_multi_gpu and args.use_gpu:
        model = torch.nn.DataParallel(model, device_ids=args.device_ids)
    model = model.to(device)

    train_data, train_loader = _get_data(args, flag='train')
    vali_data, vali_loader = _get_data(args, flag='val')

    time_now = time.time()
    train_steps = len(train_loader)
    # Build callbacks
    model_checkpoint = ModelCheckpointCallback(None, delta=args.delta, verbose=True, tune=True)
    if args.patience > 0:
        early_stopping = EarlyStoppingCallback(patience=args.patience, delta=args.delta, verbose=True)

    # Build optimizer and loss
    model_optim = _select_optimizer(args, model)
    criterion = _select_criterion(args)
    # tune_criterion = torch.nn.L1Loss() # MAE as default loss in order to use the same metric as objective when tunning loss function

    # Build LR Scheduler
    scheduler = lr_scheduler.OneCycleLR(
        optimizer=model_optim,
        steps_per_epoch=train_steps,
        pct_start=args.pct_start,
        epochs=args.train_epochs,
        max_lr=args.learning_rate)

    # Set AMP configuration
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    # Checkpoint loading
    checkpoint_path = os.path.join(checkpoint_dir, "last.ckpt")
    if os.path.exists(checkpoint_path):
        print("loading checkpoint...")
        # Load checkpoint as dictionary
        checkpoint = torch.load(checkpoint_path)
        # Restore model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        # Restore optimizer state (learning rate, weights...)
        model_optim.load_state_dict(checkpoint['optimizer_state_dict'])
        # Restore LR Scheduler state
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        # Restore modelcheckpoint state
        model_checkpoint.best_score = checkpoint['callbacks_best_score']
        # Restore early stopping state
        if args.patience > 0:
            early_stopping.counter = checkpoint['early_stopping_counter']
            early_stopping.best_score = checkpoint['callbacks_best_score']
            early_stopping.early_stop = checkpoint['early_stopping_early_stop']
        # Restore epoch at which training is to be resumed
        start_epoch = checkpoint['epoch'] + 1
        # Restore metrics dict
        metrics = checkpoint["metrics"]
        # Restore best validation loss
        best_valid_loss = checkpoint["best_valid_loss"]
    else:
        start_epoch = 0
        metrics = {"epoch": [], "train_loss": [], "valid_loss": [], "time_epoch_s": []}
        best_valid_loss = np.inf

    # Set budget as train epochs
    if budget is not None and budget > 0:
        args.train_epochs = round(budget)

    print("starting training loop")
    print("budget", budget)
    print("start_epoch", start_epoch)
    print("train_epoch", args.train_epochs)
    # Training loop
    for epoch in range(start_epoch, args.train_epochs):
        iter_count = 0
        train_loss = []
        model.train()
        epoch_time = time.time()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            iter_count += 1
            model_optim.zero_grad()

            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            if 'PEMS' == args.data or 'Solar' == args.data:
                batch_x_mark = None
                batch_y_mark = None

            # decoder input
            if args.down_sampling_layers == 0:
                dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
            else:
                dec_inp = None

            # encoder - decoder
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.output_attention:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if args.features == 'MS' else 0
                    outputs = outputs[:, -args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())
            else:
                if args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if args.features == 'MS' else 0

                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

            if (i + 1) % 100 == 0:
                print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()

            # Backpropagate and optimize
            if args.use_amp:
                scaler.scale(loss).backward()
                scaler.step(model_optim)
                scaler.update()
            else:
                loss.backward()
                model_optim.step()

            if args.lradj == 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, args, printout=False)
                scheduler.step()

        epoch_time = time.time() - epoch_time
        # Compute training and validation loss
        train_loss = np.average(train_loss)
        valid_loss = vali(args, model, vali_data, vali_loader, criterion, device)

        # Update metrics dictionary
        metrics["epoch"].append(epoch + 1)
        metrics["train_loss"].append(train_loss)
        metrics["valid_loss"].append(valid_loss)
        metrics["time_epoch_s"].append(epoch_time)
        metrics_df = pd.DataFrame(metrics, index=range(len(metrics["epoch"])))

        # Update Learning Rate Scheduler
        if args.lradj != 'TST':
            adjust_learning_rate(model_optim, scheduler, epoch + 1, args, printout=True)
        else:
            print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        # Update EarlyStopping callback
        if args.patience > 0:
            early_stopping(valid_loss)
        
        # Update best validation loss
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss

        # Update model checkpoint callback
        weights_path = os.path.join(checkpoint_dir, "best_weights.pth")
        model_checkpoint(valid_loss, model, weights_path)

        # Log epoch info
        print("Epoch: {} cost time: {}".format(epoch + 1, epoch_time))
        print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Best vali loss: {4:.7f}".format(epoch + 1, train_steps, train_loss, valid_loss, best_valid_loss))
        
        # Check early stopping break condition
        if args.patience > 0:
            if early_stopping.early_stop:
                print("Early stopping")
                break

    if args.train_epochs - start_epoch > 0:

        # Checkpointing at the end of last epoch (max budget)
        metrics_df = pd.DataFrame(metrics, index=range(len(metrics["epoch"])))
        # Save metrics-per-epoch dictionary
        metrics_path = os.path.join(checkpoint_dir, "metrics.csv")
        metrics_df.to_csv(metrics_path)
        # Build checkpoint content
        checkpoint_data = {
                "args": args,
                'epoch': epoch,
                "best_valid_loss" : best_valid_loss,
                "metrics": metrics,
                "callbacks_best_score": model_checkpoint.best_score,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': model_optim.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }
        if args.patience > 0:
            checkpoint_data.update({
                "early_stopping_counter": early_stopping.counter,
                "early_stopping_early_stop": early_stopping.early_stop,
            })
        # Save full checkpoint
        torch.save(checkpoint_data, checkpoint_path)

    print()
    print("best validation loss", best_valid_loss)
    print()

    # Report cost to SMAC
    return best_valid_loss
    
if __name__ == "__main__":
    # Parse arguments
    args = parser.parse_args()

    # Set seeds to ensure full reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # Build configuration space
    # The configuration space defines the search space of the hyperparameters and, therefore, the tunable parameters’ legal ranges and default values.
    configspace, hps = build_param_space(args)
    # Instantiate Scenario object specifying the optimization environment
    # The maximum budget in this setting is given by the number of epochs
    scenario = Scenario(
        configspace=configspace, # The configuration space from which to sample the configurations.
        name=args.smac_experiment_name, # The name of the run. If no name is passed, SMAC generates a hash from the meta data.Specify this argument to identify your run easily.
        output_directory=args.smac_storage_path, # Path, defaults to Path("smac3_output") The directory in which to save the output. The files are saved in `./output_directory/name/seed`.
        deterministic=args.smac_deterministic, # bool, defaults to False If deterministic is set to true, only one seed is passed to the target function. Otherwise, multiple seeds (if n_seeds of the intensifier is greater than 1) are passed to the target function to ensure generalization.
        objectives='cost', # str | list[str] | None, defaults to "cost" The objective(s) to optimize. This argument is required for multi-objective optimization.
        crash_cost=np.inf, # float | list[float], defaults to np.inf Defines the cost for a failed trial. In case of multi-objective, each objective can be associated with a different cost.
        termination_cost_threshold=np.inf, # float | list[float], defaults to np.inf Defines a cost threshold when the optimization should stop. In case of multi-objective, each objective *must* be associated with a cost. The optimization stops when all objectives crossed the threshold.
        walltime_limit=args.smac_time_budget_s, # float, defaults to np.inf The maximum time in seconds that SMAC is allowed to run.
        cputime_limit=np.inf, # float, defaults to np.inf The maximum time in seconds that SMAC is allowed to run.
        trial_walltime_limit=args.smac_trial_walltime_limit, # float | None, defaults to None The maximum time in seconds that a trial is allowed to run. If not specified, no constraints are enforced. Otherwise, the process will be spawned by pynisher.
        trial_memory_limit=args.smac_trial_memory_limit, # int | None, defaults to None The maximum memory in MB that a trial is allowed to use. If not specified, no constraints are enforced. Otherwise, the process will be spawned by pynisher.
        n_trials=args.smac_n_trials, # int, defaults to 100 The maximum number of trials (combination of configuration, seed, budget, and instance, depending on the task) to run.
        use_default_config=False, # bool, defaults to False. If True, the configspace's default configuration is evaluated in the initial design. For historic benchmark reasons, this is False by default. Notice, that this will result in n_configs + 1 for the initial design. Respecting n_trials, this will result in one fewer evaluated configuration in the optimization.
        instances=None, # list[str] | None, defaults to None Names of the instances to use. If None, no instances are used. Instances could be dataset names, seeds, subsets, etc.
        instance_features=None, # dict[str, list[float]] | None, defaults to None Instances can be associated with features. For example, meta data of the dataset (mean, var, ...) can be incorporated which are then further used to expand the training data of the surrogate model.
        min_budget=args.smac_min_budget, # float | int | None, defaults to None The minimum budget (epochs, subset size, number of instances, ...) that is used for the optimization. Use this argument if you use multi-fidelity or instance optimization.
        max_budget=args.train_epochs, # float | int | None, defaults to None The maximum budget (epochs, subset size, number of instances, ...) that is used for the optimization. Use this argument if you use multi-fidelity or instance optimization.
        seed=args.seed, # int, defaults to 0 The seed is used to make results reproducible. If seed is -1, SMAC will generate a random seed.
        n_workers=args.smac_n_workers, # int, defaults to 1 The number of workers to use for parallelization. If `n_workers` is greather than 1, SMAC will use Dask to parallelize the optimization.
        )
    
    # Add default config to the list of additional configs
    # The configurations from this list will be evaluated before optimization begins as part of the warm-up phase of the surrogate model
    additional_configs = None
    if args.smac_default_config is not None:
        additional_configs = [Configuration(configspace, json.loads(args.smac_default_config))]
    
    # Specify initial configurations
    initial_design = MFFacade.get_initial_design(
        scenario=scenario, 
        n_configs=args.smac_n_init_configs, # Number of initial configurations (disables the arguments ``n_configs_per_hyperparameter``)
        max_ratio=0.25, # Use at most ``scenario.n_trials`` * ``max_ratio`` number of configurations in the initial design. Additional configurations are not affected by this parameter.
        additional_configs=additional_configs, #Adds additional configurations to the initial design.
    )

    # Build intensifier
    intensifier = Hyperband(
        scenario, 
        eta=args.smac_eta, # int, defaults to 3. Input that controls the proportion of configurations discarded in each round of Successive Halving.
        incumbent_selection=args.smac_incumbent_selection, # str, defaults to "highest_observed_budget". How to select the incumbent when using budgets.
        seed=args.seed, # int, defaults to None. Internal seed used for random events like shuffle seeds.
    )

    # Build SMAC facade and pass the scenario and the train method
    facade = MFFacade(
        scenario=scenario, 
        target_function=train, 
        initial_design=initial_design, 
        intensifier=intensifier, 
        overwrite=True, 
        )
    
    # Use SMAC to find the best hyperparameters
    incumbent = facade.optimize()

    # Get cost of default configuration
    default_cost = facade.validate(configspace.get_default_configuration())
    print(f"Default cost: {default_cost}")

    # Get cost of the incumbent (best configuration)
    incumbent_cost = facade.validate(incumbent)
    print(incumbent)
    print(f"Incumbent cost: {incumbent_cost}")

    config2id_path = os.path.join(os.path.join(args.smac_storage_path, args.smac_experiment_name, "config2id.json"))
    with open(config2id_path, "rb") as f:
        config2id = json.load(f)

    cols = ['id', 'seed', 'budget', 'train_loss', 'valid_loss', 'cost', 'time_s', 'status', 'start_time', 'end_time'] + [f"config_{i}" for i in hps]
    results_dict = {i: [] for i in cols}
    # Iterate over all trials
    for trial_info, trial_value in facade.runhistory.items():
        # Trial info
        config_id = trial_info.config_id
        budget = trial_info.budget
        seed = args.seed # trial_info.seed

        config = config2dict(facade.runhistory.get_config(config_id))
        _config_id = config2id[json.dumps(config)]

        # Trial value
        cost = trial_value.cost
        time_taken = trial_value.time
        status = str(trial_value.status)
        starttime = trial_value.starttime
        endtime = trial_value.endtime
        checkpoint_dir = os.path.join(args.smac_storage_path, args.smac_experiment_name, "checkpoints", f"trial-{_config_id}")
        
        # Load metrics df
        metrics_path = os.path.join(checkpoint_dir, "metrics.csv")
        if os.path.exists(metrics_path):
            metrics_df = pd.read_csv(metrics_path)
            idx_min = np.argmin(metrics_df["valid_loss"])
            train_loss = metrics_df["train_loss"][idx_min]
            valid_loss = metrics_df["valid_loss"][idx_min]
        else:
            train_loss = None
            valid_loss = None

        results_dict["id"].append(_config_id)
        results_dict["seed"].append(seed)
        results_dict["budget"].append(budget)
        results_dict["train_loss"].append(train_loss)
        results_dict["valid_loss"].append(valid_loss)
        results_dict["cost"].append(cost)
        results_dict["time_s"].append(time_taken)
        results_dict["status"].append(status)
        results_dict["start_time"].append(starttime)
        results_dict["end_time"].append(endtime)

        for hp in hps:
            results_dict[f"config_{hp}"].append(config.get(hp, None))

    df_results = pd.DataFrame(results_dict)
    df_results.to_csv(os.path.join(args.smac_storage_path, args.smac_experiment_name, "results.csv"))