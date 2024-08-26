import os
import tempfile
import argparse
import random
import time
import json
from math import log
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from hyperopt import hp
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

# Import TimeMixer utilities
from utils.tools import adjust_learning_rate
from utils.metrics import metric
from tfm_utils.callbacks import ModelCheckpointCallback, EarlyStoppingCallback
from tfm_utils.train_utils import model_dict, _get_data, _select_criterion, _select_optimizer, vali

# Import ray utilities
from ray import train, tune
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
from ray.tune.search.basic_variant import BasicVariantGenerator
from ray.tune.schedulers import FIFOScheduler
from ray.tune.search.hyperopt import HyperOptSearch

# Build parser
parser = argparse.ArgumentParser(description='TimeMixer-RayTune')
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
parser.add_argument('--decomp_method', type=str, default='moving_avg', help='method of series decompsition, only support moving_avg or dft_decomp')
parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--channel_independence', type=int, default=1, help='0: channel dependence 1: channel independence for FreTS model')
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
# RayTune Configuration
parser.add_argument('--tune_search_algorithm', type=str, default="random_search")
parser.add_argument('--tune_trial_scheduler', type=str, default="fifo")
parser.add_argument('--tune_storage_path', type=str, default="./ray_results")
parser.add_argument('--tune_experiment_name', type=str, default='SOFTS_ETTh1_96_96_exp01')
parser.add_argument('--tune_objective', type=str, default="valid_loss")
parser.add_argument('--tune_num_samples', type=int, default=10, help='Max number of hyperparmaeter configurations to be evaluated')
parser.add_argument('--tune_time_budget_s', type=int, default=4*60*60, help='Max running time in seconds')
parser.add_argument('--tune_max_concurrent', type=int, default=1)
parser.add_argument('--tune_gpu_resources', type=float, default=1)
parser.add_argument('--tune_cpu_resources', type=int, default=1)
# Configuration of BOHB search algorithm
parser.add_argument('--tune_hb_eta', type=int, default=3)
parser.add_argument('--tune_bohb_min_points_in_model', type=int, default=None)
parser.add_argument('--tune_bohb_top_n_percent', type=int, default=15)
parser.add_argument('--tune_bohb_num_samples', type=int, default=64)
parser.add_argument('--tune_bohb_random_fraction', type=float, default=0.3333333333333333)
parser.add_argument('--tune_bohb_bandwidth_factor', type=int, default=3)
parser.add_argument('--tune_bohb_min_bandwidth', type=float, default=0.001)
# Configuration of Hyperopt TPE search algorithm
parser.add_argument('--tune_hyperopt_n_initial_points', type=int, default=20)
parser.add_argument('--tune_hyperopt_gamma', type=float, default=0.25)
# Pass search space and default config as dict-alike string (string is converted to dict using json)
parser.add_argument("--tune_param_space", type=str, required=True, help="JSON string of parameter space")
parser.add_argument('--tune_default_config', type=str, default=None)
parser.add_argument('--tune_max_trial_time_s', type=int, default=None)


def f_unpack_dict(dct):
    """
    Unpacks all sub-dictionaries in given dictionary recursively.
    There should be no duplicated keys across all nested
    subdictionaries, or some instances will be lost without warning

    Source: https://www.kaggle.com/fanvacoolt/tutorial-on-hyperopt

    Parameters:
    ----------------
    dct : dictionary to unpack

    Returns:
    ----------------
    : unpacked dictionary
    """

    res = {}
    for k, v in dct.items():
        if isinstance(v, dict):
            res = {**res, **f_unpack_dict(v)}
        else:
            res[k] = v

    return res

def build_param_space(args):
    # Parse the JSON string into a dictionary
    param_space_dict = json.loads(args.tune_param_space)
    # Convert the dictionary into a format compatible with Ray Tune
    if args.tune_search_algorithm == "random_search":
        # Build search space for Random Search algorithm
        param_space = {}
        for key, value in param_space_dict.items():
            dist_type, dist_values = value[0], value[1]
            if dist_type == "choice":
                param_space[key] = tune.choice(dist_values)
            elif dist_type == "loguniform":
                param_space[key] = tune.loguniform(*dist_values)
            elif dist_type == "uniform":
                param_space[key] = tune.uniform(*dist_values)
            elif dist_type == "normal":
                param_space[key] = tune.randn(*dist_values)
            else:
                raise ValueError(f"Unsupported distribution type: {dist_type}")
    elif args.tune_search_algorithm == "hyperopt_tpe":
        # Build search space for Hyperopt algorithms
        param_space = {}
        for key, value in param_space_dict.items():
            dist_type, dist_values = value[0], value[1]
            if dist_type == "choice":
                if isinstance(dist_values[0], list):
                    # Conditional hyperparameter logic
                    values = []
                    for sublist in dist_values:
                        if len(sublist) > 1:
                            values.append({key: sublist[0], sublist[1]: hp.choice(sublist[1], sublist[2])})
                        else:
                            values.append({key : sublist[0]})
                    param_space[key] = hp.choice(key, values)
                else:
                    param_space[key] = hp.choice(key, dist_values)
            elif dist_type == "loguniform":
                param_space[key] = hp.loguniform(key, log(dist_values[0]), log(dist_values[1]))
            elif dist_type == "uniform":
                param_space[key] = hp.uniform(key, *dist_values)
            elif dist_type == "normal":
                param_space[key] = hp.normal(key, *dist_values)
            else:
                raise ValueError(f"Unsupported distribution type: {dist_type}")
    elif args.tune_search_algorithm == "bohb":
        # Build search space for BOHB algorithm using ConfigSpace utilities
        param_space = CS.ConfigurationSpace(seed=args.seed)
        for key, value in param_space_dict.items():
            dist_type, dist_values = value[0], value[1]
            if dist_type == "choice":
                if isinstance(dist_values[0], list):
                    param_space.add_hyperparameter(CSH.CategoricalHyperparameter(key, [sublist[0] for sublist in dist_values]))
                else:
                    param_space.add_hyperparameter(CSH.CategoricalHyperparameter(key, dist_values))
            elif dist_type == "loguniform":
                param_space.add_hyperparameter(CSH.UniformFloatHyperparameter(key, lower=dist_values[0], upper=dist_values[1], log=True))
            elif dist_type == "uniform":
                param_space.add_hyperparameter(CSH.UniformFloatHyperparameter(key, lower=dist_values[0], upper=dist_values[1], log=False))
            elif dist_type == "lognormal":
                param_space.add_hyperparameter(CS.NormalFloatHyperparameter(key, mu=dist_values[0], sigma=dist_values[1], log=True))
            elif dist_type == "normal":
                param_space.add_hyperparameter(CS.NormalFloatHyperparameter(key, mu=dist_values[0], sigma=dist_values[1], log=False))
            else:
                raise ValueError(f"Unsupported distribution type: {dist_type}")
        # Add condition for hyperparameters 'decomp_method' and 'moving_avg'
        key = "decomp_method"
        if key in param_space_dict:
            value = param_space_dict[key]
            dist_type, dist_values = value[0], value[1]
            if isinstance(dist_values[0], list):
                for i in range(len(dist_values)):
                    if dist_values[i][0] == "moving_avg":
                        break
                param_space.add_hyperparameter(CSH.CategoricalHyperparameter(dist_values[i][1], dist_values[i][2]))
                param_space.add_condition(CS.EqualsCondition(param_space.get_hyperparameter(dist_values[i][1]), param_space.get_hyperparameter(key), dist_values[i][0]))
    return param_space

# The raytune 'trainable' function (receives a configuration and the namespace of arguments and returns the objective cost)
def evaluateconfig(config, args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # Unpack hyperopt configuration
    if args.tune_search_algorithm == "hyperopt_tpe":
        config = f_unpack_dict(config)

    # Clean configuration dict
    if "alpha_d_ff" in config:
        config["d_ff"] = int(config["alpha_d_ff"]*config["d_model"])
        del config["alpha_d_ff"]

    print("configuration")
    print(config)

    # Convert Namespace to dict
    args = vars(args)
    # Update dictionary with new values (configuration)
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

    # Checkpoint loading (only useful for Multifidelity Methods such as BOHB)
    checkpoint = get_checkpoint()
    if checkpoint and args.tune_search_algorithm == "bohb":
        print("loading checkpoint...")
        with checkpoint.as_directory() as checkpoint_dir:
            ckpt_path = os.path.join(checkpoint_dir, "last.ckpt")
            # Load checkpoint as dictionary
            checkpoint = torch.load(ckpt_path)
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

    print("start_epoch", start_epoch)
    print("max_epoch", args.train_epochs)
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

        # Checkpointing
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            print("saving checkpoint...")
            # Save metrics dictionary
            metrics_path = os.path.join(checkpoint_dir, "metrics.csv")
            metrics_df.to_csv(metrics_path)
            # Update model checkpoint callback and update best weights if metric improved
            weights_path = os.path.join(checkpoint_dir, "best_weights.pth")
            model_checkpoint(valid_loss, model, weights_path)
            # Build checkpoint content
            checkpoint_data = {
                "args": args,
                'epoch': epoch,
                "best_valid_loss" : best_valid_loss,
                "metrics": metrics,
                "callbacks_best_score": model_checkpoint.best_score,
            }
            if args.patience > 0:
                checkpoint_data.update({
                    "early_stopping_counter": early_stopping.counter,
                    "early_stopping_early_stop": early_stopping.early_stop,
                })
            if args.tune_search_algorithm == "bohb":
                checkpoint_data.update({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': model_optim.state_dict(),
                    'scheduler_state_dict' : scheduler.state_dict(),
                })
            # Save full checkpoint
            ckpt_path = os.path.join(checkpoint_dir, "last.ckpt")
            torch.save(checkpoint_data, ckpt_path)
            # Communicate results to tune
            train.report(
                metrics={"train_loss": train_loss, "valid_loss": valid_loss, "best_valid_loss":best_valid_loss,},
                checkpoint=Checkpoint.from_directory(checkpoint_dir),
            )

        print("Epoch: {} cost time: {}".format(epoch + 1, epoch_time))
        print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Best vali loss: {4:.7f}".format(epoch + 1, train_steps, train_loss, valid_loss, best_valid_loss))
        
        # Check early stopping break condition
        if args.patience > 0:
            if early_stopping.early_stop:
                print("Early stopping")
                break

def runexperiment(train_fn, args):
    # Build trainable function
    def _train_fn(config):
        return train_fn(config, args)
    
    # Build search space
    param_space = build_param_space(args)

    # Specify resources
    resources_dict = {
        "cpu": args.tune_cpu_resources,
        "gpu": args.tune_gpu_resources,
    }

    # Initial configurations
    points_to_evaluate = None
    if args.tune_default_config is not None:
        points_to_evaluate = [json.loads(args.tune_default_config)]

    # Build search algorithm
    if args.tune_search_algorithm == "random_search":
        search_alg = BasicVariantGenerator(
            points_to_evaluate=points_to_evaluate, # Initial parameter suggestions to be run first. This is for when you already have some good parameters you want to run first to help the algorithm make better suggestions for future parameters. Needs to be a list of dicts containing the configurations.
            max_concurrent=args.tune_max_concurrent, # Maximum number of concurrently running trials. If 0 (default), no maximum is enforced.
            constant_grid_search=False, # If this is set to True, Ray Tune will first try to sample random values and keep them constant over grid search parameters. If this is set to False (default), Ray Tune will sample new random parameters in each grid search condition.
            random_state=args.seed, # Seed or numpy random generator to use for reproducible results. If None (default), will use the global numpy random generator (np.random). Please note that full reproducibility cannot be guaranteed in a distributed environment.
        )
    elif args.tune_search_algorithm == "bohb":
        # Bayesian Optimization with HyperBand (BOHB). Requires HpBandSter and ConfigSpace to be installed. This should be used in conjunction with HyperBandForBOHB.
        search_alg = TuneBOHB( 
            space=param_space, # Continuous ConfigSpace search space. Parameters will be sampled from this space which will be used to run trials.
            metric=args.tune_objective,
            mode="min",
            bohb_config={ # configuration for HpBandSter BOHB algorithm
                'min_points_in_model' : args.tune_bohb_min_points_in_model, #  number of observations to start building a KDE. Default ‘None’ means dim+1, the bare minimum.
                'top_n_percent' : args.tune_bohb_top_n_percent, # percentage ( between 1 and 99, default 15) of the observations that are considered good.
                'num_samples' : args.tune_bohb_num_samples, # number of samples to optimize EI (default 64)
                'random_fraction' : args.tune_bohb_random_fraction, # fraction of purely random configurations that are sampled from the prior without the model.
                'bandwidth_factor' : args.tune_bohb_bandwidth_factor, # to encourage diversity, the points proposed to optimize EI, are sampled from a ‘widened’ KDE where the bandwidth is multiplied by this factor (default: 3)
                'min_bandwidth' : args.tune_bohb_min_bandwidth, #  to keep diversity, even when all (good) samples have the same value for one of the parameters, a minimum bandwidth (Default: 1e-3) is used instead of zero.
                }, 
            points_to_evaluate=points_to_evaluate, # Initial parameter suggestions to be run first. This is for when you already have some good parameters you want to run first to help the algorithm make better suggestions for future parameters. Needs to be a list of dicts containing the configurations.
            seed=args.seed, # Optional random seed to initialize the random number generator. Setting this should lead to identical initial configurations at each run.
            max_concurrent=args.tune_max_concurrent, # Number of maximum concurrent trials. If this Searcher is used in a ConcurrencyLimiter, the max_concurrent value passed to it will override the value passed here. Set to <= 0 for no limit on concurrency.
            )
    elif args.tune_search_algorithm == "hyperopt_tpe":
         search_alg = HyperOptSearch(
            space=param_space,
            metric=args.tune_objective, 
            mode="min",
            # points_to_evaluate=points_to_evaluate, # This is for when you already have some good parameters you want to run first to help the algorithm make better suggestions for future parameters. Needs to be a list of dicts containing the configurations.
            n_initial_points=args.tune_hyperopt_n_initial_points, # number of random evaluations of the objective function before starting to aproximate it with tree parzen estimators. Defaults to 20.
            random_state_seed=args.seed, # seed for reproducible results. Defaults to None.
            gamma=args.tune_hyperopt_gamma, # parameter governing the tree parzen estimators suggestion algorithm. Defaults to 0.25.
         )
    else:
        raise NotImplementedError(f"search algorithm {args.tune_search_algorithm} not implemented")
    
    # Build trial scheduler
    if args.tune_search_algorithm == "random_search":
        if args.tune_trial_scheduler == "fifo":
            # FIFO scheduler (First-In-First-Out): simple scheduler that just runs trials in submission order.
            scheduler = FIFOScheduler()
        else:
            raise NotImplementedError(f"scheduler {args.tune_trial_scheduler} not implemented")
    elif args.tune_search_algorithm == "hyperopt_tpe":
        # FIFO scheduler (First-In-First-Out): simple scheduler that just runs trials in submission order.
        scheduler = FIFOScheduler()
    elif args.tune_search_algorithm == "bohb":
        # Scheduler that extends HyperBand early stopping algorithm for BOHB.
        scheduler = HyperBandForBOHB(
            time_attr="training_iteration", # A training result attr to use for comparing time. Note that you can pass in something non-temporal such as `training_iteration` as a measure of progress, the only requirement is that the attribute should increase monotonically.
            max_t=args.train_epochs, # max time units per trial. Trials will be stopped after max_t time units (determined by time_attr) have passed.
            reduction_factor=args.tune_hb_eta, # Used to set halving rate and amount. This is simply a unit-less scalar.
            stop_last_trials=True, # Whether to terminate the trials after reaching max_t.
            )
        
    # Build stopper
    stopper = None
    if args.tune_max_trial_time_s is not None:
        stopper = {"time_total_s": args.tune_max_trial_time_s}

    if args.tune_search_algorithm in {"hyperopt_tpe", "bohb"}:
        param_space = None

    # Tuner is the recommended way of launching hyperparameter tuning jobs with Ray Tune
    tuner = tune.Tuner(
        # The trainable to be tuned.
        trainable=tune.with_resources( # This wrapper allows specification of resource requirements for a specific trainable.
            trainable=_train_fn, # Trainable to wrap.
            resources=resources_dict
            ),
        # Search space of the tuning job
        param_space=param_space,
        # Tuning algorithm specific config
        tune_config=tune.TuneConfig(
            metric=args.tune_objective, # Metric to optimize. This metric should be reported with `tune.report()`. If set, will be passed to the search algorithm and scheduler.
            mode='min', # Must be one of [min, max]. Determines whether objective is minimizing or maximizing the metric attribute. If set, will be passed to the search algorithm and scheduler.
            search_alg=search_alg, # Search algorithm for optimization. Default to random search.
            scheduler=scheduler, # Scheduler for executing the experiment. Choose among FIFO (default), MedianStopping, AsyncHyperBand, HyperBand and PopulationBasedTraining. Refer to ray.tune.schedulers for more options.
            num_samples=args.tune_num_samples, # Number of times to sample from the hyperparameter space.
            max_concurrent_trials=args.tune_max_concurrent, # Not needed in this setting since only one GPU is available and the trainable parameter of this class is defined in this way
            time_budget_s=args.tune_time_budget_s, # Global time budget in seconds after which all trials are stopped.
            reuse_actors=False, #  Whether to reuse actors between different trials when possible. This can drastically speed up experiments that start and stop actors often (e.g., PBT in time-multiplexing mode). This requires trials to have the same resource requirements.
            trial_name_creator=lambda trial: f"trial-{trial.trial_id}", # Optional function that takes in a Trial and returns its name (i.e. its string representation). Be sure to include some unique identifier (such as `Trial.trial_id`) in each trial's name.
            trial_dirname_creator=None, # Optional function that takes in a trial and generates its trial directory name as a string. Be sure to include some unique identifier (such as `Trial.trial_id`) is used in each trial's directory name.
        ),
        # Runtime configuration that is specific to individual trials. If passed, this will overwrite the run config passed to the Trainer (trainable parameter), if applicable.
        # Upon resuming from a training or tuning run checkpoint, Ray Train/Tune will automatically apply the RunConfig from the previously checkpointed run.
        run_config=train.RunConfig(
            name=args.tune_experiment_name, # Name of the trial or experiment. If not provided, will be deduced from the Trainable
            storage_path=os.path.abspath(args.tune_storage_path), # Path where all results and checkpoints are persisted. Can be a local directory or a destination on cloud storage.
            failure_config=None, # Failure mode configuration.
            checkpoint_config=train.CheckpointConfig( # Configurable parameters for defining the checkpointing strategy. Default behavior is to persist all checkpoints to disk. If ``num_to_keep`` is set, the default retention policy is to keep the checkpoints with maximum timestamp, i.e. the most recent checkpoints.
                num_to_keep=1, # Number of checkpoints that are kept in disk per evaluated hyperparameter configuration
                checkpoint_score_attribute=args.tune_objective, # The attribute that will be used to score checkpoints to determine which checkpoints should be kept on disk when there are greater than ``num_to_keep`` checkpoints. This attribute must be a key from the checkpoint dictionary which has a numerical value. Per default, the last checkpoints will be kept.
                checkpoint_score_order='min', # Either "max" or "min".
                ),
            sync_config=None, # Configuration object for syncing. See train.SyncConfig.
            stop=stopper, # Stop conditions to consider. Refer to ray.tune.stopper.Stopper for more info. Stoppers should be serializable.
        ),
    )
    # Runs hyperparameter tuning job as configured and returns a 'ResultGrid' object
    return tuner.fit()

if __name__ == "__main__":

    # Parse arguments
    args = parser.parse_args()

    # Seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # raytune requires absolute paths for checkpointing
    args.root_path = os.path.abspath(args.root_path) 

    # Run optimization
    results = runexperiment(evaluateconfig, args)

    # Query tunning results
    best_result = results.get_best_result() # Get the best result based on objective metric
    print("Best hyperparameters found were: ", best_result.config)
    # Get the best checkpoint corresponding to the best result.
    best_checkpoint = best_result.checkpoint
    # Get a dataframe for the last reported results of all of the trials
    df = results.get_dataframe()
    df.to_csv(os.path.join(args.tune_storage_path, args.tune_experiment_name, "tune_results.csv"))

    # # Note that trials of all statuses are included in the final result grid.
    # # If a trial is not in terminated state, its latest result and checkpoint as
    # # seen by Tune will be provided.

    # def _evaluateconfig(config):
    #     return evaluateconfig(config, args)
    # # Restore experiment from directory
    # restored_tuner = ray.tune.Tuner.restore(
    #     path=os.path.join(args.tune_storage_path, args.tune_experiment_name),
    #     trainable=_evaluateconfig
    #     )
    # restored_results = restored_tuner.get_results()
    # restored_df = restored_results.get_dataframe()
    # restored_df