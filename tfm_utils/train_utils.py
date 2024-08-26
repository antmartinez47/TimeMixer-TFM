from data_provider.data_factory import data_provider
from utils.tools import adjust_learning_rate
from utils.metrics import metric
from .callbacks import EarlyStoppingCallback, ModelCheckpointCallback
from models import TimeMixer
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
import os
import time
import numpy as np
import pandas as pd
import subprocess

model_dict = {
    'TimeMixer': TimeMixer,
}

def get_gpu_memory_by_current_pid():
    # Get the current process ID (PID)
    pid = os.getpid()

    # Run the nvidia-smi command and query memory used by processes
    command = "nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader,nounits"
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, text=True)

    # Split the output into lines
    lines = result.stdout.strip().split("\n")
    
    # Parse each line
    for line in lines:
        line_pid, name, memory = line.split(", ")
        if int(line_pid) == pid:
            print(f"Process: {name} (PID: {pid}) is using {memory} MiB of GPU memory.")
            return int(memory)
    
    print(f"No GPU process found with PID {pid}.")
    return None

def _get_data(args, flag):
    data_set, data_loader = data_provider(args, flag)
    return data_set, data_loader

def _select_optimizer(args, model):
    model_optim = optim.Adam(model.parameters(), lr=args.learning_rate)
    return model_optim

def _select_criterion(args):
    if args.data == 'PEMS':
        criterion = nn.L1Loss()
    else:
        criterion = nn.MSELoss()
    return criterion

def vali(args, model, vali_data, vali_loader, criterion, device):
    total_loss = []
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            if 'PEMS' == args.data or 'Solar' == args.data:
                batch_x_mark = None
                batch_y_mark = None

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
            else:
                if args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            f_dim = -1 if args.features == 'MS' else 0

            pred = outputs.detach()
            true = batch_y.detach()

            if args.data == 'PEMS':
                B, T, C = pred.shape
                pred = pred.cpu().numpy()
                true = true.cpu().numpy()
                pred = vali_data.inverse_transform(pred.reshape(-1, C)).reshape(B, T, C)
                true = vali_data.inverse_transform(true.reshape(-1, C)).reshape(B, T, C)
                mae, mse, rmse, mape, mspe = metric(pred, true)
                total_loss.append(mae)

            else:
                loss = criterion(pred, true)
                total_loss.append(loss.item())

    total_loss = np.average(total_loss)
    model.train()
    return total_loss


def evaluate(args, data, dataloader, model, device):
    preds = []
    trues = []
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(dataloader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            if 'PEMS' == args.data or 'Solar' == args.data:
                batch_x_mark = None
                batch_y_mark = None

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
            else:
                if args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            f_dim = -1 if args.features == 'MS' else 0

            outputs = outputs.detach().cpu().numpy()
            batch_y = batch_y.detach().cpu().numpy()

            pred = outputs
            true = batch_y

            preds.append(pred)
            trues.append(true)

    preds = np.array(preds)
    trues = np.array(trues)
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

    if args.data == 'PEMS':
        B, T, C = preds.shape
        preds = data.inverse_transform(preds.reshape(-1, C)).reshape(B, T, C)
        trues = data.inverse_transform(trues.reshape(-1, C)).reshape(B, T, C)

    mae, mse, rmse, mape, mspe = metric(preds, trues)
    model.train()
    return {"mae":mae, "mse":mse, "rmse":rmse, "mape":mape, "mspe":mspe}


def train(args, model, device, resume_from_checkpoint=False):
    train_data, train_loader = _get_data(args, flag='train')
    vali_data, vali_loader = _get_data(args, flag='val')
    test_data, test_loader = _get_data(args, flag='test')

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args.best_weights_path = os.path.join(args.save_dir, "best_model.pth")
    if args.save_last:
        args.last_checkpoint_path = os.path.join(args.save_dir, "last.ckpt")
    args.logs_path = os.path.join(args.save_dir, "metrics.csv")

    time_now = time.time()

    train_steps = len(train_loader)
    # Callbacks
    model_checkpoint = ModelCheckpointCallback(args.best_weights_path, delta=args.delta, verbose=True)
    if args.patience > 0:
        early_stopping = EarlyStoppingCallback(patience=args.patience, delta=args.delta, verbose=True)

    model_optim = _select_optimizer(args, model)
    criterion = _select_criterion(args)

    scheduler = lr_scheduler.OneCycleLR(
        optimizer=model_optim,
        steps_per_epoch=train_steps,
        pct_start=args.pct_start,
        epochs=args.train_epochs,
        max_lr=args.learning_rate)

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    t00 = time.time()
    start_epoch = 0
    metrics = {"epoch": [], "train_loss": [], "valid_loss": [], "time_epoch_s": []}
    elapsed_time = 0
    if resume_from_checkpoint:
        # Load checkpoint
        checkpoint = torch.load(args.last_checkpoint_path)
        args = checkpoint["args"]
        # Restore model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        # Restore optimizer state
        model_optim.load_state_dict(checkpoint['optimizer_state_dict'])
        # Restore modelcheckpoint callback state
        modelcheckpoint_state_dict = checkpoint['modelcheckpoint_statedict']
        model_checkpoint.verbose = modelcheckpoint_state_dict["verbose"]
        model_checkpoint.best_score = modelcheckpoint_state_dict["best_score"]
        model_checkpoint.delta = modelcheckpoint_state_dict["delta"]
        model_checkpoint.path = modelcheckpoint_state_dict["path"]
        # Restore early stopping callback state
        if "early_stopping_state_dict" in checkpoint:
            early_stopping_state_dict = checkpoint['early_stopping_state_dict']
            early_stopping.patience = early_stopping_state_dict['patience']
            early_stopping.verbose = early_stopping_state_dict['verbose']
            early_stopping.counter = early_stopping_state_dict['counter']
            early_stopping.best_score = early_stopping_state_dict['best_score']
            early_stopping.early_stop = early_stopping_state_dict['early_stop']
            early_stopping.delta = early_stopping_state_dict['delta']
        # Epoch at which training is to be resumed
        start_epoch = checkpoint['epoch'] + 1
        # Manually restore the learning rate if necessary
        for param_group in model_optim.param_groups:
            param_group['lr'] = checkpoint['learning_rate']
        # Restore scheduler
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        # Restore metrics dict
        metrics = checkpoint["metrics"]
        elapsed_time = checkpoint["elapsed_time"]

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
        print("Epoch: {} cost time: {}".format(epoch + 1, epoch_time))
        train_loss = np.average(train_loss)
        vali_loss = vali(args, model, vali_data, vali_loader, criterion, device)
        print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(epoch + 1, train_steps, train_loss, vali_loss))

        if args.lradj != 'TST':
            adjust_learning_rate(model_optim, scheduler, epoch + 1, args, printout=True)
        else:
            print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        # Update callback states
        model_checkpoint(vali_loss, model)
        if args.patience > 0:
            early_stopping(vali_loss)

        elapsed_time += time.time() - t00
        metrics["epoch"].append(epoch + 1)
        metrics["train_loss"].append(train_loss)
        metrics["valid_loss"].append(vali_loss)
        metrics["time_epoch_s"].append(epoch_time)
        metrics_df = pd.DataFrame(metrics, index=range(len(metrics["epoch"])))
        metrics_df.to_csv(args.logs_path)

        if args.save_last:
            checkpoint = {
                "args": args,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': model_optim.state_dict(),
                'train_loss': train_loss,
                'val_loss': vali_loss,
                'learning_rate': model_optim.param_groups[0]['lr'],
                "metrics": metrics,
                "elapsed_time": elapsed_time, 
                "scheduler_state_dict": scheduler.state_dict(),
                'modelcheckpoint_statedict':{
                    "verbose": model_checkpoint.verbose,
                    "best_score": model_checkpoint.best_score,
                    "delta": model_checkpoint.delta,
                    "path": model_checkpoint.path,
                }
            }
            if args.patience > 0:
                checkpoint.update({
                'early_stopping_state_dict': {
                    "patience": early_stopping.patience,
                    "verbose": early_stopping.verbose,
                    "counter": early_stopping.counter,
                    "best_score": early_stopping.best_score,
                    "early_stop": early_stopping.early_stop,
                    "delta": early_stopping.delta,
                },
                })
            torch.save(checkpoint, args.last_checkpoint_path)
        
        # Check early stopping break condition
        if args.patience > 0:
            if early_stopping.early_stop:
                print("Early stopping")
                break
        
        if epoch == args.break_at - 1:
            do_test = False
            break
        else:
            do_test = True

    if do_test:
        print('#####   loading best weights   #####')
        model.load_state_dict(torch.load(args.best_weights_path))

        # Build metrics dataframe for best weights
        metrics_train = evaluate(args, train_data, train_loader, model, device)
        metrics_val = evaluate(args, vali_data, vali_loader, model, device)
        metrics_test = evaluate(args, test_data, test_loader, model, device)
        vram_usage_gb = get_gpu_memory_by_current_pid() / 1024  # in Gb
        columns = [
            "train_mse", "train_mae", "val_mse", 
            "val_mae", "test_mse", "test_mae", 
            "elapsed_time", "vram_usage_gb"
        ]
        data = [
            metrics_train["mse"], metrics_train["mae"], metrics_val["mse"], 
            metrics_val["mae"], metrics_test["mse"], metrics_test["mae"],
            elapsed_time, vram_usage_gb
        ]
        best_metrics_df = pd.DataFrame([data], columns=columns)
        best_metrics_df.to_csv(args.logs_path.replace("metrics", "best_metrics"))
        print("Epoch: {0} | Elapsed Time: {1} s | VRAM usage: {2} Gb | Train MSE: {3:.4f} Train MAE: {4:.4f} Vali MSE: {5:.4f} Vali MAE: {6:.4f} Test MSE: {7:.4f} Test MAE: {8:.4f}".format(
                epoch + 1, elapsed_time, vram_usage_gb, metrics_train["mse"], metrics_train["mae"], metrics_val["mse"], metrics_val["mae"], metrics_test["mse"], metrics_test["mae"]))
        print("\n\n")

    return args