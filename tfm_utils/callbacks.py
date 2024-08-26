import numpy as np
import torch

class EarlyStoppingCallback:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = np.inf
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss):
        if val_loss >= self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0

class ModelCheckpointCallback:
    def __init__(self, path, delta=0, verbose=False, tune=False):
        self.verbose = verbose
        self.best_score = np.inf
        self.delta = delta
        self.path = path
        self.tune = tune

    def __call__(self, val_loss, model, path=None):
        if val_loss < self.best_score - self.delta:
            if self.tune:
                self.save_weights(self.best_score, val_loss, model, path)
            else:
                self.save_weights(self.best_score, val_loss, model)
            self.best_score = val_loss

    def save_weights(self, old_score, new_score, model, path=None):
        if self.verbose:
            print(f'Validation loss decreased ({old_score:.4f} --> {new_score:.4f}).  Saving model state dict ...')
        if path is not None:
            torch.save(model.state_dict(), path)
        else:
            torch.save(model.state_dict(), self.path)