import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')  # 确保整个程序使用无界面（non-interactive）的 Agg 后端，避免因 Tkinter 相关的 GUI 清理而导致错误
import matplotlib.pyplot as plt
import time
import socket


def get_internal_ip():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Use Google's public DNS server to determine internal IP
        # The IP address 8.8.8.8 is used here as a placeholder and does not need to be reachable
        sock.connect(('8.8.8.8', 80))
        internal_ip = sock.getsockname()[0]
    except Exception:
        internal_ip = '127.0.0.1'
    finally:
        sock.close()
    return internal_ip


def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate*0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate*0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate*0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate*0.1}  
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout: print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.update_model_flag = False

    def __call__(self, val_loss, model, path, early_stop_epoch, is_cluster=0):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            if is_cluster:
                for model_index in range(len(model)):  # 此时的model是一个list，里面有K个model，K为cluster个数
                    self.save_checkpoint(val_loss, model, path, is_cluster, model_index)
            else:
                self.save_checkpoint(val_loss, model, path)
            self.early_stop_epoch = early_stop_epoch
            self.update_model_flag = True
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
            self.update_model_flag = False
        else:
            self.best_score = score
            if is_cluster:
                for model_index in range(len(model)):
                    self.save_checkpoint(val_loss, model, path, is_cluster, model_index)
            else:
                self.save_checkpoint(val_loss, model, path)
            self.early_stop_epoch = early_stop_epoch
            self.counter = 0
            self.update_model_flag = True

    def save_checkpoint(self, val_loss, model, path, is_cluster=0, model_index=0):
        if is_cluster:
            if self.verbose and model_index==0:
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  '
                      f'Saving model for cluster model...')
            torch.save(model[model_index].state_dict(), path + '/' + f'checkpoint_cluster_{model_index}.pth')
        else:
            if self.verbose:
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  '
                      f'Saving model ...')
            torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')

def test_params_flop(model,x_shape):
    """
    If you want to thest former's flop, you need to give default value to inputs in model.forward(), the following code can only pass one argument to forward()
    """
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
        print('INFO: Trainable parameter count: {:.2f}M'.format(model_params / 1000000.0))
    from ptflops import get_model_complexity_info    
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model.cuda(), x_shape, as_strings=True, print_per_layer_stat=True)
        # print('Flops:' + flops)
        # print('Params:' + params)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


def plot_loss_curve(train_loss, valid_loss, test_loss, early_stop_epoch, name='loss_curve.pdf'):
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(train_loss))+1, train_loss, label='Train Loss')
    plt.plot(np.arange(len(train_loss))+1, valid_loss, label='Validation Loss')
    plt.plot(np.arange(len(train_loss))+1, test_loss, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss Over Epochs')
    plt.legend()
    max_th = np.max(np.array([train_loss, valid_loss, test_loss]))
    min_th = np.min(np.array([train_loss, valid_loss, test_loss]))
    plt.vlines(early_stop_epoch, min_th*0.9, max_th*1.1, linestyles='dashed')
    plt.savefig(name, bbox_inches='tight')
