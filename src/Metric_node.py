
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
import torch
import os

def _mse(outputs, targets):
    return mean_squared_error(outputs.reshape(-1), targets.reshape(-1))

def _mae(outputs, targets):
    return mean_absolute_error(outputs.reshape(-1), targets.reshape(-1))

def MAPE(pred, true):
    return np.mean(np.abs((true - pred) / true)) * 100

def MSPE(pred, true):
    return np.mean(np.square((true - pred) / true)) * 100

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))

def CORR_uni(pred, true):

    pred = np.sum(pred, axis=(-1))
    true = np.sum(true, axis=(-1))
    
    corrs = []
    for i in range(pred.shape[0]):
        corr = np.corrcoef(pred[i, :], true[i, :])[0, 1]
        if not np.isnan(corr).any():  
            corrs.append(corr)

    return np.mean(corrs)

def metric(outputs, actuals):

    # outputs = outputs.detach().cpu().numpy()
    # actuals = actuals.detach().cpu().numpy()
    
    mse = _mse(outputs, actuals)
    mae = _mae(outputs, actuals)
    cor = CORR_uni(outputs, actuals)
    rse = RSE(outputs, actuals)
    mape = 0
    mspe = 0

    return [mse, mae, cor, rse, mape, mspe]

    
def update(now, save, model, best, metric, em, model_name, seq_output, epoch):

    folder_path = f'results/{model_name}/{em}/{model_name}_{seq_output}_{now.month}{now.day}{now.hour}{now.minute}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    if best[0] > metric[0]: 
        best[0] = metric[0]
        if save:
            torch.save(model.state_dict(), folder_path+f'/{model_name}_{seq_output}.pth')
    if best[1] > metric[1]: best[1] = metric[1]
    if best[2] < metric[2]: best[2] = metric[2]
    if best[3] > metric[3]: best[3] = metric[3]
    if best[4] > metric[4]: best[4] = metric[4]
    if best[5] > metric[5]: best[5] = metric[5]
    # if best[6] > metric[6]: best[6] = metric[6]

    metric = np.array([best[0], best[1], best[2], best[3], best[4], best[5], epoch])
    np.save(folder_path + f'/{model_name}_{seq_output}.npy', metric)

    return best

def plot(pred, true, model_name, seq_output, em, now, flag='train'):
    
    folder_path = f'results/{model_name}/{em}/{model_name}_{seq_output}_{now.month}{now.day}{now.hour}{now.minute}'
    
    idx = [0, pred.shape[-1]*1//4, pred.shape[-1]*1//2, pred.shape[-1]*3//4]
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Cluster Sum Time Flow', fontsize=16)
    titles = ["0", "1/4", "1/2", "3/4"]
    for i, ax in enumerate(axs.flat):
        ax.plot(pred[int(pred.shape[0]*1/2), :, idx[i]], label="prediction", color="r")
        ax.plot(true[int(true.shape[0]*1/2), :, idx[i]], label="actual", color="b")
        ax.set_title(titles[i])
    plt.tight_layout()
    plt.legend()
    if flag == 'train':
        plt.savefig(folder_path+f"/{model_name}_cluster_val.png")
    else:
        plt.savefig(folder_path+f"/{model_name}_cluster_test.png")
    plt.close(fig)

    pred = np.sum(pred, axis=(-1))
    true = np.sum(true, axis=(-1))
    
    idx = [0, pred.shape[0]*1//4, pred.shape[0]*1//2, pred.shape[0]*3//4]
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('All Cluster Time Flow', fontsize=16)
    titles = ["0", "1/4", "1/2", "3/4"]
    for i, ax in enumerate(axs.flat):
        ax.plot(pred[idx[i]], label="prediction", color="r")
        ax.plot(true[idx[i]], label="actual", color="b")
        ax.set_title(titles[i])
    plt.tight_layout()
    plt.legend()
    if flag == 'train':
        plt.savefig(folder_path+f"/{model_name}_val.png")
    else:
        plt.savefig(folder_path+f"/{model_name}_test.png")
    plt.close(fig)

    