# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 20:04:01 2025

@author: Younghwi Kim
"""
import torch
from torch import nn
from torch import optim

import SToT.exp as EXP
import src.ASTD_DataLoader_node as ASTD_DataLoader

from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import argparse

def get_args():
    
    parser = argparse.ArgumentParser(description='SToT')
    
    parser.add_argument('--model_name', type=str, default='SToT')
    parser.add_argument('--features', type=str, default='MS',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--freq', type=str, default='d',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--seq_len', type=int, default=360, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=360, help='start token length') # no longer needed in inverted Transformers
    parser.add_argument('--pred_len', type=int, default=360, help='prediction sequence length')

    parser.add_argument('--use_norm', type=bool, default=True, help='ReVIN')
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
    parser.add_argument('--root_path', type=str, default='data/geo_43.npy', help='root path of the geo data file')
    parser.add_argument('--path', type=str, default='data/astd_array_43.npy', help='path of the data file')
    parser.add_argument('--node_num', type=int, default=43, help='number of nodes')
    parser.add_argument('--loss', type=str, default='mse')
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--emission', type=int, default=3)  # 0-CO, 1-CO2, 2-so2(X), 3-NOx, 4-N2O, 5-NMVOC, 6-CH4
    parser.add_argument('--save', type=bool, default=True, help='model save')
    parser.add_argument('--log_interval', type=int, default=15)
    
    # model define
    parser.add_argument('--enc_in', type=int, default=43, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=43, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=1, help='output size') # applicable on arbitrary number of variates in inverted Transformers
    parser.add_argument('--d_model', type=int, default=256, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=256*4, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=10, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', default=False, help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
    
    # optimization
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--lradj', type=str, default='cos_step', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='optimizer learning rate')    
    parser.add_argument('--decay_fac', type=float, default=0.75)
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    
    # cosin decay
    parser.add_argument('--cos_warm_up_steps', type=int, default=10)
    parser.add_argument('--cos_max_decay_steps', type=int, default=300)
    parser.add_argument('--cos_max_decay_epoch', type=int, default=10)
    parser.add_argument('--cos_max', type=float, default=1e-4)
    parser.add_argument('--cos_min', type=float, default=2e-6)
    
    return parser.parse_args()


def main():
    
    args = get_args()
    
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'{args.device} is available')
    
    now = datetime.now()
    
    train_data, train_loader = ASTD_DataLoader.data_provider(args.path, args.batch_size, args.seq_len, args.pred_len, flag='train')
    val_data, val_loader = ASTD_DataLoader.data_provider(args.path, args.batch_size, args.seq_len, args.pred_len, flag='val')
    test_data, test_loader = ASTD_DataLoader.data_provider(args.path, args.batch_size, args.seq_len, args.pred_len, flag='test')
    
    model = EXP.build_model(args)
    
    if args.loss == 'mse':
        criterion = nn.MSELoss()
    else:
        criterion = nn.MAELoss()
    if args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate) 
    
    EXP.train(args, model, criterion, optimizer, train_loader, val_loader, now)
    EXP.test(args, criterion, optimizer, test_loader, now)

if __name__ == "__main__":
    main()
    
    
    
    
    