# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 20:04:01 2025

@author: Younghwi Kim
"""
import torch
from torch import nn
import SToT.modules.SToT as SToT
import SToT.modules.tools as tools
import src.Metric_node as metrics
import numpy as np

import warnings
warnings.filterwarnings("ignore")

def build_model(args):
    
    model = SToT.Model(args).float().to(args.device)
    return model

def train(args, model, criterion, optimizer, train_loader, val_loader, now):
    
    scheduler = tools.LargeScheduler(args, optimizer)
    device = args.device
    best = [10**5, 10**5, -1 * 10**5, 10**5, 10**5, 10**5, 10**5]
    relu = nn.ReLU()
    
    for epoch in range(args.num_epochs):
        
        model.train()
        for i, (batch_x, batch_x_mark, batch_y, batch_y_mark) in enumerate(train_loader):
            
            batch_x, batch_y = batch_x[:, :, :, args.emission].type(torch.float).to(device), batch_y[:, :, :, args.emission].type(torch.float).to(device)
            batch_x_mark, batch_y_mark = batch_x_mark.float().to(device), batch_y_mark.float().to(device)
            
            optimizer.zero_grad()
            output = model(batch_x, batch_x_mark, None, None)
            
            loss = criterion(output, batch_y[:, -args.pred_len:])
            loss.backward()
            optimizer.step()

        outputs, actuals = [], []
        
        model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_x_mark, batch_y, batch_y_mark) in enumerate(val_loader):
                
                batch_x, batch_y = batch_x[:, :, :, args.emission].type(torch.float).to(device), batch_y[:, :, :, args.emission].type(torch.float).to(device)
                batch_x_mark, batch_y_mark = batch_x_mark.float().to(device), batch_y_mark.float().to(device)

                output = model(batch_x, batch_x_mark, None, None)
                
                outputs.append(relu(output))
                actuals.append(batch_y[:, -args.pred_len:])
                    
            outputs = torch.cat(outputs, dim=0).detach().cpu().numpy()
            actuals = torch.cat(actuals, dim=0).detach().cpu().numpy()
                
            metric = metrics.metric(outputs, actuals)
            best = metrics.update(now, args.save, model, best, metric, args.emission, args.model_name, args.pred_len, epoch)
        
        if epoch%args.log_interval == 0:
            metrics.plot(outputs, actuals, args.model_name, args.pred_len, args.emission, now)
            print(f'Epoch: {epoch} / Train loss: {loss:.5f}')
            print(f'Val Metric - [RMSE: {np.sqrt(best[0]):.9f} / MAE: {best[1]:.8f} / COR: {best[2]:.4f} / RSE: {best[3]:.9f}]')
        # scheduler.schedule_step(epoch)
    
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    

def test(args, criterion, optimizer, test_loader, now):
    
    device = args.device
    model_path = f'results/{args.model_name}/{args.emission}/{args.model_name}_{args.pred_len}_{now.month}{now.day}{now.hour}{now.minute}'
    model = SToT.Model(args).float().to(args.device)
    model.load_state_dict(torch.load(model_path + f'/{args.model_name}_{args.pred_len}.pth'))

    outputs, actuals = [], []
    relu = nn.ReLU()
    
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_x_mark, batch_y, batch_y_mark) in enumerate(test_loader):
            
            batch_x, batch_y = batch_x[:, :, :, args.emission].type(torch.float).to(device), batch_y[:, :, :, args.emission].type(torch.float).to(device)
            batch_x_mark, batch_y_mark = batch_x_mark.float().to(device), batch_y_mark.float().to(device)

            output = model(batch_x, batch_x_mark, None, None)
            
            outputs.append(relu(output))
            actuals.append(batch_y)
                
        outputs = torch.cat(outputs, dim=0).detach().cpu().numpy()
        actuals = torch.cat(actuals, dim=0).detach().cpu().numpy()
            
        metric = metrics.metric(outputs, actuals)
    
        metrics.plot(outputs, actuals, args.model_name, args.pred_len, args.emission, now, 'test')
        print(f'Test Metric - [RMSE: {np.sqrt(metric[0]):.9f} / MAE: {metric[1]:.8f} / COR: {metric[2]:.4f} / RSE: {metric[3]:.9f}]')
    # scheduler.schedule_step(epoch)
    
    torch.cuda.empty_cache()
    import gc
    gc.collect()
