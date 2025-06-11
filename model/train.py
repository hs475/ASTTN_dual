import time
import datetime
import torch
import math
from utils import log_string,metric,metric_by_timestep
from utils import load_data, load_data_with_wind
from torch.utils.data import DataLoader, TensorDataset
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 数据加载器函数
def data_loader(X, TE=None, Y=None, batch_size=32, shuffle=True, drop_last=False):
    """
    创建数据加载器，用于批量处理数据
    
    参数:
    - X: 特征数据 [B, T, N]
    - TE: 时间嵌入 (可选)
    - Y: 标签数据 (可选)
    - batch_size: 批次大小
    - shuffle: 是否打乱数据
    - drop_last: 是否丢弃最后不完整的批次
    
    返回:
    - DataLoader对象，或X的迭代器(如果只有X)
    """
    if Y is not None and TE is None:
        # 只有X和Y
        data = TensorDataset(X, Y)
        loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    elif Y is not None and TE is not None:
        # X、TE和Y都有
        # 注意：由于TE可能对所有样本相同，我们在加载器中不包含它，而是单独处理
        data = TensorDataset(X, Y)
        loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    else:
        # 只有X，没有标签
        if isinstance(X, torch.Tensor):
            data = TensorDataset(X)
            loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
        else:
            # 如果X不是张量，则直接返回X
            loader = X
    
    return loader

# 计算性能指标的函数
def calc_metrics(pred, target):
    """计算MAE, RMSE, MAPE指标"""
    mae = torch.mean(torch.abs(pred - target)).item()
    rmse = torch.sqrt(torch.mean((pred - target) ** 2)).item()
    mape = torch.mean(torch.abs((pred - target) / (target + 1e-5))).item()
    return mae, rmse, mape

# 修改GPU显存占用的设置
if torch.cuda.is_available():
    # 设置cudnn为非确定性模式，提高性能
    torch.backends.cudnn.deterministic = False
    # 启用cudnn基准测试模式，提高性能
    torch.backends.cudnn.benchmark = True
    
    # 报告当前GPU显存使用情况
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024 / 1024 / 1024
        reserved = torch.cuda.memory_reserved(i) / 1024 / 1024 / 1024
        print(f"训练开始时 GPU {i}: 已分配 {allocated:.2f} GB, 已预留 {reserved:.2f} GB")

import numpy as np
import os
import csv
import matplotlib
matplotlib.use('Agg')  # 设置非交互式后端，适合服务器环境
import matplotlib.pyplot as plt

def train(model, args, log, loss_criterion, optimizer, scheduler):
    # 根据参数决定是否使用风速数据
    if args.use_wind and args.model == "dual":
        log_string(log, '**** 加载交通数据和风速数据 ****')
        (trainX, trainTE, trainY, valX, valTE, valY, testX, testTE,
         testY,  traffic_mean, traffic_std,
         wind_trainX, wind_trainY, wind_valX, wind_valY, wind_testX, wind_testY,
         wind_mean, wind_std) = load_data_with_wind(args)
        mean, std = traffic_mean, traffic_std
    else:
        log_string(log, '**** 仅加载交通数据 ****')
        (trainX, trainTE, trainY, valX, valTE, valY, testX, testTE,
         testY,  mean, std) = load_data(args)
        wind_trainX, wind_valX, wind_testX = None, None, None
    
    num_train, _, _ = trainX.shape
    log_string(log, '**** training model ****')
    num_val = valX.shape[0]
    num_test = testX.shape[0]
    train_num_batch = math.ceil(num_train / args.batch_size)
    val_num_batch = math.ceil(num_val / args.batch_size)
    test_num_batch = math.ceil(num_test / args.batch_size)
    model = model.to(device)

    val_loss_min = float('inf')
    train_total_loss = []
    val_total_loss = []
    test_metrics_history = {'mae': [], 'rmse': [], 'mape': []}
    
    # 初始化早停参数
    early_stop_counter = 0
    early_stop_patience = args.early_stop if hasattr(args, 'early_stop') else 0
    best_model_state = None

    # 检测是否使用了多GPU训练
    is_parallel = isinstance(model, torch.nn.DataParallel)
    
    # Train & validation
    for epoch in range(args.max_epoch):
        permutation = torch.randperm(num_train)
        trainX = trainX[permutation]
        trainTE = trainTE[permutation]
        trainY = trainY[permutation]
        if args.use_wind and args.model == "dual":
            wind_trainX = wind_trainX[permutation]
        
        # train
        start_train = time.time()
        model.train()
        train_loss = 0
        for batch_idx in range(train_num_batch):
            start_idx = batch_idx * args.batch_size
            end_idx = min(num_train, (batch_idx + 1) * args.batch_size)
            X = trainX[start_idx: end_idx]
            TE = trainTE[start_idx: end_idx]
            label = trainY[start_idx: end_idx]
            X, TE = X.to(device), TE.to(device)
            label = label.to(device)
            
            # 处理风速数据输入
            if args.use_wind and args.model == "dual":
                wind_X = wind_trainX[start_idx: end_idx].to(device)
                optimizer.zero_grad()
                pred = model(X, TE, wind_X)
            else:
                optimizer.zero_grad()
                pred = model(X, TE)
            
            pred = pred * std + mean
            loss_batch = loss_criterion(pred, label)
            train_loss += float(loss_batch) * (end_idx - start_idx)
            loss_batch.backward()
            
            # 添加梯度裁剪
            if hasattr(args, 'grad_clip') and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            
            optimizer.step()
            print(f'训练批次: {batch_idx+1}/{train_num_batch}, 轮次:{epoch+1}批次损失:{loss_batch:.4f}')
            
            # 释放显存
            if args.use_wind and args.model == "dual":
                del X, TE, label, pred, loss_batch, wind_X
            else:
                del X, TE, label, pred, loss_batch
                
        train_loss /= num_train
        train_total_loss.append(train_loss)
        end_train = time.time()

        # val loss
        start_val = time.time()
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for batch_idx in range(val_num_batch):
                start_idx = batch_idx * args.batch_size
                end_idx = min(num_val, (batch_idx + 1) * args.batch_size)
                X = valX[start_idx: end_idx]
                TE = valTE[start_idx: end_idx]
                label = valY[start_idx: end_idx]
                X, TE = X.to(device), TE.to(device)
                label = label.to(device)
                
                # 处理风速数据输入
                if args.use_wind and args.model == "dual":
                    wind_X = wind_valX[start_idx: end_idx].to(device)
                    pred = model(X, TE, wind_X)
                else:
                    pred = model(X, TE)
                
                pred = pred * std + mean
                loss_batch = loss_criterion(pred, label)
                val_loss += loss_batch * (end_idx - start_idx)
        val_loss /= num_val
        val_total_loss.append(val_loss)
        end_val = time.time()
        
        #test metrics
        model.eval()
        testPred = []
        with torch.no_grad():
            for batch_idx in range(test_num_batch):
                start_idx = batch_idx * args.batch_size
                end_idx = min(num_test, (batch_idx + 1) * args.batch_size)
                X = testX[start_idx: end_idx]
                TE = testTE[start_idx: end_idx]
                X, TE = X.to(device), TE.to(device)
                
                # 处理风速数据输入
                if args.use_wind and args.model == "dual":
                    wind_X = wind_testX[start_idx: end_idx].to(device)
                    pred_batch = model(X, TE, wind_X)
                else:
                    pred_batch = model(X, TE)
                
                testPred.append(pred_batch.cpu().detach().clone())
        testPred = torch.from_numpy(np.concatenate(testPred, axis=0))
        testPred = testPred* std + mean
        test_mae, test_rmse, test_mape = metric(testPred, testY)
        test_metrics_history['mae'].append(test_mae)
        test_metrics_history['rmse'].append(test_rmse)
        test_metrics_history['mape'].append(test_mape)
        
        # 计算每个时间步的指标
        mae_by_step, rmse_by_step, mape_by_step = metric_by_timestep(testPred, testY)
        
        # 输出每个时间步的MAE
        log_string(log, f"\n每个时间步的MAE:")
        for t, mae_t in enumerate(mae_by_step):
            minutes = (t+1) * 5  # 假设每个时间步为5分钟
            log_string(log, f"时间步 {t+1} ({minutes}分钟): MAE = {mae_t:.4f}")
            
        # 输出详细的训练信息
        log_string(
            log,
            '%s | 轮次: %04d/%d, 训练时间: %.1fs, 推理时间: %.1fs' %
            (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch + 1,
             args.max_epoch, end_train - start_train, end_val - start_val))
        log_string(
            log, f'训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}')
        log_string(
            log, f'测试指标 - MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, MAPE: {test_mape*100:.2f}%')
        
        # 打印当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        log_string(log, f'当前学习率: {current_lr:.6f}')
        
        if val_loss <= val_loss_min:
            log_string(
                log,
                f'验证损失从 {val_loss_min:.4f} 降低到 {val_loss:.4f}, 保存模型到 {args.model_file}')
            val_loss_min = val_loss
            early_stop_counter = 0  # 重置早停计数器
            
            # 保存模型时，如果是DataParallel，需要保存原始模型
            if is_parallel:
                torch.save(model.module, args.model_file)
                # 保存最佳模型状态
                best_model_state = {k: v.clone() for k, v in model.module.state_dict().items()}
            else:
                torch.save(model, args.model_file)
                # 保存最佳模型状态
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            # 验证损失没有改善，增加早停计数器
            early_stop_counter += 1
            log_string(log, f'验证损失未改善，早停计数器: {early_stop_counter}/{early_stop_patience}')
            
            # 如果启用早停且计数器达到耐心值，则停止训练
            if early_stop_patience > 0 and early_stop_counter >= early_stop_patience:
                log_string(log, f'早停触发! 最佳验证损失: {val_loss_min:.4f}')
                # 恢复最佳模型
                if best_model_state is not None:
                    if is_parallel:
                        model.module.load_state_dict(best_model_state)
                    else:
                        model.load_state_dict(best_model_state)
                break
                
        scheduler.step()

    log_string(log, f'训练和验证已完成，模型已保存为 {args.model_file}')
    
    # 输出训练结束后的最佳性能
    best_epoch = test_metrics_history['mae'].index(min(test_metrics_history['mae'])) + 1
    log_string(log, f'训练结束 - 最佳性能 (轮次 {best_epoch}):')
    log_string(log, f'MAE: {min(test_metrics_history["mae"]):.4f}')
    log_string(log, f'RMSE: {test_metrics_history["rmse"][test_metrics_history["mae"].index(min(test_metrics_history["mae"]))]:.4f}')
    log_string(log, f'MAPE: {test_metrics_history["mape"][test_metrics_history["mae"].index(min(test_metrics_history["mae"]))]*100:.2f}%')
    
    # 确保所有返回值都是CPU张量
    if torch.is_tensor(train_total_loss):
        train_total_loss = [t.cpu().detach().numpy() if torch.is_tensor(t) else t for t in train_total_loss]
    if torch.is_tensor(val_total_loss):
        val_total_loss = [t.cpu().detach().numpy() if torch.is_tensor(t) else t for t in val_total_loss]
    
    # 确保所有指标都是CPU张量
    for key in test_metrics_history:
        if torch.is_tensor(test_metrics_history[key]):
            test_metrics_history[key] = [t.cpu().detach().numpy() if torch.is_tensor(t) else t for t in test_metrics_history[key]]
    
    return train_total_loss, val_total_loss, test_metrics_history
