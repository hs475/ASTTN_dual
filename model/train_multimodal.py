import torch
import time
from utils import log_string, metric, load_data

def train_multimodal(model, args, log, loss_criterion, optimizer, scheduler, 
                    trainWind, valWind, testWind):
    """
    训练多模态模型，整合交通数据和风速数据
    
    参数:
    - model: 多模态模型
    - args: 参数对象
    - log: 日志对象
    - loss_criterion: 损失函数
    - optimizer: 优化器
    - scheduler: 学习率调度器
    - trainWind: 训练集风速数据
    - valWind: 验证集风速数据
    - testWind: 测试集风速数据
    
    返回:
    - train_losses: 训练损失列表
    - val_losses: 验证损失列表
    - metrics_history: 评估指标历史
    """
    device = args.device if hasattr(args, 'device') else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # 重新加载数据，因为原始数据已被删除
    log_string(log, "重新加载交通数据用于训练...")
    (trainX, trainTE, trainY, valX, valTE, valY, testX, testTE,
     testY, mean, std) = load_data(args)
    
    # 将数据移动到设备上
    trainX = trainX.to(device)
    trainTE = trainTE.to(device)
    trainY = trainY.to(device)
    valX = valX.to(device)
    valTE = valTE.to(device)
    valY = valY.to(device)
    testX = testX.to(device)
    testTE = testTE.to(device)
    testY = testY.to(device)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    metrics_history = {'mae': [], 'rmse': [], 'mape': []}
    
    log_string(log, "开始训练多模态模型...")
    log_string(log, f"训练样本数: {trainX.shape[0]}, 验证样本数: {valX.shape[0]}, 测试样本数: {testX.shape[0]}")
    log_string(log, f"风速数据形状 - 训练: {trainWind.shape}, 验证: {valWind.shape}, 测试: {testWind.shape}")
    
    # 记录开始时间
    train_start_time = time.time()
    
    for epoch in range(args.max_epoch):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0
        train_mae = 0
        train_rmse = 0
        train_mape = 0
        batches = 0
        
        # 训练阶段
        for i in range(0, trainX.shape[0], args.batch_size):
            optimizer.zero_grad()
            end_idx = min(i + args.batch_size, trainX.shape[0])
            x_batch = trainX[i:end_idx]
            te_batch = trainTE[i:end_idx]
            y_batch = trainY[i:end_idx]
            wind_batch = trainWind[i:end_idx]
            
            y_pred = model(x_batch, wind_batch, te_batch)
            loss = loss_criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            batch_mae, batch_rmse, batch_mape = metric(y_pred, y_batch)
            train_mae += batch_mae
            train_rmse += batch_rmse
            train_mape += batch_mape
            batches += 1
        
        # 计算训练指标平均值
        train_loss /= batches
        train_mae /= batches
        train_rmse /= batches
        train_mape /= batches
        train_losses.append(train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_mae = 0
        val_rmse = 0
        val_mape = 0
        batches = 0
        
        with torch.no_grad():
            for i in range(0, valX.shape[0], args.batch_size):
                end_idx = min(i + args.batch_size, valX.shape[0])
                x_batch = valX[i:end_idx]
                te_batch = valTE[i:end_idx]
                y_batch = valY[i:end_idx]
                wind_batch = valWind[i:end_idx]
                
                y_pred = model(x_batch, wind_batch, te_batch)
                loss = loss_criterion(y_pred, y_batch)
                
                val_loss += loss.item()
                batch_mae, batch_rmse, batch_mape = metric(y_pred, y_batch)
                val_mae += batch_mae
                val_rmse += batch_rmse
                val_mape += batch_mape
                batches += 1
        
        # 计算验证指标平均值
        val_loss /= batches
        val_mae /= batches
        val_rmse /= batches
        val_mape /= batches
        val_losses.append(val_loss)
        
        # 记录指标
        metrics_history['mae'].append(val_mae)
        metrics_history['rmse'].append(val_rmse)
        metrics_history['mape'].append(val_mape)
        
        # 计算本轮耗时
        epoch_time = time.time() - epoch_start_time
        
        # 输出日志
        log_string(log, f'Epoch: {epoch+1:03d}/{args.max_epoch}, 耗时: {epoch_time:.2f}秒')
        log_string(log, f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        log_string(log, f'Train: MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}, MAPE: {train_mape*100:.4f}%')
        log_string(log, f'Val: MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}, MAPE: {val_mape*100:.4f}%')
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.model_file)
            log_string(log, f'验证损失从降低到 {val_loss:.4f}, 保存模型到 {args.model_file}')
        
        # 更新学习率
        scheduler.step()
    
    # 训练完成，计算总耗时
    total_train_time = time.time() - train_start_time
    log_string(log, f'训练完成，总耗时: {total_train_time/3600:.2f}小时')
    
    # 加载最佳模型并进行测试
    log_string(log, "加载最佳模型进行测试...")
    model.load_state_dict(torch.load(args.model_file))
    model.eval()
    
    test_loss = 0
    test_mae = 0
    test_rmse = 0
    test_mape = 0
    batches = 0
    
    with torch.no_grad():
        for i in range(0, testX.shape[0], args.batch_size):
            end_idx = min(i + args.batch_size, testX.shape[0])
            x_batch = testX[i:end_idx]
            te_batch = testTE[i:end_idx]
            y_batch = testY[i:end_idx]
            wind_batch = testWind[i:end_idx]
            
            y_pred = model(x_batch, wind_batch, te_batch)
            loss = loss_criterion(y_pred, y_batch)
            
            test_loss += loss.item()
            batch_mae, batch_rmse, batch_mape = metric(y_pred, y_batch)
            test_mae += batch_mae
            test_rmse += batch_rmse
            test_mape += batch_mape
            batches += 1
    
    # 计算测试指标平均值
    test_loss /= batches
    test_mae /= batches
    test_rmse /= batches
    test_mape /= batches
    
    # 输出测试结果
    log_string(log, f'测试结果:')
    log_string(log, f'Loss: {test_loss:.4f}, MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, MAPE: {test_mape*100:.4f}%')
    
    # 确保所有值都是Python原生类型，而非张量
    train_losses = [float(loss) for loss in train_losses]
    val_losses = [float(loss) for loss in val_losses]
    
    for key in metrics_history:
        metrics_history[key] = [float(val) for val in metrics_history[key]]
    
    return train_losses, val_losses, metrics_history 