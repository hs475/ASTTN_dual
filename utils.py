import pandas as pd
import torch
import pickle
import dgl
import numpy as np
from scipy import sparse as sp
import hashlib
# log string
def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)

def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))

def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    
    loss = torch.abs(preds-labels)/labels
    
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    return torch.mean(loss)

def metric(pred, real):
    mae = masked_mae(pred,real,0.0).item()
    mape = masked_mape(pred,real,0.0).item()
    rmse = masked_rmse(pred,real,0.0).item()
    return mae,rmse,mape

def metric_by_timestep(pred, real):
    """
    计算每个时间步的指标
    
    参数:
    - pred: 预测值 [batch_size, num_timesteps, num_nodes]
    - real: 真实值 [batch_size, num_timesteps, num_nodes]
    
    返回:
    - mae_by_step: 每个时间步的MAE列表
    - rmse_by_step: 每个时间步的RMSE列表
    - mape_by_step: 每个时间步的MAPE列表
    """
    num_timesteps = pred.shape[1]
    mae_by_step = []
    rmse_by_step = []
    mape_by_step = []
    
    for t in range(num_timesteps):
        # 提取当前时间步的预测和真实值
        pred_t = pred[:, t, :]
        real_t = real[:, t, :]
        
        # 计算当前时间步的指标
        mae_t = masked_mae(pred_t, real_t, 0.0).item()
        rmse_t = masked_rmse(pred_t, real_t, 0.0).item()
        mape_t = masked_mape(pred_t, real_t, 0.0).item()
        
        mae_by_step.append(mae_t)
        rmse_by_step.append(rmse_t)
        mape_by_step.append(mape_t)
    
    return mae_by_step, rmse_by_step, mape_by_step

def seq2instance(data, num_his, num_pred):
    num_step, dims = data.shape
    num_sample = num_step - num_his - num_pred + 1
    x = torch.zeros(num_sample, num_his, dims)
    y = torch.zeros(num_sample, num_pred, dims)
    for i in range(num_sample):
        x[i] = data[i: i + num_his]
        y[i] = data[i + num_his: i + num_his + num_pred]
    return x, y

def load_data(args):
    # Traffic
    df = pd.read_hdf(args.traffic_file)
    traffic = torch.from_numpy(df.values)
    # train/val/test
    num_step = df.shape[0]
    train_steps = round(args.train_ratio * num_step)
    test_steps = round(args.test_ratio * num_step)
    val_steps = num_step - train_steps - test_steps
    train = traffic[: train_steps]
    val = traffic[train_steps: train_steps + val_steps]
    test = traffic[-test_steps:]
    # X, Y
    trainX, trainY = seq2instance(train, args.num_his, args.num_pred)
    valX, valY = seq2instance(val, args.num_his, args.num_pred)
    testX, testY = seq2instance(test, args.num_his, args.num_pred)
    # normalization
    mean, std = torch.mean(trainX), torch.std(trainX)
    trainX = (trainX - mean) / std
    valX = (valX - mean) / std
    testX = (testX - mean) / std

    # temporal embedding
    time = pd.DatetimeIndex(df.index)
    dayofweek = torch.reshape(torch.tensor(time.weekday), (-1, 1))
    
    # 修复时间间隔计算，假设时间间隔为5分钟（300秒）
    seconds_per_day = 24 * 60 * 60
    seconds_in_time = time.hour * 3600 + time.minute * 60 + time.second
    timeofday = seconds_in_time // 300  # 使用300秒（5分钟）作为时间间隔
    
    timeofday = torch.reshape(torch.tensor(timeofday), (-1, 1))
    time = torch.cat((dayofweek, timeofday), -1)
    # train/val/test
    train = time[: train_steps]
    val = time[train_steps: train_steps + val_steps]
    test = time[-test_steps:]
    # shape = (num_sample, num_his + num_pred, 2)
    trainTE = seq2instance(train, args.num_his, args.num_pred)
    trainTE = torch.cat(trainTE, 1).type(torch.int32)
    valTE = seq2instance(val, args.num_his, args.num_pred)
    valTE = torch.cat(valTE, 1).type(torch.int32)
    testTE = seq2instance(test, args.num_his, args.num_pred)
    testTE = torch.cat(testTE, 1).type(torch.int32)

    return (trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY,
             mean, std)

def load_data_with_wind(args):
    """加载交通数据和风速数据，同时返回两个数据集的处理结果
    
    参数:
    - args: 参数对象，需要包含traffic_file和wind_file字段
    
    返回:
    - 交通数据：trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY, traffic_mean, traffic_std
    - 风速数据：wind_trainX, wind_trainY, wind_valX, wind_valY, wind_testX, wind_testY, wind_mean, wind_std
    """
    # 加载交通数据
    traffic_df = pd.read_hdf(args.traffic_file)
    traffic = torch.from_numpy(traffic_df.values)
    
    # 加载风速数据
    wind_df = pd.read_hdf(args.wind_file)
    wind = torch.from_numpy(wind_df.values)
    
    # 确保两个数据集有相同的时间点数量
    assert traffic.shape[0] == wind.shape[0], "交通数据和风速数据的时间点数量不一致"
    
    # 划分训练/验证/测试集
    num_step = traffic_df.shape[0]
    train_steps = round(args.train_ratio * num_step)
    test_steps = round(args.test_ratio * num_step)
    val_steps = num_step - train_steps - test_steps
    
    # 划分交通数据
    traffic_train = traffic[: train_steps]
    traffic_val = traffic[train_steps: train_steps + val_steps]
    traffic_test = traffic[-test_steps:]
    
    # 划分风速数据
    wind_train = wind[: train_steps]
    wind_val = wind[train_steps: train_steps + val_steps]
    wind_test = wind[-test_steps:]
    
    # 生成交通数据的输入序列和标签
    traffic_trainX, traffic_trainY = seq2instance(traffic_train, args.num_his, args.num_pred)
    traffic_valX, traffic_valY = seq2instance(traffic_val, args.num_his, args.num_pred)
    traffic_testX, traffic_testY = seq2instance(traffic_test, args.num_his, args.num_pred)
    
    # 生成风速数据的输入序列和标签
    wind_trainX, wind_trainY = seq2instance(wind_train, args.num_his, args.num_pred)
    wind_valX, wind_valY = seq2instance(wind_val, args.num_his, args.num_pred)
    wind_testX, wind_testY = seq2instance(wind_test, args.num_his, args.num_pred)
    
    # 交通数据标准化
    traffic_mean, traffic_std = torch.mean(traffic_trainX), torch.std(traffic_trainX)
    traffic_trainX = (traffic_trainX - traffic_mean) / traffic_std
    traffic_valX = (traffic_valX - traffic_mean) / traffic_std
    traffic_testX = (traffic_testX - traffic_mean) / traffic_std
    
    # 风速数据标准化
    wind_mean, wind_std = torch.mean(wind_trainX), torch.std(wind_trainX)
    wind_trainX = (wind_trainX - wind_mean) / wind_std
    wind_valX = (wind_valX - wind_mean) / wind_std
    wind_testX = (wind_testX - wind_mean) / wind_std
    
    # 生成时间嵌入
    time = pd.DatetimeIndex(traffic_df.index)
    dayofweek = torch.reshape(torch.tensor(time.weekday), (-1, 1))
    
    # 修复时间间隔计算，假设时间间隔为5分钟（300秒）
    seconds_per_day = 24 * 60 * 60
    seconds_in_time = time.hour * 3600 + time.minute * 60 + time.second
    timeofday = seconds_in_time // 300  # 使用300秒（5分钟）作为时间间隔
    
    timeofday = torch.reshape(torch.tensor(timeofday), (-1, 1))
    time = torch.cat((dayofweek, timeofday), -1)
    
    # 划分时间嵌入数据
    train_time = time[: train_steps]
    val_time = time[train_steps: train_steps + val_steps]
    test_time = time[-test_steps:]
    
    # 生成时间嵌入序列
    trainTE = seq2instance(train_time, args.num_his, args.num_pred)
    trainTE = torch.cat(trainTE, 1).type(torch.int32)
    valTE = seq2instance(val_time, args.num_his, args.num_pred)
    valTE = torch.cat(valTE, 1).type(torch.int32)
    testTE = seq2instance(test_time, args.num_his, args.num_pred)
    testTE = torch.cat(testTE, 1).type(torch.int32)
    
    return (traffic_trainX, trainTE, traffic_trainY, traffic_valX, valTE, traffic_valY, 
            traffic_testX, testTE, traffic_testY, traffic_mean, traffic_std,
            wind_trainX, wind_trainY, wind_valX, wind_valY, wind_testX, wind_testY, 
            wind_mean, wind_std)

def laplacian_positional_encoding(g, pos_enc_dim = 64):
    # Laplacian
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort() # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
    # g.ndata['lap_pos_enc'] = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float() 
    lap_pos_enc = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float() 
    return lap_pos_enc

def wl_positional_encoding(g):
    """
        WL-based absolute positional embedding 
        adapted from 
        
        "Graph-Bert: Only Attention is Needed for Learning Graph Representations"
        Zhang, Jiawei and Zhang, Haopeng and Xia, Congying and Sun, Li, 2020
        https://github.com/jwzhanggy/Graph-Bert
    """
    max_iter = 4
    node_color_dict = {}
    node_neighbor_dict = {}
    edge_list = torch.nonzero(g.adj().to_dense() != 0, as_tuple=False).numpy()
    node_list = g.nodes().numpy()
    # setting init
    for node in node_list:
        node_color_dict[node] = 1
        node_neighbor_dict[node] = {}
    for pair in edge_list:
        u1, u2 = pair
        if u1 not in node_neighbor_dict:
            node_neighbor_dict[u1] = {}
        if u2 not in node_neighbor_dict:
            node_neighbor_dict[u2] = {}
        node_neighbor_dict[u1][u2] = 1
        node_neighbor_dict[u2][u1] = 1
    # WL recursion
    iteration_count = 1
    exit_flag = False
    while not exit_flag:
        new_color_dict = {}
        for node in node_list:
            neighbors = node_neighbor_dict[node]
            neighbor_color_list = [node_color_dict[neb] for neb in neighbors]
            color_string_list = [str(node_color_dict[node])] + sorted([str(color) for color in neighbor_color_list])
            color_string = "_".join(color_string_list)
            hash_object = hashlib.md5(color_string.encode())
            hashing = hash_object.hexdigest()
            new_color_dict[node] = hashing
        color_index_dict = {k: v+1 for v, k in enumerate(sorted(set(new_color_dict.values())))}
        for node in new_color_dict:
            new_color_dict[node] = color_index_dict[new_color_dict[node]]
        if node_color_dict == new_color_dict or iteration_count == max_iter:
            exit_flag = True
        else:
            node_color_dict = new_color_dict
        iteration_count += 1
    wl_pos_enc = torch.LongTensor(list(node_color_dict.values())).unsqueeze(1)
    return wl_pos_enc 

def load_graph(args):
    dsadj_file = './data/' + 'adj_' + args.ds + '.pkl'
    _,_, adj_mx = load_pickle(dsadj_file)
    src, dst = np.nonzero(adj_mx)
    g = dgl.graph((src, dst))

    if args.se_type == 'node2vec':
        # spatial embedding
        with open("./data/node2vec_"+args.ds+".txt", mode='r') as f:
            lines = f.readlines()
            temp = lines[0].split(' ')
            num_vertex, dims = int(temp[0]), int(temp[1])
            SE = torch.zeros((num_vertex, dims), dtype=torch.float32)
            for line in lines[1:]:
                temp = line.split(' ')
                index = int(temp[0])
                SE[index] = torch.tensor([float(ch) for ch in temp[1:]])
    elif args.se_type == 'wl':
        SE = wl_positional_encoding(g)
    elif args.se_type == 'lap':
        SE = laplacian_positional_encoding(g)
    return  SE, g

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

# statistic model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


