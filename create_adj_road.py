import numpy as np
import pandas as pd
import pickle
import os
import h5py
import scipy.sparse as sp

def calculate_adjacency_matrix(data, method="correlation", threshold=0.5):
    """
    基于数据计算邻接矩阵
    
    参数:
    - data: 数据矩阵 (时间步, 节点数)
    - method: 计算方法，"correlation"表示相关性，"distance"表示欧氏距离
    - threshold: 阈值，相关性超过此值或距离低于此值的节点被认为是相连的
    
    返回:
    - adj_mx: 邻接矩阵
    """
    num_nodes = data.shape[1]
    adj_mx = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    
    if method == "correlation":
        # 计算节点之间的相关系数
        correlation_matrix = np.corrcoef(data.T)
        adj_mx = np.abs(correlation_matrix)
        # 设置阈值，保留高相关性的连接
        adj_mx[adj_mx < threshold] = 0
    
    elif method == "distance":
        # 计算节点之间的欧氏距离
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    # 计算节点i和j之间的欧氏距离
                    dist = np.sqrt(np.sum((data[:, i] - data[:, j])**2))
                    # 距离越远，连接权重越小
                    adj_mx[i, j] = 1.0 / (1.0 + dist)
        
        # 设置阈值，保留近距离的连接
        adj_mx[adj_mx < threshold] = 0
    
    # 确保对角线为1（自连接）
    np.fill_diagonal(adj_mx, 1)
    return adj_mx

def save_pickle(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def load_h5_data(filename):
    """加载h5文件数据"""
    with h5py.File(filename, 'r') as hf:
        if 'data' in hf:
            data = np.array(hf['data'])
        else:
            # 如果没有'data'键，则尝试读取DataFrame格式的h5文件
            data = pd.read_hdf(filename).values
    return data

def generate_adj_matrix_file(ds_name, method="correlation", threshold=0.5):
    """为指定数据集生成邻接矩阵文件"""
    # 数据集文件路径
    data_file = f'./data/{ds_name}.h5'
    adj_file = f'./data/adj_{ds_name}.pkl'
    
    print(f"正在为数据集 {ds_name} 生成邻接矩阵...")
    
    # 加载数据
    try:
        data = load_h5_data(data_file)
        print(f"数据形状: {data.shape}")
    except Exception as e:
        print(f"加载数据文件 {data_file} 时出错: {e}")
        return False
    
    # 计算邻接矩阵
    try:
        adj_mx = calculate_adjacency_matrix(data, method=method, threshold=threshold)
        print(f"邻接矩阵形状: {adj_mx.shape}")
        
        # 计算邻接矩阵的稀疏度
        sparsity = 1.0 - (np.count_nonzero(adj_mx) / (adj_mx.shape[0] * adj_mx.shape[1]))
        print(f"邻接矩阵稀疏度: {sparsity:.4f} (值为0的元素比例)")
        
        # 将邻接矩阵转换为稀疏矩阵格式
        sp_mx = sp.coo_matrix(adj_mx, dtype=np.float32)
        
        # 创建符合代码要求的格式
        # 这里我们根据load_graph函数的调用格式 _,_, adj_mx = load_pickle(dsadj_file)
        # 创建包含三个元素的元组，前两个为None
        adj_data = (None, None, adj_mx)
        
        # 保存为pickle文件
        save_pickle(adj_data, adj_file)
        print(f"邻接矩阵已保存到 {adj_file}")
        return True
    except Exception as e:
        print(f"生成邻接矩阵时出错: {e}")
        return False

if __name__ == "__main__":
    # 生成road数据集的邻接矩阵
    generate_adj_matrix_file("road", method="correlation", threshold=0.5)
    
    # 检查生成的文件是否存在
    if os.path.exists('./data/adj_road.pkl'):
        print("成功生成road数据集的邻接矩阵文件!")
    else:
        print("生成邻接矩阵文件失败，请检查错误信息。") 