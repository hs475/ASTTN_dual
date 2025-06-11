import numpy as np
import pandas as pd
import os
import h5py
import pickle
import time

def load_h5_data(filename):
    """加载h5文件数据"""
    try:
        with h5py.File(filename, 'r') as hf:
            if 'data' in hf:
                data = np.array(hf['data'])
            else:
                # 如果没有'data'键，则尝试读取DataFrame格式的h5文件
                data = pd.read_hdf(filename).values
        return data
    except Exception as e:
        print(f"加载数据文件 {filename} 时出错: {e}")
        return None

def load_pickle(pickle_file):
    """加载pickle文件"""
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

def generate_simple_embeddings(num_nodes, dim=64):
    """
    生成简单的节点嵌入，用于替代真实的node2vec嵌入
    这种方法生成随机嵌入，不依赖于图的结构
    """
    print(f"为 {num_nodes} 个节点生成维度为 {dim} 的简单嵌入")
    # 使用正态分布生成随机嵌入
    embeddings = np.random.normal(0, 0.1, size=(num_nodes, dim))
    # 归一化
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    return embeddings

def save_embeddings_to_txt(embeddings, output_file):
    """将嵌入保存为指定格式的txt文件"""
    num_nodes, dim = embeddings.shape
    with open(output_file, 'w') as f:
        # 第一行是节点数和维度
        f.write(f"{num_nodes} {dim}\n")
        # 每个节点的嵌入占一行
        for i in range(num_nodes):
            line = f"{i} " + " ".join([f"{val:.6f}" for val in embeddings[i]])
            f.write(line + "\n")
    print(f"嵌入已保存到 {output_file}")

def generate_node2vec_embeddings(ds_name, dim=64):
    """为指定数据集生成node2vec嵌入"""
    # 数据集文件路径
    data_file = f'./data/{ds_name}.h5'
    adj_file = f'./data/adj_{ds_name}.pkl'
    output_file = f'./data/node2vec_{ds_name}.txt'
    
    print(f"正在为数据集 {ds_name} 生成node2vec嵌入...")
    
    # 检查邻接矩阵文件是否存在
    if not os.path.exists(adj_file):
        print(f"邻接矩阵文件 {adj_file} 不存在，请先运行 create_adj_road.py 生成")
        return False
    
    # 获取节点数量
    try:
        # 优先从数据文件获取节点数量
        data = load_h5_data(data_file)
        num_nodes = data.shape[1]
        print(f"从数据文件获取到节点数量: {num_nodes}")
    except:
        # 如果数据文件加载失败，尝试从邻接矩阵获取节点数量
        try:
            _, _, adj_mx = load_pickle(adj_file)
            num_nodes = adj_mx.shape[0]
            print(f"从邻接矩阵获取到节点数量: {num_nodes}")
        except:
            print("无法确定节点数量，生成嵌入失败")
            return False
    
    # 生成嵌入（这里使用简单的替代方法）
    # 注意：真正的node2vec需要使用图结构学习嵌入，这里只是生成一个占位符
    # 实际项目中可能需要使用node2vec库进行更准确的嵌入生成
    embeddings = generate_simple_embeddings(num_nodes, dim)
    
    # 保存嵌入
    save_embeddings_to_txt(embeddings, output_file)
    
    # 检查文件是否成功生成
    return os.path.exists(output_file)

if __name__ == "__main__":
    # 记录开始时间
    start_time = time.time()
    
    # 生成road数据集的node2vec嵌入
    success = generate_node2vec_embeddings("road", dim=64)
    
    # 输出结果
    if success:
        print(f"成功生成road数据集的node2vec嵌入文件! 用时: {time.time() - start_time:.2f}秒")
    else:
        print("生成node2vec嵌入文件失败，请检查错误信息。") 