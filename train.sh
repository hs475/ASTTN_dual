#!/bin/bash

# 检查并生成road数据集所需的文件
echo "检查road数据集是否有必要的邻接矩阵和嵌入文件..."

# 检查adj_road.pkl是否存在，不存在则生成
if [ ! -f "./data/adj_road.pkl" ]; then
    echo "正在生成road数据集的邻接矩阵文件..."
    python create_adj_road.py
else
    echo "road数据集的邻接矩阵文件已存在"
fi

# 检查node2vec_road.txt是否存在，不存在则生成
if [ ! -f "./data/node2vec_road.txt" ]; then
    echo "正在生成road数据集的node2vec嵌入文件..."
    python create_node2vec_road.py
else
    echo "road数据集的node2vec嵌入文件已存在"
fi

# 训练命令 - pems-bay数据集
# echo "训练 pems-bay 数据集..."
# CUDA_VISIBLE_DEVICES=0,1,2 python main.py --batch_size 500 --max_epoch 50 --use_multi_gpu

# CUDA_VISIBLE_DEVICES=5,6 python main.py --batch_size 300 --max_epoch 50 --use_multi_gpu 

# CUDA_VISIBLE_DEVICES=7 python main.py --batch_size 100 --max_epoch 50 --use_multi_gpu --ds pems-bay

# 训练命令 - road数据集
# echo "训练 road 数据集..."
# CUDA_VISIBLE_DEVICES=0 python main.py --batch_size 100 --max_epoch 100 --use_multi_gpu --ds road

# 训练命令 - 使用交通+风速数据预测交通情况
echo "训练 road+风速 数据集 (dual模型)..."
CUDA_VISIBLE_DEVICES=1 python main.py --batch_size 32 --max_epoch 100 --L 1 --K 10 --d 12 --window 6 --learning_rate 0.0005 --weight_decay 1e-4 --dropout 0.4 --scheduler cosine --early_stop 15 --use_multi_gpu --ds road --model dual --use_wind --wind_file ./data/data_wind_speed_all.h5 --remark _optimized

# 训练命令 - 单模型(adp模型)
echo "训练 road 数据集 (adp单模型)..."
CUDA_VISIBLE_DEVICES=0 python main.py --batch_size 32 --max_epoch 100 --L 1 --K 10 --d 12 --window 6 --learning_rate 0.0005 --weight_decay 1e-4 --dropout 0.4 --scheduler cosine --early_stop 15 --use_multi_gpu --ds road --model adp --remark _single_model

# 训练命令 - 简单baseline (简化参数的单模型)
echo "训练 road 数据集 (简单baseline模型)..."
CUDA_VISIBLE_DEVICES=2 python main.py --batch_size 32 --max_epoch 50 --L 1 --K 5 --d 8 --window 3 --learning_rate 0.001 --weight_decay 1e-3 --dropout 0.2 --scheduler step --ds road --model adp --remark _simple_baseline

# 训练命令 - 极简baseline (非常简化的模型)
echo "训练 road 数据集 (极简baseline模型)..."
CUDA_VISIBLE_DEVICES=2 python main.py --batch_size 64 --max_epoch 30 --L 1 --K 3 --d 4 --window 1 --learning_rate 0.005 --weight_decay 1e-2 --dropout 0.1 --scheduler step --ds road --model adp --remark _ultra_simple_baseline


