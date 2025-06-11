import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn.functional import edge_softmax
from model.model_utils import *
from model.model_adp import ST_Layer, STAttention_Adp

class Model_Dual(nn.Module):
    """将交通数据和风速数据作为输入，但只预测交通情况的模型"""
    def __init__(self, SE, args, window_size = 3, T = 12, N=None):
        super(Model_Dual, self).__init__()
        L = args.L
        K = args.K
        d = args.d
        D = K * d
        
        # 添加dropout支持
        self.dropout_rate = args.dropout if hasattr(args, 'dropout') else 0.3

        self.num_his = args.num_his
        self.SE = SE
        emb_dim = SE.shape[1]
        self.STEmbedding = STEmbedding(D, emb_dim=emb_dim)

        # 共享的编码器层
        self.STAttBlock_1 = nn.ModuleList([ST_Layer(K, d, T=T, window_size=window_size, N=N) for _ in range(L)])
        self.transformAttention = TransformAttention(K, d)
        
        # 共享的解码器层
        self.STAttBlock_2 = nn.ModuleList([ST_Layer(K, d, T=T, window_size=window_size, N=N) for _ in range(L)])
        
        # 交通数据处理层
        self.traffic_mlp_1 = CONVs(input_dims=[1, D], units=[D, D], activations=[F.relu, None])
        
        # 风速数据处理层
        self.wind_mlp_1 = CONVs(input_dims=[1, D], units=[D, D], activations=[F.relu, None])
        
        # 融合层 - 将交通特征和风速特征融合
        self.fusion_layer = nn.Linear(D*2, D)
        
        # 添加dropout层
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # 最终输出层 - 只输出交通预测
        self.output_mlp = CONVs(input_dims=[D, D], units=[D, 1], activations=[F.relu, None])

    def forward(self, X, TE, wind_X):
        """
        前向传播函数
        
        参数:
        - X: 交通数据输入 [B, T, N]
        - TE: 时间嵌入 [B, 2*(T+T'), N]
        - wind_X: 风速数据输入 [B, T, N]
        
        返回:
        - out: 交通数据预测输出 [B, T', N]
        """
        # 确保SE与输入X在同一设备上
        SE = self.SE.to(X.device)
        
        # 交通数据处理
        X = torch.unsqueeze(X, -1)  # [B, T, N, 1]
        traffic_features = self.traffic_mlp_1(X)  # [B, T, N, D]
        traffic_features = self.dropout(traffic_features)
        
        # 风速数据处理
        wind_X = torch.unsqueeze(wind_X, -1)  # [B, T, N, 1]
        wind_features = self.wind_mlp_1(wind_X)  # [B, T, N, D]
        wind_features = self.dropout(wind_features)
        
        # 时空嵌入
        STE = self.STEmbedding(SE, TE)
        STE_his = STE[:, :self.num_his]
        STE_pred = STE[:, self.num_his:]
        
        # 单独编码交通特征
        encoded_traffic = traffic_features
        for net in self.STAttBlock_1:
            encoded_traffic = net(encoded_traffic, STE_his)
        encoded_traffic = self.transformAttention(encoded_traffic, STE_his, STE_pred)
        encoded_traffic = self.dropout(encoded_traffic)
        
        # 单独编码风速特征
        encoded_wind = wind_features
        for net in self.STAttBlock_1:
            encoded_wind = net(encoded_wind, STE_his)
        encoded_wind = self.transformAttention(encoded_wind, STE_his, STE_pred)
        encoded_wind = self.dropout(encoded_wind)
        
        # 特征融合: 拼接交通和风速特征
        B, T, N, D = encoded_traffic.shape
        encoded_traffic_flat = encoded_traffic.view(B, T, N, -1)
        encoded_wind_flat = encoded_wind.view(B, T, N, -1)
        
        # 拼接特征
        concat_features = torch.cat([encoded_traffic_flat, encoded_wind_flat], dim=3)  # [B, T, N, 2*D]
        
        # 融合特征
        fused_features = self.fusion_layer(concat_features)  # [B, T, N, D]
        fused_features = F.relu(fused_features)
        fused_features = self.dropout(fused_features)
        
        # 解码融合特征
        decoded_features = fused_features
        for net in self.STAttBlock_2:
            decoded_features = net(decoded_features, STE_pred)
        decoded_features = self.dropout(decoded_features)
        
        # 输出层 - 只预测交通
        out = self.output_mlp(decoded_features)  # [B, T', N, 1]
        
        return torch.squeeze(out, 3)  # [B, T', N] 