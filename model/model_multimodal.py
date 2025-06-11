import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn.functional import edge_softmax
from model.model_utils import *
from model.model_adp import ST_Layer

class MultiModalModel(nn.Module):
    def __init__(self, SE, args, window_size=3, T=12, N=None):
        super(MultiModalModel, self).__init__()
        L = args.L
        K = args.K
        d = args.d
        D = K * d
        
        self.num_his = args.num_his
        self.SE = SE
        emb_dim = SE.shape[1]
        self.STEmbedding = STEmbedding(D, emb_dim=emb_dim)
        
        # 交通数据处理
        self.traffic_mlp = CONVs(input_dims=[1, D], units=[D, D], activations=[F.relu, None])
        
        # 风速数据处理 - 使用较小的维度
        self.wind_dim = D // 2
        self.wind_mlp = CONVs(input_dims=[1, self.wind_dim], units=[self.wind_dim, self.wind_dim], activations=[F.relu, None])
        
        # 交通编码器
        self.traffic_encoder = nn.ModuleList([ST_Layer(K, d, T=T, window_size=window_size, N=N) for _ in range(L)])
        
        # 风速编码器 - 简化版
        self.wind_encoder = nn.ModuleList([
            SimplifiedST_Layer(K//2, d//2, T=T, window_size=window_size) 
            for _ in range(L-1)  # 使用较少的层
        ])
        
        # 高效融合层
        self.gated_fusion = GatedFusion(D, self.wind_dim)
        
        # 自适应特征权重
        self.feature_weighting = AdaptiveFeatureWeighting(D)
        
        # 剩余的网络层
        self.transformAttention = TransformAttention(K, d)
        self.decoder = nn.ModuleList([ST_Layer(K, d, T=T, window_size=window_size, N=N) for _ in range(L)])
        self.output_mlp = CONVs(input_dims=[D, D], units=[D, 1], activations=[F.relu, None])
        
        # 残差校准
        self.calibration = nn.Sequential(
            nn.Linear(D, D),
            nn.LayerNorm(D),
            nn.ReLU(),
            nn.Linear(D, D),
            nn.Sigmoid()
        )
    
    def forward(self, X_traffic, X_wind, TE):
        # 确保SE与输入在同一设备上
        SE = self.SE.to(X_traffic.device)
        
        # 处理交通数据
        X_traffic = torch.unsqueeze(X_traffic, -1)
        X_traffic = self.traffic_mlp(X_traffic)
        
        # 处理风速数据
        X_wind = torch.unsqueeze(X_wind, -1)
        X_wind = self.wind_mlp(X_wind)
        
        # 时空嵌入
        STE = self.STEmbedding(SE, TE)
        STE_his = STE[:, :self.num_his]
        STE_pred = STE[:, self.num_his:]
        
        # 保存原始交通特征用于残差连接
        X_traffic_orig = X_traffic
        
        # 交通数据编码
        for i, net in enumerate(self.traffic_encoder):
            X_traffic = net(X_traffic, STE_his)
            
            # 在中间层进行风速融合
            if i < len(self.wind_encoder):
                # 风速编码
                X_wind = self.wind_encoder[i](X_wind, STE_his)
                
                # 融合风速和交通特征
                X_traffic = self.gated_fusion(X_traffic, X_wind)
        
        # 应用自适应特征权重
        X_fused = self.feature_weighting(X_traffic)
        
        # 增加残差连接
        calibration_weights = self.calibration(X_fused)
        X_fused = X_fused + calibration_weights * X_traffic_orig
        
        # 转换注意力
        X_fused = self.transformAttention(X_fused, STE_his, STE_pred)
        
        # 解码
        for net in self.decoder:
            X_fused = net(X_fused, STE_pred)
        
        # 输出层
        X_out = self.output_mlp(X_fused)
        
        return torch.squeeze(X_out, 3)

class SimplifiedST_Layer(nn.Module):
    """简化版的时空层，专门用于风速数据处理"""
    def __init__(self, K, d, T=12, window_size=5):
        super(SimplifiedST_Layer, self).__init__()
        D = K * d
        
        # 简化的时空注意力
        self.temporal_attn = nn.Sequential(
            nn.Conv2d(D, D, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.BatchNorm2d(D)
        )
        
        # MLP处理
        self.mlp = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, D),
            nn.ReLU(),
            nn.Linear(D, D)
        )
    
    def forward(self, X, STE):
        # 时间注意力
        X_temp = X.permute(0, 3, 1, 2)  # B,D,T,N
        X_temp = self.temporal_attn(X_temp)
        X_temp = X_temp.permute(0, 2, 3, 1)  # B,T,N,D
        
        # 残差连接
        X = X + X_temp
        
        # MLP处理
        X_out = self.mlp(X)
        X = X + X_out
        
        return X

class GatedFusion(nn.Module):
    """门控融合模块"""
    def __init__(self, traffic_dim, wind_dim):
        super(GatedFusion, self).__init__()
        
        # 将风速特征映射到交通特征的维度
        self.wind_transform = nn.Sequential(
            nn.Linear(wind_dim, traffic_dim),
            nn.ReLU()
        )
        
        # 门控单元
        self.gate = nn.Sequential(
            nn.Linear(traffic_dim + traffic_dim, traffic_dim),
            nn.Sigmoid()
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(traffic_dim + traffic_dim, traffic_dim),
            nn.LayerNorm(traffic_dim),
            nn.ReLU()
        )
        
        # 输出校准
        self.calibration = nn.Sequential(
            nn.Linear(traffic_dim, traffic_dim),
            nn.Sigmoid()
        )
    
    def forward(self, traffic_feat, wind_feat):
        # 转换风速特征
        wind_feat_trans = self.wind_transform(wind_feat)
        
        # 计算门控值
        concat_feat = torch.cat([traffic_feat, wind_feat_trans], dim=-1)
        gate_value = self.gate(concat_feat)
        
        # 融合特征
        fused_feat = self.fusion(concat_feat)
        
        # 应用门控
        gated_feat = traffic_feat + gate_value * fused_feat
        
        # 校准
        calibration = self.calibration(gated_feat)
        
        return gated_feat * calibration

class AdaptiveFeatureWeighting(nn.Module):
    """自适应特征权重模块"""
    def __init__(self, feature_dim):
        super(AdaptiveFeatureWeighting, self).__init__()
        
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 全局平均池化
            nn.Flatten(),
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, feature_dim),
            nn.Sigmoid()
        )
        
        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        batch_size, T, N, C = x.shape
        
        # 通道注意力
        channel_weights = self.channel_attention(x.permute(0, 3, 1, 2))
        channel_weights = channel_weights.view(batch_size, 1, 1, C)
        
        # 通道加权
        channel_weighted = x * channel_weights
        
        # 空间注意力 (在时间和空间维度上)
        x_perm = channel_weighted.permute(0, 3, 1, 2)  # B,C,T,N
        avg_pool = torch.mean(x_perm, dim=1, keepdim=True)  # B,1,T,N
        max_pool, _ = torch.max(x_perm, dim=1, keepdim=True)  # B,1,T,N
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)  # B,2,T,N
        spatial_weights = self.spatial_attention(spatial_input)  # B,1,T,N
        
        # 应用空间权重
        spatial_weights = spatial_weights.permute(0, 2, 3, 1)  # B,T,N,1
        spatial_weighted = channel_weighted * spatial_weights
        
        return spatial_weighted + x  # 残差连接

def create_multimodal_model(SE, args, g=None):
    """创建多模态模型"""
    device = args.device if hasattr(args, 'device') else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    N = args.num_nodes if hasattr(args, 'num_nodes') else SE.shape[0]
    
    model = MultiModalModel(
        SE=SE, 
        args=args, 
        window_size=args.window, 
        T=args.num_his, 
        N=N
    )
    
    return model.to(device) 