import time
import argparse
import torch.optim as optim
from utils import log_string, masked_mae
from utils import count_parameters, load_data, load_graph, load_data_with_wind
# from model.model_both import Model_Both

from model.train import train
from model.model_adp import Model_Adp
from model.model_both import Model_Both
from model.model_dual import Model_Dual
import matplotlib
# 设置 matplotlib 使用非交互式后端，适合服务器环境
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import math

# 添加GPU显存管理函数
def configure_gpu_memory():
    if torch.cuda.is_available():
        # 设置cudnn为非确定性模式，以获得更好的性能
        torch.backends.cudnn.deterministic = False
        # 启用cudnn基准测试模式，提高性能并增加显存占用
        torch.backends.cudnn.benchmark = True
        
        # 设置PyTorch的内存分配策略
        # 这将使PyTorch预先分配大量显存，并在其中管理内存
        torch.cuda.empty_cache()  # 清空缓存
        
        # 禁用PyTorch的内存缓存释放
        # 这会让PyTorch一次性分配大量显存，然后在内部管理，而不是频繁地申请和释放
        os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '0'  # 启用内存缓存
        
        # 报告可用的GPU数量和信息
        gpu_count = torch.cuda.device_count()
        print(f"检测到 {gpu_count} 个可用GPU")
        
        # 为每个GPU报告信息但不预分配显存
        reserved_tensors = []
        for i in range(gpu_count):
            with torch.cuda.device(i):
                prop = torch.cuda.get_device_properties(i)
                print(f"GPU {i}: {prop.name}, 显存总量: {prop.total_memory / 1024 / 1024 / 1024:.2f} GB")
                
                # 报告当前分配情况
                allocated = torch.cuda.memory_allocated(i) / 1024 / 1024 / 1024
                reserved = torch.cuda.memory_reserved(i) / 1024 / 1024 / 1024
                print(f"GPU {i}: 已分配 {allocated:.2f} GB, 已预留 {reserved:.2f} GB")
                
                # 创建一个小张量保持设备活跃
                keep_alive = torch.ones(1, device=f"cuda:{i}")
                reserved_tensors.append(keep_alive)
        
        return reserved_tensors
    return None

# 添加绘制训练曲线的函数
def plot_training_curves(train_loss, val_loss, metrics_history=None, save_path='./save/', log=None):
    """
    绘制训练和验证损失曲线以及评估指标曲线，保存为图片文件
    
    参数:
    - train_loss: 训练损失列表
    - val_loss: 验证损失列表
    - metrics_history: 包含评估指标的字典，格式为 {'mae': [], 'rmse': [], 'mape': []}
    - save_path: 保存图像的路径
    - log: 日志对象
    """
    # 确保张量数据移动到CPU
    if torch.is_tensor(train_loss):
        train_loss = train_loss.cpu().numpy()
    if torch.is_tensor(val_loss):
        val_loss = val_loss.cpu().numpy()
    
    # 确保保存目录存在
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        if log:
            log_string(log, f"创建保存目录: {save_path}")
    
    # 设置中文字体支持 (如果服务器上安装了中文字体)
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    except:
        # 如果没有中文字体，使用英文
        pass
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_loss, 'b-', label='训练损失 (Train Loss)')
    plt.plot(epochs, val_loss, 'r-', label='验证损失 (Val Loss)')
    plt.title('训练和验证损失 (Train and Validation Loss)')
    plt.xlabel('轮次 (Epoch)')
    plt.ylabel('损失 (Loss)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'loss_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    if log:
        log_string(log, f"保存损失曲线图到: {os.path.join(save_path, 'loss_curves.png')}")
    
    # 如果提供了评估指标历史，则绘制评估指标曲线
    if metrics_history is not None:
        # 确保指标数据也转移到CPU
        metrics_cpu = {}
        for key, value in metrics_history.items():
            if torch.is_tensor(value):
                metrics_cpu[key] = value.cpu().numpy()
            else:
                metrics_cpu[key] = value
        
        # 绘制MAE曲线
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, metrics_cpu['mae'], 'g-', label='MAE')
        plt.title('测试集MAE指标 (Test MAE)')
        plt.xlabel('轮次 (Epoch)')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_path, 'mae_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
        if log:
            log_string(log, f"保存MAE曲线图到: {os.path.join(save_path, 'mae_curve.png')}")
        
        # 绘制RMSE曲线
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, metrics_cpu['rmse'], 'm-', label='RMSE')
        plt.title('测试集RMSE指标 (Test RMSE)')
        plt.xlabel('轮次 (Epoch)')
        plt.ylabel('RMSE')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_path, 'rmse_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
        if log:
            log_string(log, f"保存RMSE曲线图到: {os.path.join(save_path, 'rmse_curve.png')}")
        
        # 绘制MAPE曲线
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, [x*100 for x in metrics_cpu['mape']], 'c-', label='MAPE(%)')
        plt.title('测试集MAPE指标 (Test MAPE)')
        plt.xlabel('轮次 (Epoch)')
        plt.ylabel('MAPE(%)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_path, 'mape_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
        if log:
            log_string(log, f"保存MAPE曲线图到: {os.path.join(save_path, 'mape_curve.png')}")
        
        # 绘制所有指标在一个图中
        plt.figure(figsize=(12, 6))
        
        # 归一化指标以便在同一图表中比较趋势
        mae_norm = np.array(metrics_cpu['mae']) / max(metrics_cpu['mae'])
        rmse_norm = np.array(metrics_cpu['rmse']) / max(metrics_cpu['rmse'])
        mape_norm = np.array(metrics_cpu['mape']) / max(metrics_cpu['mape'])
        
        plt.plot(epochs, mae_norm, 'g-', label='归一化MAE (Normalized MAE)')
        plt.plot(epochs, rmse_norm, 'm-', label='归一化RMSE (Normalized RMSE)')
        plt.plot(epochs, mape_norm, 'c-', label='归一化MAPE (Normalized MAPE)')
        plt.title('测试集评估指标趋势比较 (Normalized Metrics Comparison)')
        plt.xlabel('轮次 (Epoch)')
        plt.ylabel('归一化值 (Normalized Value)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_path, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        if log:
            log_string(log, f"保存归一化指标比较图到: {os.path.join(save_path, 'metrics_comparison.png')}")
        
        # 保存指标数据到CSV文件，方便后续分析
        try:
            import pandas as pd
            metrics_df = pd.DataFrame({
                'epoch': epochs,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'mae': metrics_cpu['mae'],
                'rmse': metrics_cpu['rmse'],
                'mape': [x*100 for x in metrics_cpu['mape']]
            })
            metrics_df.to_csv(os.path.join(save_path, 'training_metrics.csv'), index=False)
            if log:
                log_string(log, f"训练指标数据已保存到 {os.path.join(save_path, 'training_metrics.csv')}")
        except Exception as e:
            if log:
                log_string(log, f"保存CSV数据失败: {str(e)}")
    
    if log:
        log_string(log, f"所有训练曲线图像已保存到 {save_path} 目录")


parser = argparse.ArgumentParser()
parser.add_argument('--time_slot', type=int, default=5,
                    help='a time step is 5 mins')
parser.add_argument('--num_his', type=int, default=12,
                    help='history steps')
parser.add_argument('--num_pred', type=int, default=12,
                    help='prediction steps')
parser.add_argument('--L', type=int, default=1,
                    help='number of block layers')
parser.add_argument('--K', type=int, default=8,
                    help='number of attention heads')
parser.add_argument('--d', type=int, default=8,
                    help='dims of each head attention outputs')
parser.add_argument('--window', type=int, default=6,
                    help='temporal window size for attentions')
parser.add_argument('--train_ratio', type=float, default=0.7,
                    help='training set [default : 0.7]')
parser.add_argument('--val_ratio', type=float, default=0.1,
                    help='validation set [default : 0.1]')
parser.add_argument('--test_ratio', type=float, default=0.2,
                    help='testing set [default : 0.2]')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch size')
parser.add_argument('--max_epoch', type=int, default=100,
                    help='epoch to run')
parser.add_argument('--optimizer', type=str, default='adam',
                    help='optimizer type: adam, adamw, sgd')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='weight decay/L2 regularization strength')
parser.add_argument('--dropout', type=float, default=0.3,
                    help='dropout rate')
parser.add_argument('--scheduler', type=str, default='cosine',
                    help='learning rate scheduler: step, cosine, onecycle')
parser.add_argument('--decay_epoch', type=int, default=5,
                    help='learning rate decay period (for step scheduler)')
parser.add_argument('--decay_rate', type=float, default=0.9,
                    help='learning rate decay rate (for step scheduler)')
parser.add_argument('--early_stop', type=int, default=15,
                    help='early stopping patience, 0 means no early stopping')
parser.add_argument('--grad_clip', type=float, default=5.0,
                    help='gradient clipping threshold')
parser.add_argument('--ds', default='road',
                    help='dataset name')
parser.add_argument('--se_type', default='node2vec',
                    help='spatial embedding file')
parser.add_argument('--remark', default='',
                    help='remark')
parser.add_argument('--model', default='adp',
                    help='model_type: adp, both, or multimodal')
parser.add_argument('--use_multi_gpu', action='store_true',
                    help='use multiple GPUs if available')
parser.add_argument('--wind_file', default='/data/jcw/ASTTN_pytorch/data/data_wind_speed_all.h5',
                    help='path to wind speed data file')
parser.add_argument('--use_wind', action='store_true',
                    help='use wind speed data as additional input')

args = parser.parse_args()

args.traffic_file = './data/' + args.ds + '.h5'
args.model_file = './save/' + args.ds + '_' + args.model + args.remark + '.pkl'

save_log =  './save/'+args.ds+'_'+args.model+args.remark + '.log'
log = open(save_log, 'a')
log_string(log, "New...")
log_string(log, str(args)[10: -1])

T = 24 * 60 // args.time_slot

# 确定使用的设备
# 不再硬编码设备ID
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
log_string(log, f'使用设备: {device}')

# load data
log_string(log, 'loading data...')
(trainX, trainTE, trainY, valX, valTE, valY, testX, testTE,
 testY, mean, std) = load_data(args)
SE, g = load_graph(args)

# 将数据移动到主设备
SE = SE.to(device)

# 检查DGL是否支持CUDA
has_cuda_dgl = False
if device.type == 'cuda':
    try:
        # 尝试一个简单的操作来测试DGL的CUDA支持
        if g is not None:
            import dgl
            # 创建一个小的测试图来检查CUDA支持
            test_g = dgl.graph(([0], [1]))
            test_g = test_g.to(device)
            test_g = None  # 释放测试图
            has_cuda_dgl = True
            log_string(log, "DGL支持CUDA，继续使用GPU")
    except Exception as e:
        log_string(log, f"DGL不支持CUDA: {str(e)}")
        log_string(log, "切换到CPU模式")
        device = torch.device('cpu')
        SE = SE.to(device)

# 根据DGL是否支持CUDA决定是否将图迁移到GPU
if g is not None:
    if has_cuda_dgl and device.type == 'cuda':
        try:
            g = g.to(device)
            log_string(log, "成功将图数据迁移到GPU")
        except Exception as e:
            log_string(log, f"将图迁移到GPU时出错: {str(e)}")
            log_string(log, "切换到CPU模式")
            device = torch.device('cpu')
            SE = SE.to(device)
    else:
        log_string(log, "在CPU上使用图数据")

# 如果使用both模型但DGL不支持CUDA，则切换到adp模型
if args.model == "both" and (not has_cuda_dgl) and device.type == 'cuda':
    log_string(log, "模型both需要GPU支持的DGL，但当前环境不支持。切换到adp模型")
    args.model = "adp"

num_nodes = trainY.shape[2]
log_string(log, f'trainX: {trainX.shape}\t\t trainY: {trainY.shape}')
log_string(log, f'valX:   {valX.shape}\t\tvalY:   {valY.shape}')
log_string(log, f'testX:   {testX.shape}\t\ttestY:   {testY.shape}')
log_string(log, f'mean:   {mean:.4f}\t\tstd:   {std:.4f}')
log_string(log, 'data loaded!')

# 保存数据形状以便后续创建风速数据
trainX_shape = trainX.shape
valX_shape = valX.shape
testX_shape = testX.shape

del trainX, trainTE, valX, valTE, testX, testTE, mean, std
# build model
log_string(log, 'compiling model...')

if args.model == "adp":
    model = Model_Adp(SE, args, N = num_nodes, T = args.num_his, window_size = args.window)
elif args.model == "both":
    model = Model_Both(SE, args, N = num_nodes, T=args.num_his, window_size=args.window, g = g)
elif args.model == "multimodal":
    # 导入多模态模型
    from model.model_multimodal import create_multimodal_model
    model = create_multimodal_model(SE, args, g=g)
    args.num_nodes = num_nodes  # 设置节点数以便后续使用
    # 添加device属性
    args.device = device
    args.log = log
elif args.model == "dual":
    log_string(log, "使用双模态模型 (traffic + wind)")
    # 添加num_nodes属性到args
    args.num_nodes = num_nodes
    # 检查SE是否已经是张量，如果不是则转换
    if not isinstance(SE, torch.Tensor):
        SE = torch.from_numpy(SE).to(device)
    else:
        SE = SE.to(device)
    # 将dropout参数传递给模型
    model = Model_Dual(SE, args, window_size=args.window, T=args.num_his, N=args.num_nodes)

# 将模型移动到选择的设备
try:
    model = model.to(device)
    log_string(log, f'模型已成功移动到 {device} 设备')
except Exception as e:
    log_string(log, f'将模型移动到 {device} 时出错: {str(e)}')
    if device.type == 'cuda':
        log_string(log, '尝试在CPU上运行模型')
        device = torch.device('cpu')
        model = model.to(device)
    
# 如果指定使用多GPU并且可用，则使用DataParallel
if args.use_multi_gpu and torch.cuda.device_count() > 1 and device.type == 'cuda':
    try:
        log_string(log, f'尝试使用 {torch.cuda.device_count()} 个GPU进行训练')
        model = torch.nn.DataParallel(model)
        log_string(log, '成功启用多GPU模式')
    except Exception as e:
        log_string(log, f'启用多GPU模式失败: {str(e)}')
        log_string(log, '使用单GPU模式继续')
elif args.use_multi_gpu and device.type != 'cuda':
    log_string(log, '已指定多GPU但当前运行在CPU模式，禁用多GPU')
elif args.use_multi_gpu and torch.cuda.device_count() <= 1:
    log_string(log, f'已指定多GPU但只检测到 {torch.cuda.device_count()} 个GPU，禁用多GPU')

loss_criterion = masked_mae

# 设置优化器，根据命令行参数选择
if args.optimizer.lower() == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
elif args.optimizer.lower() == 'adamw':
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
elif args.optimizer.lower() == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
else:
    log_string(log, f"未知的优化器类型 {args.optimizer}，使用默认的Adam")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

# 选择学习率调度器
if args.scheduler.lower() == 'step':
    # 步进式学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_epoch, gamma=args.decay_rate)
    log_string(log, f"使用步进式学习率调度器 - 每{args.decay_epoch}个epoch衰减为{args.decay_rate}倍")
elif args.scheduler.lower() == 'cosine':
    # 余弦退火学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch, eta_min=1e-6)
    log_string(log, f"使用余弦退火学习率调度器 - T_max={args.max_epoch}, eta_min=1e-6")
elif args.scheduler.lower() == 'onecycle':
    # One Cycle学习率调度器
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.learning_rate, 
        steps_per_epoch=1, epochs=args.max_epoch,
        pct_start=0.3, anneal_strategy='cos'
    )
    log_string(log, f"使用OneCycle学习率调度器 - max_lr={args.learning_rate}")
else:
    # 默认使用StepLR
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_epoch, gamma=args.decay_rate)
    log_string(log, f"未知的调度器类型 {args.scheduler}，使用默认的步进式学习率调度器")

parameters = count_parameters(model)

log_string(log, 'trainable parameters: {:,}'.format(parameters))

if __name__ == '__main__':
    # 配置GPU显存占用
    reserved_tensors = configure_gpu_memory()
    
    start = time.time()
    
    # 如果是多模态模型，需要加载风速数据
    if args.model == "multimodal":
        from wind_data_utils import load_wind_data, preprocess_wind_data, split_wind_data
        
        # 加载风速数据
        log_string(log, '加载风速数据...')
        wind_data = load_wind_data(args)
        
        # 获取交通数据的总形状
        total_traffic_shape = (trainY.shape[0], trainY.shape[1], trainY.shape[2])
        
        if len(wind_data) > 0:
            log_string(log, '预处理风速数据...')
            processed_wind = preprocess_wind_data(wind_data, total_traffic_shape)
            
            # 分割风速数据
            train_len = trainY.shape[0]
            val_len = valY.shape[0]
            trainWind, valWind, testWind = split_wind_data(processed_wind, train_len, val_len, log)
        else:
            # 如果没有风速数据，创建零填充的数据
            log_string(log, '无法加载风速数据，使用零填充替代')
            trainWind = torch.zeros(trainX_shape[0], trainX_shape[1], trainX_shape[2])
            valWind = torch.zeros(valX_shape[0], valX_shape[1], valX_shape[2])
            testWind = torch.zeros(testX_shape[0], testX_shape[1], testX_shape[2])
        
        # 将风速数据移到设备上
        trainWind = trainWind.to(device)
        valWind = valWind.to(device)
        testWind = testWind.to(device)
        
        # 使用专门的多模态训练函数
        from model.train_multimodal import train_multimodal
        loss_train, loss_val, metrics_history = train_multimodal(
            model, args, log, loss_criterion, optimizer, scheduler,
            trainWind, valWind, testWind
        )
    else:
        # 使用原始训练函数
        loss_train, loss_val, metrics_history = train(model, args, log, loss_criterion, optimizer, scheduler)
    
    # 绘制训练曲线
    save_path = './save/curves_' + args.ds + '_' + args.model + args.remark
    plot_training_curves(train_loss=loss_train, val_loss=loss_val, metrics_history=metrics_history, save_path=save_path, log=log)
    
    end = time.time()
    log_string(log, f'总训练时间: {(end - start)/3600:.2f}小时')
    
    # 关闭日志文件
    log.close()
    print(f"训练完成！日志已保存到 {save_log}")
    print(f"训练曲线已保存到 {save_path} 目录")
    
    # 训练结束后再次报告显存使用情况
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024 / 1024 / 1024
            reserved = torch.cuda.memory_reserved(i) / 1024 / 1024 / 1024
            print(f"训练结束后 GPU {i}: 已分配 {allocated:.2f} GB, 已预留 {reserved:.2f} GB")
    
    # 清理显存占用
    if reserved_tensors:
        del reserved_tensors
        torch.cuda.empty_cache()
        print("已清理GPU显存占用")