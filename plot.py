import re
import matplotlib.pyplot as plt

# 日志文件路径
log_file_path = '/data/jcw/jcw/ASTTN_pytorch/save/road_dual_optimized.log'

# 用于存储提取的数据
epochs = []
train_losses = []
val_losses = []
test_maes = []
test_rmses = []
test_mapes = []
learning_rates = []

# 定义正则表达式
epoch_pattern = re.compile(r"轮次:\s*(\d+)/\d+")
train_loss_pattern = re.compile(r"训练损失:\s*([\d.]+)")
val_loss_pattern = re.compile(r"验证损失:\s*([\d.]+)")
test_metrics_pattern = re.compile(r"测试指标\s*-\s*MAE:\s*([\d.]+),\s*RMSE:\s*([\d.]+),\s*MAPE:\s*([\d.]+)%")
lr_pattern = re.compile(r"当前学习率:\s*([\d.]+)")

with open(log_file_path, 'r', encoding='utf-8') as f:
    for line in f:
        # 提取轮次
        m = epoch_pattern.search(line)
        if m:
            epochs.append(int(m.group(1)))
        # 提取训练损失
        m = train_loss_pattern.search(line)
        if m:
            train_losses.append(float(m.group(1)))
        # 提取验证损失
        m = val_loss_pattern.search(line)
        if m:
            val_losses.append(float(m.group(1)))
        # 提取测试指标
        m = test_metrics_pattern.search(line)
        if m:
            test_maes.append(float(m.group(1)))
            test_rmses.append(float(m.group(2)))
            test_mapes.append(float(m.group(3)))
        # 提取学习率
        m = lr_pattern.search(line)
        if m:
            learning_rates.append(float(m.group(1)))

# 确保所有数据长度一致
min_length = min(len(epochs), len(train_losses), len(val_losses))
epochs = epochs[:min_length]
train_losses = train_losses[:min_length]
val_losses = val_losses[:min_length]

# 确保测试指标长度一致
test_min_length = min(len(epochs), len(test_maes), len(test_rmses), len(test_mapes))
epochs_for_test = epochs[:test_min_length]
test_maes = test_maes[:test_min_length]
test_rmses = test_rmses[:test_min_length]
test_mapes = test_mapes[:test_min_length]

# 打印数据长度信息
print(f"数据长度 - 轮次: {len(epochs)}, 训练损失: {len(train_losses)}, 验证损失: {len(val_losses)}")
print(f"测试指标长度 - MAE: {len(test_maes)}, RMSE: {len(test_rmses)}, MAPE: {len(test_mapes)}")

# 绘制并保存：训练与验证损失
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_losses, marker='o', label='Training Loss')
plt.plot(epochs, val_losses, marker='s', label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training & Validation Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('loss_curve.png', dpi=300)
plt.close()

# 绘制并保存：测试指标（MAE、RMSE、MAPE）
plt.figure(figsize=(8, 5))
plt.plot(epochs_for_test, test_maes, marker='o', label='Test MAE')
plt.plot(epochs_for_test, test_rmses, marker='s', label='Test RMSE')
plt.plot(epochs_for_test, test_mapes, marker='^', label='Test MAPE')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Test Metrics Over Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('test_metrics_curve.png', dpi=300)
plt.close()

# 绘制并保存：学习率变化
if learning_rates:
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(learning_rates) + 1), learning_rates, marker='o')
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('lr_schedule.png', dpi=300)
    plt.close()
else:
    print("未检测到学习率数据，跳过学习率曲线绘制")

print("图表已分别保存为：")
print("  • loss_curve.png")
print("  • test_metrics_curve.png")
if learning_rates:
    print("  • lr_schedule.png")
