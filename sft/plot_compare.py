import matplotlib.pyplot as plt
from numpy import average

def plot_layer_diffs(data_lines, output_file='mae_diff.png', bins=20):
    layers = []
    avg_mae = []
    max_mae = []

    for line in data_lines:
        if "diff" not in line:
            continue
        parts = line.strip().split()
        layer_name = parts[0] + " " + parts[1]
        nums = line.split('(')[1].split(')')[0].split(',')
        mean_val = float(nums[0])
        max_val = float(nums[1])
        layers.append(layer_name)
        avg_mae.append(mean_val)
        max_mae.append(max_val)

    if len(layers) > bins:
        layers = layers[:bins]
        avg_mae = avg_mae[:bins]
        max_mae = max_mae[:bins]

    x = range(len(layers))
    width = 0.4

    plt.figure(figsize=(12, 6))
    plt.bar([i - width/2 for i in x], avg_mae, width=width, label='average MAE')
    plt.bar([i + width/2 for i in x], max_mae, width=width, label='max MAE')

    plt.xticks(x, layers, rotation=45, ha='right')
    plt.ylabel('MAE')
    plt.title('layers comparision')
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig(output_file, dpi=300)  
    print(f"✅ png saved to : {output_file}")

# 示例数据（可从文件或标准输入读取）
data = """
layer0 input diff  (0.0, 0.0)
layer0 attn_input diff  (0.0009002685546875, 0.03125)
layer0 attn_output diff  (0.00157928466796875, 0.0625)
layer0 mlp_input diff  (0.06396484375, 3.015625)
layer0 mlp_output diff  (0.006683349609375, 0.375)
layer1 input diff  (0.006683349609375, 0.375)
layer1 attn_input diff  (0.00970458984375, 0.953125)
layer1 attn_output diff  (0.01239013671875, 0.75)
layer1 mlp_input diff  (0.005523681640625, 0.050048828125)
layer1 moe_output diff  (0.00640869140625, 0.390625)
layer2 mlp_output diff  (0.09619140625, 3.5)
""".strip().splitlines()

import sys

if __name__ == "__main__":
    # 从 stdin 读取输入数据
    data = sys.stdin.read().strip().splitlines()
    
    # 从命令行参数读取输出文件名（可选）
    output_file = sys.argv[1] if len(sys.argv) > 2 else 'mae_diff.png'
    bins = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    plot_layer_diffs(data, output_file,bins)
