#!/bin/bash

# 检查是否提供了文件参数
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <data-file>"
    exit 1
fi

# 文件路径
data_file="$1"

# 检查文件是否存在
if [ ! -f "$data_file" ]; then
    echo "File not found: $data_file"
    exit 1
fi

# 初始化表头
header="| Test Name   | Iterations | Run Seconds | Precision | Batch Size | Throughput (images/sec) | Memory Average (GFLOPS) | GFLOPS | Power Average (W) | Energy Efficiency (GFLOPS/W) |"
separator="| ----------- | ---------- | ----------- | --------- | ---------- | ----------------------- | ----------------------- | ------- | ------------------ | ---------------------------- |"

# 打印表头
echo "$header"
echo "$separator"

# 处理文件中的每一行数据
while IFS= read -r line; do
    # 提取各项数据
    test_name=$(echo "$line" | awk '{print $1}')
    iterations=$(echo "$line" | awk '{print $3}')
    run_seconds=$(echo "$line" | awk '{print $6}')
    precision=$(echo "$line" | awk '{print $8}')
    batch_size=$(echo "$line" | awk '{print $10}')
    throughput=$(echo "$line" | awk '{print $13}')
    memory_avg=$(echo "$line" | awk '{print $17}')
    gflops=$(echo "$line" | awk '{print $20}')
    power_avg=$(echo "$line" | awk '{print $24}')
    energy_efficiency=$(echo "$line" | awk '{print $28}')
    
    # 输出为Markdown表格格式
    echo "| $test_name | $iterations | $run_seconds | $precision | $batch_size | $throughput | $memory_avg | $gflops | $power_avg | $energy_efficiency |"
done < "$data_file"
