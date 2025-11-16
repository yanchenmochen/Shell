#!/bin/bash
# monitor_modeling_deepseek.sh - 监控 modeling_deepseek.py 文件变化并自动拷贝

# 要监控的文件路径
MONITOR_FILE="/root/.cache/huggingface/modules/transformers_modules/hf_iter_0032800/modeling_deepseek.py"

# 目标目录1
TARGET_DIR1="/mnt/seed-program-nas/001688/dongjie/tokenizer/021-16b/hf_iter_0032800"

# 目标目录2
TARGET_DIR2="/mnt/seed-program-nas/001688/songquanheng/Shell/huggingface/model/021-16B"


# 确保目标目录存在
mkdir -p "$TARGET_DIR1"
mkdir -p "$TARGET_DIR2"

# 日志函数
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" 
    echo "$1"
}

# 检查文件是否存在
if [ ! -f "$MONITOR_FILE" ]; then
    log_message "错误: 监控文件 $MONITOR_FILE 不存在"
    exit 1
fi

# 检查 inotifywait 是否可用
if ! command -v inotifywait &> /dev/null; then
    log_message "错误: inotifywait 未安装，请先运行: sudo apt install inotify-tools"
    exit 1
fi

log_message "开始监控文件: $MONITOR_FILE"
log_message "目标目录1: $TARGET_DIR1"
log_message "目标目录2: $TARGET_DIR2"

# 初始拷贝（确保开始时文件是最新的）
if [ -f "$MONITOR_FILE" ]; then
    cp "$MONITOR_FILE" "$TARGET_DIR1/"
    cp "$MONITOR_FILE" "$TARGET_DIR2/"
    log_message "初始拷贝完成"
fi

# 监控文件变化
inotifywait -m -e modify --format "%w%f %e" "$MONITOR_FILE" | while read FILE EVENT
do
    echo "触发到文件修改事件"
    # 检查文件是否仍然存在（防止在监控期间被删除）
    if [ -f "$FILE" ]; then
        # 执行拷贝操作
        cp "$FILE" "$TARGET_DIR1/" && \
        cp "$FILE" "$TARGET_DIR2/"
        
        if [ $? -eq 0 ]; then
            log_message "文件已更新并拷贝到目标目录 (事件: $EVENT)"
            
            # 可选：验证拷贝结果
            if cmp -s "$FILE" "$TARGET_DIR1/modeling_deepseek.py" && \
               cmp -s "$FILE" "$TARGET_DIR2/modeling_deepseek.py"; then
                log_message "拷贝验证成功"
            else
                log_message "警告: 拷贝文件可能不匹配"
            fi
        else
            log_message "错误: 文件拷贝失败"
        fi
    else
        log_message "警告: 文件不存在，跳过本次处理"
    fi
done