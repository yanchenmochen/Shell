#!/bin/bash
dev_id=$1
lines=$(xpu-smi -q -i $dev_id | grep -E "Used|Draw")

# 从$lines中提取Power Draw和Used的值
power=$(echo "$lines" | grep "Power Draw" | grep -o "[0-9.]*")
memory_used=$(echo "$lines" | grep "Used" | head -n 1 | grep -o '[0-9]*')
echo "$power    $memory_used"