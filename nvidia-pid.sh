#!/bin/bash

#使用 nvidia-smi 过滤并打印占用显存的 PID
nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sort -u
