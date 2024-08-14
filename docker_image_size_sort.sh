#!/bin/bash
# 参数的含义如下所示：
# -k 指定排序的列，默认为7，即镜像大小
# -n 指定排序的方式，将103G等作为数字103，而非1.
# -r 逆序排列

docker images | grep -v MB | sort -k 7 -n -r
