#!/bin/bash
dev_id=$1
xpu-smi -i $dev_id -q | grep "Memory Usage"  -A  4 | grep  Used | grep -o '[0-9]*'