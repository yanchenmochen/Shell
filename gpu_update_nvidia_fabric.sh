#!/bin/bash
# 解决nvidia-fabricmanager自动更新的问题

apt install nvidia-fabricmanager-550=550.54.14-1

systemctl restart nvidia-fabricmanager

systemctl status nvidia-fabricmanager
