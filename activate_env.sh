#!/bin/bash

grep -q "songquanheng/env.sh" ~/.bashrc || echo "source /mnt/seed-program-nas/001688/songquanheng/env.sh" >> ~/.bashrc && source ~/.bashrc