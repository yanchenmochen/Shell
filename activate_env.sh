#!/bin/bash

grep -q "songquanheng/env.sh" ~/.bashrc || echo "source /mnt/self-define/songquanheng/env.sh" >> ~/.bashrc && source ~/.bashrc