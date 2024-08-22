#!/bin/bash


xpu-smi -q -i $1 | grep "Power Draw" | grep -o "[0-9.]*"