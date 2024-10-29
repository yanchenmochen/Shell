#!/bin/bash

dir=$1
process_num=$2
find "$dir" -mindepth 1 -maxdepth 1 -type d | xargs -I {} -P "$process_num" du -sh {}


