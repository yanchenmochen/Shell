#!/bin/bash

if command -v zjlab-gmi >/dev/null 2>&1; then
    CMD=zjlab-gmi
else
    CMD=mthreads-gmi
fi

$CMD | grep Processes -A 100 | awk '/^[0-9]+[[:space:]]+[0-9]+/ {print $2}' | sort -n | uniq | xargs -r kill -9