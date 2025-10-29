#!/bin/bash

zjlab-gmi | grep Processes -A 100 | awk '/^[0-9]+[[:space:]]+[0-9]+/ {print $2}' | sort -n | uniq | xargs -r kill -9