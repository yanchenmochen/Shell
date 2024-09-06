#!/bin/bash

docker ps -as --format '{{.ID}}: {{.Names}}: {{.Size}}' | sort -rh -k3
