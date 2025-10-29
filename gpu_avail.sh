#!/bin/bash

python -c "import torch; print('device count:',torch.cuda.device_count(), 'available: ', torch.cuda.is_available())"

