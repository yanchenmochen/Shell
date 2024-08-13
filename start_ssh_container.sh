#!/bin/bash
docker restart vllm-smoothquant
docker restart smoothquant
docker restart lmdeploy050

docker exec -it smoothquant /etc/init.d/ssh restart
docker exec -it lmdeploy050 /etc/init.d/ssh restart
docker exec -it vllm-smoothquant /etc/init.d/ssh restart

