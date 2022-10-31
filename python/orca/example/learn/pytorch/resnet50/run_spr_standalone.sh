#!/usr/bin/env bash

#export SSL_CERT_DIR=/etc/ssl/certs
#export LD_PRELOAD="/opt/work/kai/libjemalloc.so":$LD_PRELOAD
#export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
#
#export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
#export KMP_BLOCKTIME=1
#export KMP_AFFINITY=granularity=fine,compact,1,0

# int8
python inference.py /opt/work/imagenet --workers_per_node 1 --steps 200 \
    --int8 --configure_dir /opt/work/kai/resnet50_configure_sym.json \
    --ipex --seed 2020 -j 0 -b 116 \
    --cluster_mode standalone --num_nodes 2 --cores 48 --master spark://10.67.124.192:7077

# bf16
#python inference.py /opt/work/imagenet --workers_per_node 1 --steps 200 \
#     --bf16 --jit \
#     --ipex --seed 2020 -j 0 -b 68 \
#     --cluster_mode standalone --num_nodes 2 --cores 48 --master spark://10.67.124.192:7077

# fp32
#python inference.py /opt/work/imagenet --workers_per_node 1 --steps 200 \
#     --jit \
#     --ipex --seed 2020 -j 0 -b 64 \
#     --cluster_mode standalone --num_nodes 2 --cores 48 --master spark://10.67.124.192:7077
