# Building Recommender Systems with Intel® Recsys Toolkit

Use Intel® Recsys Toolkit, BigDL Friesian, to easily build large-scale distributed training and online serving
pipelines for modern recommender systems. This page demonstrates how to use this toolkit to build a recommendation solution with Wide & Deep Learning model.

Check out more workflow examples and reference implementations in the [Developer Catalog](https://developer.intel.com/aireferenceimplementations).

## Overview
Building an end-to-end recommender system that meets production demands from scratch could be rather challenging.
Intel® Recsys Toolkit, i.e. BigDL Friesian, can greatly help relieve the efforts of building distributed offline training
and online serving pipelines. The recommendation solutions built with the toolkit are optimized on Intel® Xeon
and could be directly deployed on large clusters to handle production big data.

Highlights and benefits of Intel® Recsys Toolkit are as follows:

- Provide various built-in distributed feature engineering operations to efficient process user and item features.
- Support the distributed training of any standard TensorFlow or PyTorch model. 
- Implement a complete, highly available and scalable pipeline for online serving (including recall and ranking) with low latency.
- Include out-of-use reference use cases of many popular recommendation algorithms.

For more details, visit the BigDL Friesian [GitHub repository](https://github.com/intel-analytics/BigDL/tree/main/python/friesian) and
[documentation page](https://bigdl.readthedocs.io/en/latest/doc/Friesian/index.html).

## Hardware Requirements

Intel® Recsys Toolkit and the workflow example shown below could be run widely on Intel® Xeon® series processors.

|| Recommended Hardware         |
|---| ---------------------------- |
|CPU| Intel® Xeon® Scalable processors with Intel®-AVX512|
|Memory|>10G|
|Disk|>10G|


## How it Works

<img src="https://github.com/intel-analytics/BigDL/blob/main/scala/friesian/src/main/resources/images/architecture.png" width="80%" />

The architecture above illustrates the main components in Intel® Recsys Toolkit.

- The offline training workflow is implemented based on Spark, Ray and BigDL to efficiently scale the data processing and DNN model training on large Xeon clusters.
- The online serving workflow is implemented based on gRPC and HTTP, which consists of Recall, Ranking, Feature and Recommender services. The Recall Service integrates Intel® Optimized Faiss to significantly speed up the vector search step.


## Get Started

### Download the Workflow Repository
Create a working directory for the workflow and clone the [Main
Repository](https://github.com/intel-analytics/BigDL) repository into your working
directory.

```
mkdir ~/work && cd ~/work
git clone https://github.com/intel-analytics/BigDL.git
cd BigDL
```

### Download the Datasets

This workflow uses the [Twitter Recsys Challenge 2021 dataset](http://www.recsyschallenge.com/2021/), each record of which contains the tweet along with engagement features, user features, and tweet features.

The original dataset includes 46 million users and 340 million tweets (items). Here in this workflow, we provide a script to generate some dummy data for this dataset. In the running command below, you can specify the number of records to generate and the output folder respectively.

```
cd apps/wide-deep-recommendation
mkdir recsys_data
# You can modify the number of records and the output folder when running the script
python generate_dummy_data.py 100000 recsys_data/
cd ../..
```

---

## Run Training Workflow Using Docker
Follow these instructions to set up and run our provided Docker image.
For running the training workflow on bare metal, see the [bare metal instructions](#run-training-workflow-using-bare-metal)
instructions.

### Set Up Docker Engine
You'll need to install Docker Engine on your development system.
Note that while **Docker Engine** is free to use, **Docker Desktop** may require
you to purchase a license.  See the [Docker Engine Server installation
instructions](https://docs.docker.com/engine/install/#server) for details.

If the Docker image is run on a cloud service, mention they may also need
credentials to perform training and inference related operations (such as these
for Azure):
- [Set up the Azure Machine Learning Account](https://azure.microsoft.com/en-us/free/machine-learning)
- [Configure the Azure credentials using the Command-Line Interface](https://docs.microsoft.com/en-us/cli/azure/authenticate-azure-cli)
- [Compute targets in Azure Machine Learning](https://learn.microsoft.com/en-us/azure/machine-learning/concept-compute-target)
- [Virtual Machine Products Available in Your Region](https://azure.microsoft.com/en-us/explore/global-infrastructure/products-by-region/?products=virtual-machines&regions=us-east)

### Set Up Docker Image
Pull the provided docker image.
```
docker pull intelanalytics/bigdl-spark-3.1.3:latest
```

If your environment requires a proxy to access the internet, export your
development system's proxy settings to the docker environment:
```
export DOCKER_RUN_ENVS="-e ftp_proxy=${ftp_proxy} \
  -e FTP_PROXY=${FTP_PROXY} -e http_proxy=${http_proxy} \
  -e HTTP_PROXY=${HTTP_PROXY} -e https_proxy=${https_proxy} \
  -e HTTPS_PROXY=${HTTPS_PROXY} -e no_proxy=${no_proxy} \
  -e NO_PROXY=${NO_PROXY} -e socks_proxy=${socks_proxy} \
  -e SOCKS_PROXY=${SOCKS_PROXY}"
```

### Run Docker Image
Run the workflow using the ``docker run`` command, as shown:
```
export DATASET_DIR=/path/to/BigDL/apps/wide-deep-recommendation/recsys_data
export OUTPUT_DIR=/output
docker run -a stdout $DOCKER_RUN_ENVS \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --volume ${DATASET_DIR}:/workspace/data \
  --volume ${OUTPUT_DIR}:/output \
  --volume ${PWD}:/workspace \
  --workdir /workspace \
  --privileged --init -it --rm --pull always \
  intelanalytics/bigdl-spark-3.1.3:latest \
  bash
```

---

## Run Training Workflow Using Bare Metal
Follow these instructions to set up and run this workflow on your own development
system. For running the training workflow with a provided Docker image, see the [Docker
instructions](#run-training-workflow-using-docker).


### Set Up System Software
Our examples use the ``conda`` package and environment on your local computer.
If you don't already have ``conda`` installed, see the [Conda Linux installation
instructions](https://docs.conda.io/projects/conda/en/stable/user-guide/install/linux.html).

### Set Up Workflow
Run these commands to set up the workflow's conda environment and install required software:
```
conda create -n recsys python=3.9 --yes
conda activate recsys
pip install --pre --upgrade bigdl-friesian
pip install intel-tensorflow==2.9.0
```

### Run Workflow
Use these commands to run the workflow:
```
python python/friesian/example/wnd/recsys2021/wnd_preprocess_recsys.py \
    --executor_cores 8 \
    --executor_memory 10g \
    --input_train_folder apps/wide-deep-recommendation/recsys_data/train \
    --input_test_folder apps/wide-deep-recommendation/recsys_data/test \
    --output_folder apps/wide-deep-recommendation/recsys_data/preprocessed \
    --cross_sizes 600

python python/friesian/example/wnd/recsys2021/wnd_train_recsys.py \
    --executor_cores 8 \
    --executor_memory 10g \
    --data_dir apps/wide-deep-recommendation/recsys_data/preprocessed \
    --model_dir recsys_wnd/ \
    --batch_size 3200 \
    --epoch 5 \
    --learning_rate 1e-4 \
    --early_stopping 3

cd python/friesian/example/two_tower
python train_2tower.py \
    --executor_cores 8 \
    --executor_memory 10g \
    --data_dir apps/wide-deep-recommendation/recsys_data/preprocessed \
    --model_dir recsys_2tower/ \
    --batch_size 8000
```

## Expected Training Workflow Output
Check out the processed data and saved models of the workflow:
```
ll apps/wide-deep-recommendation/recsys_data/preprocessed
ll recsys_wnd/
ll recsys_2tower/
```
Check out the logs of the console for training results:

- wnd_train_recsys.py:
```
22/25 [=========================>....] - ETA: 1s - loss: 0.2367 - binary_accuracy: 0.9391 - binary_crossentropy: 0.2367 - auc: 0.5637 - precision: 0.9392 - recall: 1.0000
23/25 [==========================>...] - ETA: 0s - loss: 0.2374 - binary_accuracy: 0.9388 - binary_crossentropy: 0.2374 - auc: 0.5644 - precision: 0.9388 - recall: 1.0000
24/25 [===========================>..] - ETA: 0s - loss: 0.2378 - binary_accuracy: 0.9386 - binary_crossentropy: 0.2378 - auc: 0.5636 - precision: 0.9386 - recall: 1.0000
25/25 [==============================] - ETA: 0s - loss: 0.2379 - binary_accuracy: 0.9385 - binary_crossentropy: 0.2379 - auc: 0.5635 - precision: 0.9385 - recall: 1.0000
25/25 [==============================] - 10s 391ms/step - loss: 0.2379 - binary_accuracy: 0.9385 - binary_crossentropy: 0.2379 - auc: 0.5635 - precision: 0.9385 - recall: 1.0000 - val_loss: 0.6236 - val_binary_accuracy: 0.8491 - val_binary_crossentropy: 0.6236 - val_auc: 0.4988 - val_precision: 0.9342 - val_recall: 0.9021
(Worker pid=11371) Epoch 4: early stopping
Training time is:  53.32298707962036
```
- train_2tower.py:
```
7/10 [====================>.........] - ETA: 0s - loss: 0.3665 - binary_accuracy: 0.8124 - recall: 0.8568 - auc: 0.5007
8/10 [=======================>......] - ETA: 0s - loss: 0.3495 - binary_accuracy: 0.8282 - recall: 0.8747 - auc: 0.5004
9/10 [==========================>...] - ETA: 0s - loss: 0.3370 - binary_accuracy: 0.8403 - recall: 0.8886 - auc: 0.5002
10/10 [==============================] - ETA: 0s - loss: 0.3262 - binary_accuracy: 0.8503 - recall: 0.8998 - auc: 0.5002
10/10 [==============================] - 7s 487ms/step - loss: 0.3262 - binary_accuracy: 0.8503 - recall: 0.8998 - auc: 0.5002 - val_loss: 0.2405 - val_binary_accuracy: 0.9352 - val_recall: 1.0000 - val_auc: 0.4965
```

---

## Run Online Serving Pipeline Using Docker
You are highly recommended to run the online serving pipeline for the recsys workflow using our provided Docker image.

### Set Up Docker Image
Pull the provided docker image.
```
docker pull intelanalytics/friesian-serving:2.2.0-SNAPSHOT
```

If your environment requires a proxy to access the internet, export your
development system's proxy settings to the docker environment:
```
export DOCKER_RUN_ENVS="-e ftp_proxy=${ftp_proxy} \
  -e FTP_PROXY=${FTP_PROXY} -e http_proxy=${http_proxy} \
  -e HTTP_PROXY=${HTTP_PROXY} -e https_proxy=${https_proxy} \
  -e HTTPS_PROXY=${HTTPS_PROXY} -e no_proxy=${no_proxy} \
  -e NO_PROXY=${NO_PROXY} -e socks_proxy=${socks_proxy} \
  -e SOCKS_PROXY=${SOCKS_PROXY}"
```

Download & install [redis](https://redis.io/download/#redis-downloads)

### Run Workflow
- Run the nearline pipeline

1. Flush all key-values in the redis
```bash
redis-cli flushall
```
2. Check the initial redis status
```bash
redis-cli info keyspace
```
Output:
```bash
# Keyspace
```

3. Run the following script to launch the nearline pipeline
```bash
docker_name=intelanalytics/friesian-serving:2.2.0-SNAPSHOT

docker run -it --net host --rm -v $(pwd):/opt/work/mnt $docker_name feature-init -c mnt/nearline/config_feature.yaml

docker run -it --net host --rm -v $(pwd):/opt/work/mnt $docker_name feature-init -c mnt/nearline/config_feature_vec.yaml

docker run -it --net host --rm -v $(pwd):/opt/work/mnt $docker_name recall-init -c mnt/nearline/config_recall.yaml
```

4. Check the redis-server status
```bash
redis-cli info keyspace
```
Output:
```bash
# Keyspace
db0:keys=2003,expires=0,avg_ttl=0
```

5. Check the existance of the generated faiss index
```bash
item_50.idx
```

- Run the online pipeline
1. Run the following script to launch the online pipeline
```bash
docker_name=intelanalytics/friesian-serving:2.2.0-SNAPSHOT

docker run -itd --net host  --rm --name ranking -v $(pwd):/opt/work/mnt -e OMP_NUM_THREADS=1 $docker_name ranking -c mnt/config_ranking.yaml

docker run -itd --net host --rm --name feature -v $(pwd):/opt/work/mnt $docker_name feature -c mnt/config_feature.yaml

docker run -itd --net host --rm --name feature_recall -v $(pwd):/opt/work/mnt $docker_name feature -c mnt/config_feature_vec.yaml

docker run -itd --net host --rm --name recall -v $(pwd):/opt/work/mnt $docker_name recall -c mnt/config_recall.yaml

#docker run -itd --net host --rm --name recommender -v $(pwd):/opt/work/mnt $docker_name recommender -c mnt/config_recommender.yaml

docker run -itd --net host  --rm --name recommender_http -v $(pwd):/opt/work/mnt $docker_name recommender-http -c mnt/config_recommender.yaml -p 8000
```

2. Check the status of the containers
- There are 5 containers running:
    - recommender_http
    - recall
    - feature_recall
    - feature
    - ranking

3. Confirm the application is accessible
```bash
curl http://localhost:8000/recommender/recommend/15
```
Output:
```bash
{
  "ids" : [ 640, 494, 90, 481, 772, 314, 6, 272, 176, 284 ],
  "probs" : [ 0.80175865, 0.6995631, 0.6851486, 0.6811177, 0.67750615, 0.67231035, 0.6655403, 0.65543735, 0.6547779, 0.6547779 ],
  "success" : true,
  "errorCode" : null,
  "errorMsg" : null
}
```


See [here](https://github.com/intel-analytics/BigDL/tree/main/scala/friesian) for more detailed guidance to run the online serving workflow.

See [here](https://github.com/intel-analytics/BigDL/tree/main/apps/friesian-server-helm) to deploy the serving workflow on a Kubernetes cluster.

## Summary and Next Steps
This page demonstrates how to use Intel® Recsys Toolkit to build end-to-end training and serving pipelines for Wide & Deep model.
You can continue to explore more use cases or recommendation models provided in the toolkit or try to use the toolkit to build
the recommender system on your own dataset!

## Learn More
For more information about Intel® Recsys Toolkit or to read about other relevant workflow
examples, see these guides and software resources:

- More recommendation models in the recsys toolkit: https://github.com/intel-analytics/BigDL/tree/main/python/friesian/example
- Online serving guidance in the recsys toolkit: https://github.com/intel-analytics/BigDL/tree/main/scala/friesian
- [Intel® AI Analytics Toolkit (AI Kit)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html)
- [Azure Machine Learning Documentation](https://learn.microsoft.com/en-us/azure/machine-learning/)

## Troubleshooting
- If you encounter the error `E0129 21:36:55.796060683 1934066 thread_pool.cc:254] Waiting for thread pool to idle before forking` during the training, it may be caused by the installed version of grpc. See [here](https://github.com/grpc/grpc/pull/32196) for more details about this issue. To fix it, a recommended grpc version is 1.43.0:
```bash
pip install grpcio==1.43.0
```

## Support
If you have questions or issues about this workflow, contact the Support Team through [GitHub](https://github.com/intel-analytics/BigDL/issues) or [Google User Group](https://groups.google.com/g/bigdl-user-group).
