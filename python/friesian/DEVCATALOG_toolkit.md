# Building Recommender Systems with Intel® Recsys Toolkit

Use Intel® Recsys Toolkit, also known as BigDL Friesian, to easily build large-scale distributed training and online serving
pipelines for modern recommender systems. This page demonstrates how to use this toolkit to build a recommendation solution for the Wide & Deep Learning workflow.

Check out more toolkits and reference implementations in the [Developer Catalog](https://developer.intel.com/aireferenceimplementations).

## Overview
Building an end-to-end recommender system that meets production demands from scratch could be rather challenging.
Intel® Recsys Toolkit, i.e. BigDL Friesian, can greatly help relieve the efforts of building distributed offline training
and online serving pipelines. The recommendation solutions built with the toolkit are optimized on Intel® Xeon
and could be directly deployed on existing large clusters to handle production big data.

Highlights and benefits of Intel® Recsys Toolkit are as follows:

- Provide various built-in distributed feature engineering operations to efficiently process user and item features.
- Support the distributed training of any standard TensorFlow or PyTorch model. 
- Implement a complete, highly available and scalable pipeline for online serving (including recall and ranking) with low latency.
- Include out-of-box reference use cases of many popular recommendation algorithms.

Intel® Recsys Toolkit is a domain-specific submodule in BigDL focusing on recommendation workloads, visit the BigDL Friesian [GitHub repository](https://github.com/intel-analytics/BigDL/tree/main/python/friesian) and
[documentation page](https://bigdl.readthedocs.io/en/latest/doc/Friesian/index.html) for more details.

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


---

## Get Started

### 1. Prerequisites

You are highly recommended to use the toolkit under the following system and software settings:
- OS: Linux or Mac
- Python: 3.7, 3.8, 3.9


### 2. Download the Toolkit Repository

Create a working directory for the workflow of Intel® Recsys Toolkit and clone the [Main
Repository](https://github.com/intel-analytics/BigDL) repository into your working
directory. Intel® Recsys Toolkit is included in the BigDL project and
this step downloads the example scripts in BigDL to demonstrate the toolkit.
Follow the steps in the next section to easily install Intel® Recsys Toolkit via [Docker](#31-install-from-docker) or [pip](#32-install-from-pypi-on-bare-metal).

```
mkdir ~/work && cd ~/work
git clone https://github.com/intel-analytics/BigDL.git
cd BigDL
```

### 3. Installation

#### 3.1 Install from Docker
Follow these instructions to set up and run our provided Docker image.
For running the training workflow on bare metal, see the [bare metal instructions](#32-install-from-pypi-on-bare-metal).

**a. Set Up Docker Engine**

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

**b. Set Up Docker Image**

Pull the provided Docker image:
```
docker pull intelanalytics/bigdl-orca:latest
```

If your environment requires a proxy to access the Internet, export your
development system's proxy settings to the Docker environment by adding `--env http_proxy=${http_proxy}` when you create the docker container in the next step.

**c. Create Docker Container**

Create the Docker container for BigDL using the ``docker run`` command, as shown below:
```
docker run -a stdout \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --env no_proxy=${no_proxy} \
  --volume ${PWD}:/workspace \
  --workdir /workspace \
  --privileged --init -it --rm \
  intelanalytics/bigdl-orca:latest \
  bash
```

**d. Install Packages in Docker Container**

Run these commands to install additional software used for the workflow in the Docker container:
```
pip install tensorflow==2.9.0
```


#### 3.2 Install from Pypi on Bare Metal
Follow these instructions to set up and run this workflow on your own development
system. For running the training workflow with a provided Docker image, see the [Docker
 instructions](#31-install-from-docker).


**a. Set Up System Software**

Our examples use the ``conda`` package and environment on your local computer.
If you don't already have ``conda`` installed, see the [Conda Linux installation
instructions](https://docs.conda.io/projects/conda/en/stable/user-guide/install/linux.html).

**b. Install Packages in Conda Environment**

Run these commands to set up the workflow's ``conda`` environment and install required software:
```
conda create -n recsys python=3.9 --yes
conda activate recsys
pip install --pre --upgrade bigdl-friesian-spark3
pip install tensorflow==2.9.0
```

---

## How To Run

### 1. Download the Datasets

This workflow of the toolkit uses the [Twitter Recsys Challenge 2021 dataset](http://www.recsyschallenge.com/2021/) as an example. In this dataset, each record contains the tweet along with engagement features, user features, and tweet features.

The original dataset includes 46 million users and 340 million tweets (items). Alternatively, here we provide a script to generate some dummy data for this dataset. In the running command below, you can specify the number of records to generate and the output folder respectively.

```
cd python/friesian/example/wnd/recsys2021
mkdir recsys_data

# You can modify the number of records and the output folder when running the script
python generate_dummy_data.py 100000 recsys_data/
```

### 2. Run Training Workflow

The training workflow of the toolkit will preprocess the dataset, train the Wide & Deep Learning model (for ranking) and two-tower model (for embeddings) with the processed data.

Use these commands to run the training workflow:
```
python wnd_preprocess_recsys.py \
    --executor_cores 8 \
    --executor_memory 6g \
    --data_dir recsys_data \
    --cross_sizes 600

python wnd_train_recsys.py \
    --backend spark \
    --executor_cores 8 \
    --executor_memory 6g \
    --data_dir recsys_data/preprocessed \
    --model_dir recsys_wnd \
    --batch_size 3200 \
    --epoch 5 \
    --learning_rate 1e-4 \
    --early_stopping 3

cd ../../two_tower
python train_2tower.py \
    --backend spark \
    --executor_cores 8 \
    --executor_memory 6g \
    --data_dir ../wnd/recsys2021/recsys_data/preprocessed \
    --model_dir ../wnd/recsys2021/recsys_2tower \
    --batch_size 8000

python predict_2tower.py \
    --backend spark \
    --executor_cores 8 \
    --executor_memory 6g \
    --data_dir ../wnd/recsys2021/recsys_data/preprocessed \
    --model_dir ../wnd/recsys2021/recsys_2tower
```

**Expected Training Workflow Output**

Check out the processed data and saved models after the training:
```
cd ../wnd/recsys2021
ll recsys_data/preprocessed
ll recsys_wnd
ll recsys_2tower
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


### Run Online Serving Pipeline Using Docker

After completing the training pipeline, you can use the trained model to deploy and test the online serving pipeline of the toolkit.

You are highly recommended to run the online serving pipeline using our provided Docker image.

Note that we have already prepared scripts to easily launch the Docker containers for online serving. You need to run the following steps on bare metal to start the online services. If you run the training pipeline in the Docker image, first type `Ctrl+D` or `exit` to exit the container and go back to your development system.


**a. Set Up Docker Image**

Pull the provided Docker image:

```
docker pull intelanalytics/friesian-serving:2.2.0-SNAPSHOT
```


**b. Download & install [redis](https://redis.io/download/#redis-downloads)**

```bash
wget https://github.com/redis/redis/archive/7.2-rc1.tar.gz
tar -xzf 7.2-rc1.tar.gz
cd 7.2-rc1.tar.gz && make
src/redis-server &
```

**c. Prepare model and features**

Copy the trained model and processed features to the folder where we run the serving scripts.

```bash
cd ~/work/BigDL/
cp -r recsys_wnd scala/friesian/
cp -r apps/wide-deep-recommendation/recsys_data/preprocessed/*.parquet scala/friesian/
cd scala/friesian/
```

**d. Run Workflow**

1. Flush all key-values in the redis:
```bash
redis-cli flushall
```

2. Check the initial redis status:
```bash
redis-cli info keyspace
```
Output:
```bash
# Keyspace
```

3. Run the following script to launch the nearline pipeline:
```bash
bash scripts/run_nearline.sh
```

4. Check the redis-server status:
```bash
redis-cli info keyspace
```
Output:
```bash
# Keyspace
db0:keys=300003,expires=0,avg_ttl=0
```

5. Check the existence of the generated [Faiss](https://github.com/facebookresearch/faiss) index for vector search:
```bash
ls -la item_128.idx
```

6. If your environment requires a proxy to access the Internet, unset it before running the online pipeline:
```bash
unset http_proxy https_proxy
```

7. Run the following script to launch the online pipeline:
```bash
bash scripts/run_online.sh
```

8. Check the status of the containers:
```bash
docker container ls
```
There should be 5 containers running:
- recommender_http
- recall
- feature_recall
- feature
- ranking

9. Confirm the application is accessible
```bash
curl http://localhost:8000/recommender/recommend/99999
```
Output:
```bash
{
  "ids" : [ 49498, 90939, 9237, 37407, 18638, 10772, 83555, 1175, 41118, 56338 ],
  "probs" : [ 0.8125731, 0.7951641, 0.78238714, 0.7734338, 0.7725358, 0.7724836, 0.7694705, 0.76804805, 0.76270276, 0.76186526 ],
  "success" : true,
  "errorCode" : null,
  "errorMsg" : null
}
```


---

## Summary and Next Steps
This page demonstrates how to use Intel® Recsys Toolkit to build end-to-end training and serving pipelines for Wide & Deep Learning model.
You can continue to explore more use cases or recommendation models provided in the toolkit or try to use the toolkit to build
the recommender system on your own dataset!

## Learn More
For more information about Intel® Recsys Toolkit or to read about other relevant workflow
examples, see these guides and software resources:

- More recommendation models and use cases in the recsys toolkit: https://github.com/intel-analytics/BigDL/tree/main/python/friesian/example
- Online serving guidance in the recsys toolkit: https://github.com/intel-analytics/BigDL/tree/main/scala/friesian
- To scale the training workflow of the recsys toolkit to Kubernetes clusters: https://bigdl.readthedocs.io/en/latest/doc/Orca/Tutorial/k8s.html
- To scale the online serving workflow of the recsys toolkit to Kubernetes clusters: https://github.com/intel-analytics/BigDL/tree/main/apps/friesian-server-helm
- [Intel® AI Analytics Toolkit (AI Kit)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html)
- [Azure Machine Learning Documentation](https://learn.microsoft.com/en-us/azure/machine-learning/)

## Troubleshooting
- If you encounter the error `E0129 21:36:55.796060683 1934066 thread_pool.cc:254] Waiting for thread pool to idle before forking` during the training, it may be caused by the installed version of grpc. See [here](https://github.com/grpc/grpc/pull/32196) for more details about this issue. To fix it, a recommended grpc version is 1.43.0:
```bash
pip install grpcio==1.43.0
```

## Support
If you have questions or issues about this workflow, contact the Support Team through [GitHub](https://github.com/intel-analytics/BigDL/issues) or [Google User Group](https://groups.google.com/g/bigdl-user-group).
