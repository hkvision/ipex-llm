# Distributed Recsys Training & Inference Workflows on BigDL 

Learn to use BigDL's recommendation framework Friesian to easily build distributed training and online serving
pipelines for Wide & Deep Learning model.

Check out more workflow examples and reference implementations in the [Developer Catalog](https://developer.intel.com/aireferenceimplementations).

## Overview
Building an end-to-end recommender system that meets production demands from scratch could be rather challenging.
As an application framework for large-scale recommender solutions optimized on Intel Xeon,
BigDL Friesian can greatly help relieve the efforts of
building distributed training and online serving workflows.

Highlights and benefits of Friesian are as follows:

- Friesian provides various built-in distributed feature engineering operations to efficient process user and item features.
- Friesian supports the distributed training of any standard TensorFlow or PyTorch model. 
- Friesian implements a complete, highly available and scalable pipeline for online serving (including recall and ranking) with low latency.
- Friesian contains reference user cases of many popular recommendation algorithms.

For more details, visit the BigDL Friesian [GitHub repository](https://github.com/intel-analytics/BigDL/tree/main/python/friesian) and
[documentation page](https://bigdl.readthedocs.io/en/latest/doc/Friesian/index.html).

## Hardware Requirements

| Recommended Hardware         | Precision  |
| ---------------------------- | ---------- |
| Intel® 4th Gen Xeon® Scalable Performance processors|BF16 |
| Intel® 1st, 2nd, 3rd, and 4th Gen Xeon® Scalable Performance processors| FP32 |


## How it Works

<img src="https://github.com/intel-analytics/BigDL/blob/main/scala/friesian/src/main/resources/images/architecture.png" width="100%" />

The architecture above illustrates the main components in Friesian.

- Friesian's training workflow pis implemented based on Spark, Ray and [BigDL Orca](https://bigdl.readthedocs.io/en/latest/doc/Orca/index.html) to efficiently scale the data processing and model training on large Xeon clusters.
- Friesian's online serving workflow is implemented based on gRPC and HTTP services with Intel Optimized Faiss integrated to significantly speed up the vector search step.


## Get Started

### Download the Workflow Repository
Create a working directory for the workflow and clone the [Main
Repository](https://github.com/intel-analytics/BigDL) repository into your working
directory.

```
mkdir ~/work && cd ~/work
git clone https://github.com/intel-analytics/BigDL.git
cd BigDL
git checkout ai-workflow
```

### Download the Datasets

This workflow uses the [Twitter Recsys Challenge 2021 dataset](http://www.recsyschallenge.com/2021/), each record of which contains the tweet along with engagement features, user features, and tweet features.

The original dataset includes 46 million users and 340 million tweets (items). Here in this workflow, we provide a script to generate some dummy data for this dataset.

```
cd apps/wide-deep-recommendation
mkdir recsys_data
python generate_dummy_data.py --output_path recsys_data/ --num_samples 100000
cd ..
```

---

## Run Using Docker
Follow these instructions to set up and run our provided Docker image.
For running on bare metal, see the [bare metal instructions](#run-using-bare-metal)
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
# TODO: release a docker image for friesian
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
Run the workflow using the ``docker run`` command, as shown:  (example)
```
# TODO: test and modify this

export DATASET_DIR=apps/recsys_data
export OUTPUT_DIR=/output
docker run -a stdout $DOCKER_RUN_ENVS \
  --env DATASET=${DATASET} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --volume ${DATASET_DIR}:/workspace/data \
  --volume ${OUTPUT_DIR}:/output \
  --volume ${PWD}:/workspace \
  --workdir /workspace \
  --privileged --init -it --rm --pull always \
  intel/ai-workflows:bigdl-training \
  ./run.sh
```

---

## Run Using Bare Metal
Follow these instructions to set up and run this workflow on your own development
system. For running a provided Docker image with Docker, see the [Docker
instructions](#run-using-docker).


### Set Up System Software
Our examples use the ``conda`` package and environment on your local computer.
If you don't already have ``conda`` installed, see the [Conda Linux installation
instructions](https://docs.conda.io/projects/conda/en/stable/user-guide/install/linux.html).

### Set Up Workflow
Run these commands to set up the workflow's conda environment and install required software:
```
conda create -n friesian python=3.8 --yes
conda activate friesian
pip install --pre --upgrade bigdl-friesian[train]
pip install tensorflow==2.9.0
```

### Run Workflow
Use these commands to run the workflow:
```
python python/friesian/example/wnd/recsys2021/wnd_preprocess_recsys.py \
    --executor_cores 8 \
    --executor_memory 10g \
    --train_files 0-1 \
    --input_train_folder apps/recsys_data/train \
    --input_test_folder apps/recsys_data/test \
    --output_folder apps/recsys_data/preprocessed \
    --cross_sizes 600

python python/friesian/example/wnd/recsys2021/wnd_train_recsys.py \
    --executor_cores 8 \
    --executor_memory 10g \
    --data_dir apps/recsys_data/preprocessed \
    --model_dir recsys_wnd/ \
    --batch_size 3200 \
    --epoch 10 \
    --learning_rate 1e-4 \
    --early_stopping 3

cd python/friesian/example/two_tower
python train_2tower.py \
    --executor_cores 8 \
    --executor_memory 10g \
    --data_dir apps/recsys_data/preprocessed \
    --model_dir recsys_2tower/
```

## Expected Output
Check out the processed data and saved models of the workflow:
```
ll apps/recsys_data/preprocessed
ll recsys_wnd/
ll recsys_2tower/
```
Check out the logs of the console for training results:
```
76/79 [===========================>..] - ETA: 0s - loss: 0.4995 - binary_accuracy: 0.7749 - binary_crossentropy: 0.4995 - auc: 0.6873
77/79 [============================>.] - ETA: 0s - loss: 0.4995 - binary_accuracy: 0.7749 - binary_crossentropy: 0.4995 - auc: 0.6874
79/79 [==============================] - ETA: 0s - loss: 0.4993 - binary_accuracy: 0.7750 - binary_crossentropy: 0.4993 - auc: 0.6875
79/79 [==============================] - 6s 81ms/step - loss: 0.4993 - binary_accuracy: 0.7750 - binary_crossentropy: 0.4993 - auc: 0.6875 - val_loss: 0.5382 - val_binary_accuracy: 0.7730 - val_binary_crossentropy: 0.5382 - val_auc: 0.6826
Training time is:  26.327171802520752
Stopping orca context
```

## Summary and Next Steps
Now you have successfully tried the recsys workflows of BigDL to build an end-to-end pipeline for Wide & Deep model.
You can continue to try more use cases or recommendation models provided in BigDL or try to build the workflows on your own dataset!

## Learn More
For more information about BigDL Recsys workflows or to read about other relevant workflow
examples, see these guides and software resources:

- BigDL Friesian Recsys training examples: https://github.com/intel-analytics/BigDL/tree/main/python/friesian/example
- BigDL Friesian Recsys serving guide: https://github.com/intel-analytics/BigDL/tree/main/scala/friesian
- [Intel® AI Analytics Toolkit (AI Kit)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html)
- [Azure Machine Learning Documentation](https://learn.microsoft.com/en-us/azure/machine-learning/)

## Troubleshooting
N/A

## Support
If you have questions or issues about this workflow, contact the Support Team through [GitHub](https://github.com/intel-analytics/BigDL/issues) or [Google User Group](https://groups.google.com/g/bigdl-user-group).
