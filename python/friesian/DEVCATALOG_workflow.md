# Distributed Training & Inference Workflows using BigDL 

Learn to use BigDL to easily build and seamlessly scale distributed training and inference workflows
for TensorFlow and PyTorch. This page takes the recsys workflows for Neural Collaborative Filtering (NCF) model as an example.

Check out more workflow examples and reference implementations in the [Developer Catalog](https://developer.intel.com/aireferenceimplementations).

## Overview
Usually one would take many efforts to develop distributed training and inference pipelines to handle large datasets in production.
With the adoption of BigDL, AI workflows written in Python notebooks running on a single laptop can be seamlessly
scaled out across large clusters so as to process distributed big data.

Highlights and benefits of BigDL are as follows:

- Easily build efficient in-memory, distributed data analytics and AI pipelines that runs on a single Intel® Xeon cluster.
- Seamlessly scale TensorFlow and PyTorch applications to big data platforms.
- Direct deployment on production clusters including Hadoop/YARN and Kubernetes clusters.


For more details, visit the BigDL [GitHub repository](https://github.com/intel-analytics/BigDL/tree/main) and
[documentation page](https://bigdl.readthedocs.io/en/latest/).

## Hardware Requirements

BigDL and the workflow example shown below could be run widely on Intel® Xeon® series processors.

|| Recommended Hardware         |
|---| ---------------------------- |
|CPU| Intel® Xeon® Scalable processors|
|Memory|>10G|
|Disk|>10G|


## How it Works

<img src="./bigdl-workflow.png" width="80%" />

The architecture above illustrates how BigDL can build end-to-end, distributed and in-memory pipelines on Intel® Xeon clusters.

- BigDL supports loading data from various distributed data sources and data formats that are widely used in the big data ecosystem.
- BigDL supports distributed data processing with Spark DataFrame, Ray Dataset and provides APIs for distributed data parallel processing of Python libraries.
- BigDL supports seamlessly scaling many popular deep learning frameworks and includes runtime optimizations on Xeon.

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

This workflow uses the [ml-100k dataset](https://grouplens.org/datasets/movielens/100k/) of [MovieLens](https://movielens.org/), which contains users' ratings for the movies.

```
cd python/orca/tutorial/NCF
wget https://files.grouplens.org/datasets/movielens/ml-100k.zip
unzip ml-100k.zip
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
docker pull intelanalytics/bigdl-spark-3.1.3:latest
```

If your environment requires a proxy to access the internet, export your
development system's proxy settings to the docker environment via `--env http_proxy=${http_proxy}`.

### Run Docker Image
Run the workflow using the ``docker run`` command, as shown:
```
docker run -a stdout $DOCKER_RUN_ENVS \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --env no_proxy=${no_proxy} \
  --volume ${PWD}:/workspace \
  --workdir /workspace \
  --privileged --init -it --rm --pull always \
  intelanalytics/bigdl-spark-3.1.3:latest \
  bash
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
conda create -n bigdl python=3.9 --yes
conda activate bigdl
pip install --pre --upgrade bigdl-orca-spark3
pip install intel-tensorflow==2.9.1
```

### Run Workflow
Use these commands to run the workflow:
```
# Distributed training
python tf_train_spark_dataframe.py --dataset ml-100k

# Distributed inference
python tf_predict_spark_dataframe.py --dataset ml-100k
```

## Expected Output
Check out the processed data and saved models of the workflow:
```
ll train_processed_dataframe.parquet
ll test_processed_dataframe.parquet
ll test_predictions_dataframe.parquet
ll NCF_model.h5
```
Check out the logs of the console for training results:

- tf_train_spark_dataframe.py:
```
Train results:
verbose: 1
epochs: 2
steps: 40
loss: [0.43995094299316406, 0.3661007285118103]
accuracy: [0.7953540086746216, 0.83966064453125]
auc: [0.7579973936080933, 0.8532146215438843]
precision: [0.3079235553741455, 0.6916240453720093]
recall: [0.017480555921792984, 0.3596118092536926]
val_loss: [0.3642280697822571, 0.3102317750453949]
val_accuracy: [0.8291406035423279, 0.865966796875]
val_auc: [0.855850875377655, 0.8976992964744568]
val_precision: [0.7450749278068542, 0.7225849628448486]
val_recall: [0.21683736145496368, 0.5314843058586121]

Evaluation results:
validation_loss: 0.3142370283603668
validation_accuracy: 0.8649218678474426
validation_auc: 0.8953894376754761
validation_precision: 0.7280181050300598
validation_recall: 0.5282735824584961
```
- tf_predict_spark_dataframe.py:
```
+----+----+-----+-------+------+----------+--------------------+--------+--------------------+
|item|user|label|zipcode|gender|occupation|                 age|category|          prediction|
+----+----+-----+-------+------+----------+--------------------+--------+--------------------+
|   1| 627|  0.0|    172|     1|         5|[0.25757575757575...|     102| [0.841558039188385]|
|   1| 697|  1.0|     74|     1|         2|[0.2727272727272727]|     102|[0.6905139088630676]|
|   3| 500|  1.0|    705|     1|         4|[0.3181818181818182]|       8|[0.31478825211524...|
|   5| 815|  0.0|    314|     1|         2|[0.3787878787878788]|      30|[0.27323466539382...|
|  16| 682|  0.0|    477|     1|         6|[0.24242424242424...|       3|[0.10835579037666...|
+----+----+-----+-------+------+----------+--------------------+--------+--------------------+
only showing top 5 rows
```


## Summary and Next Steps
Now you have successfully tried the recsys workflows of BigDL to build an end-to-end pipeline for Wide & Deep model.
You can continue to try other use cases provided in BigDL or build the training and inference workflows on your own dataset!

## Learn More
For more information about BigDL distributed training and inference workflows or to read about other relevant workflow
examples, see these guides and software resources:

- More BigDL workflow examples for TensorFlow: https://github.com/intel-analytics/BigDL/tree/main/python/orca/example/learn/tf2
- More BigDL workflow examples for PyTorch: https://github.com/intel-analytics/BigDL/tree/main/python/orca/example/learn/pytorch
- [Intel® AI Analytics Toolkit (AI Kit)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html)
- [Azure Machine Learning Documentation](https://learn.microsoft.com/en-us/azure/machine-learning/)

## Troubleshooting
- If you encounter the error `E0129 21:36:55.796060683 1934066 thread_pool.cc:254] Waiting for thread pool to idle before forking` during the training, it may be caused by the installed version of grpc. See [here](https://github.com/grpc/grpc/pull/32196) for more details about this issue. To fix it, a recommended grpc version is 1.43.0:
```bash
pip install grpcio==1.43.0
```

## Support
If you have questions or issues about this workflow, contact the Support Team through [GitHub](https://github.com/intel-analytics/BigDL/issues) or [Google User Group](https://groups.google.com/g/bigdl-user-group).
