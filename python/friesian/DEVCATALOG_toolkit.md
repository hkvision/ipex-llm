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

| Recommended Hardware         | Precision  |
| ---------------------------- | ---------- |
| Intel® 4th Gen Xeon® Scalable Performance processors|BF16 |
| Intel® 1st, 2nd, 3rd, and 4th Gen Xeon® Scalable Performance processors| FP32 |


## How it Works

<img src="https://github.com/intel-analytics/BigDL/blob/main/scala/friesian/src/main/resources/images/architecture.png" width="80%" />

The architecture above illustrates the main components in Intel® Recsys Toolkit.

- The offline training workflow is implemented based on Spark, Ray and BigDL to efficiently scale the data processing and DNN model training on large Xeon clusters.
- The online serving workflow is implemented based on gRPC and HTTP, which consists of Recall, Ranking, Feature and Recommender services. The Recall Service integrates Intel® Optimized Faiss to significantly speed up the vector search step.


## Get Started
Same as workflow. Copy here after the workflow contents are confirmed.

---

## Run Using Docker
Same as workflow. Copy here after the workflow contents are confirmed.

---

## Run Using Bare Metal
Same as workflow. Copy here after the workflow contents are confirmed.


## Expected Output
Same as workflow. Copy here after the workflow contents are confirmed.


## Summary and Next Steps
This page demonstrates how to use Intel® Recsys Toolkit to build end-to-end training and serving pipelines for Wide & Deep model.
You can continue to explore more use cases or recommendation models provided in the toolkit or try to use the toolkit to build
the recommender system on your own dataset!

## Learn More
For more information about Intel® Recsys Toolkit or to read about other relevant workflow
examples, see these guides and software resources:

- More recommendation models in the recsys toolkit: https://github.com/intel-analytics/BigDL/tree/main/python/friesian/example
- Online serving guide in the recsys toolkit: https://github.com/intel-analytics/BigDL/tree/main/scala/friesian
- [Intel® AI Analytics Toolkit (AI Kit)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html)
- [Azure Machine Learning Documentation](https://learn.microsoft.com/en-us/azure/machine-learning/)

## Troubleshooting
Same as workflow. Copy here after the workflow contents are confirmed.


## Support
If you have questions or issues about this workflow, contact the Support Team through [GitHub](https://github.com/intel-analytics/BigDL/issues) or [Google User Group](https://groups.google.com/g/bigdl-user-group).
