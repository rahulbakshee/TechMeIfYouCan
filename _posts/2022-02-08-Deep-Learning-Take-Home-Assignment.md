![nn]({{ '/images/2022-02-08-food.png' | relative_url }})
{: style="width: 600px; max-width: 100%;"}

While searching for jobs, we are often given take-home assignments to test our skills on a near real world problem. Although it's a very good way to judge a candidate's proficiency in solving a problem, it may be time consuming and sometimes very frustrating when no feedback is given after hours of hard work.

But anyway, in this challenge, we are given a food classification dataset which has 101 classes. [Source](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)

We need to analyze and preprocess the dataset as well as build deep learning models for performing food classification. We are free to choose any Deep Learning framework out there.

As this dataset is already present at tensorflow_datasets and can be easily downloaded as a tfds as shown https://www.tensorflow.org/datasets/catalog/food101 . But, we will download raw data and then preprocess it according to our needs to showcase technical skills as real world data may come in any shape and size.

Judging criteria would be how to format raw data and put in respective directories, create train-test split, data visualization, data preprocessing, data augmnetation, input pipelines, modeling (transfer learning, fine tuning), validation, metrics visualization, inference etc.

Bonus: Deployment

> follow the code here **[google colab](https://colab.research.google.com/drive/1k-MocSgk8OoaNQqtkbsjDLdBCfeJ7kMV?usp=sharing)**


# 1. download the data
