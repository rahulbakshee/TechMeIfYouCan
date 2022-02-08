![nn]({{ '/images/2022-02-08-food.png' | relative_url }})
{: style="width: 600px; max-width: 100%;"}

While searching for jobs, we are often given take-home assignments to test our skills on a near real world problem. Although it's a very good way to judge a candidate's proficiency in solving a problem, it may be time consuming and sometimes very frustrating when no feedback is given after hours of hard work.

But anyway, in this challenge, we are given a food classification dataset which has 101 classes. [Source](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) . We need to analyze and preprocess the dataset as well as build deep learning models for performing food classification. We are free to choose any Deep Learning framework out there.

As this dataset is already present at tensorflow_datasets and can be easily downloaded as a tfds as shown https://www.tensorflow.org/datasets/catalog/food101 . But, we will download raw data and then preprocess it according to our needs to showcase technical skills as real world data may come in any shape and size.

Judging criteria would be how to format raw data and put in respective directories, create train-test split, data visualization, data preprocessing, data augmnetation, input pipelines, modeling (transfer learning, fine tuning), validation, metrics visualization, inference etc.

Bonus: You may get extra points for *Deployment*

> follow the code here **[google colab](https://colab.research.google.com/drive/1k-MocSgk8OoaNQqtkbsjDLdBCfeJ7kMV?usp=sharing)**


# 1. download the data
```
# download raw data
!wget http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz

# unzip the data
!tar xzf food-101.tar.gz #add v for verbose #xvzf

!ls food-101

# list all the subdirectories(101 classes of food) under "/images"
!ls food-101/images

# README.txt shows how the directory structure
!cat food-101/README.txt
```
import required libraries
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import sys, gc, os, shutil, PIL
from tqdm.notebook import tqdm
seed = 42

import tensorflow as tf
print("tensorflow version", tf.__version__)

AUTOTUNE = tf.data.experimental.AUTOTUNE
tf.config.list_physical_devices('GPU')

```
```
base_dir = "food-101/"

os.listdir(base_dir)

# list the number of classes
labels = []

with open (base_dir+"meta/classes.txt", 'r') as file:
    for cls in tqdm(file):
        labels.append(cls[:-1])
    
print("[INFO] Total number of classes", len(labels))
# print(labels)

```
 
 
 
 
 # 2. Create train and test subdirectories
 
 ```
 def create_train_test(folder):
    """ 
    creates subfolders for train and test under root dir
    copies files from subfolders(classes) under "/images" to subfolders of "/train" & "/test
    
    """

    # creates subfolders for train and test under root dir
    for cls in labels:
        os.makedirs(base_dir + folder + '/' + cls)
    
    # copies image files
    with open (base_dir+"meta/" + folder + ".txt", 'r') as file:
        for f in tqdm(file) :
            shutil.move(src=base_dir+"images/" + f[:-1]+ ".jpg", 
                        dst=base_dir+folder+"/" + f[:-1]+ ".jpg" )

            
create_train_test(folder="train" )
print("[INFO] subfolders created for train data")
create_train_test(folder="test" )
print("[INFO] subfolders created for test data")

# see the newly created train/test folders
os.listdir(base_dir)

```
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
