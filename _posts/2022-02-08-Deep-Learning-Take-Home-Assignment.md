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
### import required libraries
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
 
# 3. List number of samples in training and testing folders.
The list is too long...so this piece of code just prints total number of samples.

If we want to list all the files, we can set list_files = True

```
def list_and_count_files(folder, list_files=False):
    """ 
    lists number of samples in training and testing folders
    
    """
    counter = 0
    for root, dirs, files in os.walk(base_dir + folder + "/"):
        for file in files:
            if list_files:
                print(os.path.join(root,file))
        counter += len(files)
    print("[INFO] Total number of samples in {} folder".format(folder), counter)

# avoiding printing the all the file names in the display
# set list_files to true if needed to print file names
list_and_count_files(folder="train",list_files=False)
list_and_count_files(folder="test",list_files=False)

```

# 4. Plot sample images from training and testing folders.

```
def display_images(folder):
    """
    plots sample images from training and testing datasets.
    """
    _, axes_list = plt.subplots(5, 7, figsize=(15,10))
    for axes in axes_list:
        for ax in axes:
            ax.axis('off')
            lbl = np.random.choice(os.listdir(base_dir+folder+"/"))
            img = np.random.choice(os.listdir(base_dir+folder+"/" + lbl))
            ax.imshow(PIL.Image.open(base_dir+folder+"/" + lbl+"/"+img))
            ax.set_title(lbl)

```
```
# display_images(folder="train")
# display_images(folder="test")
```


# 5. create the train, val, test datasets out of train and test

```
# params
IMG_SIZE = (200, 200) # keep it low to avoid colab crashing
IMG_SHAPE = IMG_SIZE + (3,)
NUM_CLASSES = 101
BATCH_SIZE = 32
INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 10
TOTAL_EPOCHS = INITIAL_EPOCHS + FINE_TUNE_EPOCHS

```
```
from tensorflow.keras.preprocessing import image_dataset_from_directory

train_dataset = image_dataset_from_directory(os.path.join(base_dir,"train"),
                                            batch_size=BATCH_SIZE,
                                            image_size=IMG_SIZE,
                                            shuffle=True,
                                            seed=seed,
                                            validation_split=0.2,
                                            subset="training")


val_dataset = image_dataset_from_directory(os.path.join(base_dir,"train"),
                                            batch_size=BATCH_SIZE,
                                            image_size=IMG_SIZE,
                                            shuffle=True,
                                            seed=seed,
                                            validation_split=0.2,
                                            subset="validation")


test_dataset = image_dataset_from_directory(os.path.join(base_dir,"test"),
                                            batch_size=BATCH_SIZE,
                                            image_size=IMG_SIZE,
                                            shuffle=True,
                                            seed=seed)


```


# 6. augment the data

```
# craete a tensorflow layer for augmentation
data_augmentation = tf.keras.models.Sequential([
                                    tf.keras.layers.RandomFlip('horizontal'),
                                    tf.keras.layers.RandomRotation(0.2),
                                    tf.keras.layers.RandomZoom(0.2)
                                            ])
```

# 7. Preview the preprocessed dataset.

```
for image, _ in train_dataset.take(1):
    plt.figure(figsize=(10, 10))
    first_image = image[0]
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
        plt.imshow(augmented_image[0] / 255)
        plt.axis('off')

```
////////////////////////////////add image

# 8. configure dataset for performance

```
train_dataset = train_dataset.shuffle(500).prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
```

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
