![nn]({{ '/images/2022-01-16-tf_logo.png' | relative_url }})
{: style="width: 600px; max-width: 100%;"}

Most of the frameworks these days provide easy ways of loading, preprocessing and pipelining of data. Today, we will discuss various ways we can load data using TensorFlow and Keras. This is the first step followed by `data augmentation` and `preprocessing`.

> try code yourself at this **[colab](https://colab.research.google.com/drive/1k-MocSgk8OoaNQqtkbsjDLdBCfeJ7kMV?usp=sharing)**

# 1. image_dataset_from_directory
A high-level Keras preprocessing utility to read a directory of images on disk. Data is expected to be in a directory structure where each subdirectory represents a class.
 
 ```
main_directory/
...class_a/
......a_image_1.jpg
......a_image_2.jpg
...class_b/
......b_image_1.jpg
......b_image_2.jpg
```

Calling `image_dataset_from_directory(main_directory, labels='inferred')` will return a `tf.data.Dataset` that yields batches of images from the subdirectories `class_a` and  `class_b`, together with labels 0 and 1 (0 corresponding to `class_a` and 1 corresponding to `class_b`).


In case of more than two subdirectories, the labels will be inferred and start from `0,1,2,3...` as this is a **multi-class classification** problem.

I found two ways to utilize this either from `tf.keras.utils.image_dataset_from_directory` or `tf.keras.preprocessing.image_dataset_from_directory`

```
batch_size = 32
img_height, img_width = 150, 150
seed = 42
import tensorflow as tf

# download raw data
import pathlib
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file(origin=dataset_url,
                                   fname='flower_photos',
                                   untar=True)
data_dir = pathlib.Path(data_dir)

# Load data off disc using a Keras utility
train_ds = tf.keras.utils.image_dataset_from_directory(			
                            data_dir,
                            validation_split=0.2,
                            subset="training",
                            seed=seed,
                            image_size=(img_height, img_width),
                            batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
                            data_dir,
                            validation_split=0.2,
                            subset="validation",
                            seed=seed,
                            image_size=(img_height, img_width),
                            batch_size=batch_size)


# Found 3670 files belonging to 5 classes.
# Using 2936 files for training.
# Found 3670 files belonging to 5 classes.
# Using 734 files for validation.
                            
```

# 2. tf.data
An API for input pipelines for finer control, where we can write our own pipeline using `tf.data` . We will create dataset by passing the directory and its contents to `tf.data.Dataset.list_files`. Here `list_files` expects glob patterns to be matched.

```
import os
import numpy as np
import tensorflow as tf

img_height, img_width = 150, 150
AUTOTUNE = tf.data.AUTOTUNE

# download raw data
import pathlib
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file(origin=dataset_url,
                                   fname='flower_photos',
                                   untar=True)
data_dir = pathlib.Path(data_dir)

# total number of images
image_count = len(list(data_dir.glob('*/*.jpg')))

list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'), shuffle=False)

val_size = int(image_count * 0.2)
train_ds = list_ds.skip(val_size)
val_ds = list_ds.take(val_size)

class_names = np.array(sorted([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"]))

print("Using {} files for training.".format(len(train_ds)))
print("Using {} files for validation.".format(len(val_ds)))

def get_label(file_path):
  # Convert the path to a list of path components
  parts = tf.strings.split(file_path, os.path.sep)
  # The second to last is the class-directory
  one_hot = parts[-2] == class_names
  # Integer encode the label
  return tf.argmax(one_hot)

def decode_img(img):
  # Convert the compressed string to a 3D uint8 tensor
  img = tf.io.decode_jpeg(img, channels=3)
  # Resize the image to the desired size
  return tf.image.resize(img, [img_height, img_width])

def process_path(file_path):
  label = get_label(file_path)
  # Load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label

# Use Dataset.map to create a dataset of image, label pairs:
# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)


# Using 2936 files for training.
# Using 734 files for validation.

```

# 3. tensorflow_datasets 
TensorFlow provides a large [catalog](https://www.tensorflow.org/datasets/catalog/overview) of easy-to-download datasets. Using `tfds.load` arguments such as `split` we can choose which split to read (e.g. 'train', ['train', 'test'], 'train[80%:]',...)


```
import tensorflow as tf
import tensorflow_datasets as tfds

(train_ds, val_ds), info = tfds.load(
                                    'tf_flowers',
                                    split=['train[:80%]', 'train[80%:]'],
                                    with_info=True,
                                    as_supervised=True,
                                    )

print("Using {} files for training.".format(len(train_ds)))
print("Using {} files for validation.".format(len(val_ds)))

# Using 2936 files for training.
# Using 734 files for validation.
```
This is the easiest of all but has limited(but growing) number of datasets.

Now it may happen that raw data is not according to the directory-format expected by above APIs. So, to rearrange the data according to our needs, we can use python modules such as `shutils` etc. and then feed it to TensorFlow APIs.

That's it for today. We discussed how to load data using TensorFlow and Keras. I will be back with next steps as to how to do `augmentation` , `preprocessing` and how to feed input to `Model`.

For an end-to-end `Deep Learning` flow please visit [github](https://rahulbakshee.github.io/iWriteHere/2022/02/08/Deep-Learning-Take-Home-Assignment.html) 

let us connect on [linkedin](https://www.linkedin.com/in/rahulbakshee/) and [twitter](https://twitter.com/rahulbakshee)

[linkedin article](https://www.linkedin.com/pulse/quick-ways-data-loading-tensorflow-rahul-bakshee)

Read [this article](https://rahulbakshee.github.io/iWriteHere/2022/01/16/Quick-ways-of-data-loading-in-TensorFlow.html) and [other articles](https://rahulbakshee.github.io/iWriteHere/) on `Machine Learning, Deep Learning and Computer Vision`.

References:

[tensorflow.org](https://www.tensorflow.org/tutorials/load_data/images)
