![nn]({{ '/images/2022-01-16-tf_logo.png' | relative_url }})
{: style="width: 600px; max-width: 100%;"}

{: class="table-of-content"}
* TOC
{:toc}

Most of the frameworks these days provide easy ways of loading, preprocessing and pipelining of data. Today, we will discuss various ways we can load data off-disc using TensorFlow and Keras. 

# 1. `image_dataset_from_directory` A high-level Keras preprocessing utility to read a directory of images on disk.
 Data is expected to be in a directory structure where each subdirectory represents a class.
 
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


> `image_dataset_from_directory` can be utilised via `tf.keras.utils.image_dataset_from_directory` or `tf.keras.preprocessing.image_dataset_from_directory`

```
import tensorflow as tf

# Load data off disc using a Keras utility
train_ds = tf.keras.utils.image_dataset_from_directory(			
                            main_directory,
                            validation_split=0.2,
                            subset="training",
                            seed=seed,
                            image_size=(img_height, img_width),
                            batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
                            main_directory,
                            validation_split=0.2,
                            subset="validation",
                            seed=seed,
                            image_size=(img_height, img_width),
                            batch_size=batch_size)
                            
```




### References:
[tensorflow.org](https://www.tensorflow.org/tutorials/load_data/images)
