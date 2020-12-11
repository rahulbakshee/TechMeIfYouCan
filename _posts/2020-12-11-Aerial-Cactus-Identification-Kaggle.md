![nn]({{ '/images/2020-12-11-cactus.png' | relative_url }})
{: style="width: 600px; max-width: 100%;"}
*[credits](https://unsplash.com/photos/3guU1kCxxy0)*

{: class="table-of-content"}
* TOC
{:toc}

---
Hello World !!! 

In my quest of learning **`Computer vision`** I am starting with small datasets and lesser complex problems.So, today I am trying my hands on a kaggle problem which is basically an **`Image Classification`** problem from [Kaggle](https://www.kaggle.com/c/aerial-cactus-identification/)

## Description
To assess the impact of climate change on Earth's flora and fauna, it is vital to quantify how human activities such as logging, mining, and agriculture are impacting our protected natural areas. Researchers in Mexico have created the VIGIA project, which aims to build a system for autonomous surveillance of protected areas. A first step in such an effort is the ability to recognize the vegetation inside the protected areas. In this competition, you are tasked with creation of an algorithm that can identify a specific type of cactus in aerial imagery.


## Evaluation
Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target.

## Data
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a columnar cactus (Neobuxbaumia tetetzo). Kaggle has resized the images from the original dataset to make them uniform in size. The file name of an image corresponds to its id.

We must create a classifier capable of predicting whether an images contains a cactus.

Files
- train/ - the training set images
- test/ - the test set images (you must predict the labels of these)
- train.csv - the training set labels, indicates whether the image has a cactus (has_cactus = 1)
- sample_submission.csv - a sample submission file in the correct format


## Approach

### load the data
We have been given two folders for train and test images which contain images for both positive and negative classes. First we load the `.csv` file which has the IDs of images and their labels.

### visualize the data
The first thing to do in Machine Learning is to understand the data yourself before fitting any model. It is better to understand the data and have a good intuition about the features(e.g. in case of tabular data) which may help in curve fitting and later interpreting the model. In our case we will pick randomly few samples and see the distribution of positive(image has catus) and negative(image does not have cactus) classes.

```
fig,ax = plt.subplots(2,5, figsize=(10,7))

for i, index in enumerate(train_df[train_df["has_cactus"] == 1]["id"][:5]):
    path = os.path.join(train_dir, index)
    img = load_img(path)
    ax[0,i].set_title("Cactus")
    ax[0,i].imshow(img)
    
for i, index in enumerate(train_df[train_df["has_cactus"] == 0]["id"][:5]):
    path = os.path.join(train_dir, index)
    img = load_img(path)
    ax[1,i].set_title("No Cactus")
    ax[1,i].imshow(img)
```
![nn]({{ '/images/2020-12-11-cactus-distribution.png' | relative_url }})
{: style="width: 600px; max-width: 100%;"}


### split the data into train and val
Next we will create a validation split to evaluate our training on train dataset.
```
# convert target variable into 'str' type to avoid below error
# TypeError: If class_mode="binary", y_col="has_cactus" column values must be strings.
train_df["has_cactus"] = train_df["has_cactus"].astype('str')

print("before split train shape", train_df.shape)
train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df["has_cactus"], shuffle=True, random_state=seed)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

print("after split train/val shape", train_df.shape, val_df.shape)
```


### prepare data for modeling
Loading the data using TensorFLow/Keras API from a folder where images of all the classes are clubbed into one folder. I mean there is no subfolder for each class. So, we will be using  `ImageDataGenerator` class which generates batches of tensor images with real-time data augmentation. From this class we will be using `flow_from_dataframe` method which will help us map image IDs to their respective images from the directory and generates batches of images.
We will only be rescaling and not perform any other augmentation on images.

```
img_width, img_height = 32,32
target_size = (img_width, img_height)
batch_size = 32

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(dataframe=train_df,
                                                   directory=train_dir,
                                                   x_col='id',
                                                   y_col='has_cactus',
                                                   target_size=target_size,
                                                    class_mode='binary',
                                                   batch_size=batch_size,
                                                   seed = seed)
val_generator = val_datagen.flow_from_dataframe(dataframe=val_df,
                                                   directory=train_dir,
                                                   x_col='id',
                                                   y_col='has_cactus',
                                                   target_size=target_size,
                                                    class_mode='binary',
                                                   batch_size=batch_size,
                                                   seed = seed)
```
### creating a Convolutional Neural Networks(CNN) model from scratch
### visualize learning curve
### use pretrained model
### visualize learning curve
### prediction on test data

## Learnings
How to load the data using TensorFLow/Keras from a folder of all images.
How to visualise the data.
How to prepare data for augmentation.
How to write CNN model from scratch using functional API of keras Model.
How to load and use pre-trained CNN models for finetuning.
How to visualize accuracy/loss learnign curves for train/val data.




