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
### visualize the data
### prepare data for modeling
### model fitting
### visualize learning curve
### prediction on test data
