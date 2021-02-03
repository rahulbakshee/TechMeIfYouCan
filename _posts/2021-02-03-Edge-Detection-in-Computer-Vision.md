![edge detection]({{ '/images/2021-02-03-edge_main.png' | relative_url }})
{: style="width: 600px; max-width: 100%;"}
<span>Photo by <a href="https://unsplash.com/@danieljschwarz?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">Daniel J. Schwarz</a> on <a href="https://unsplash.com/s/photos/city-night?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">Unsplash</a></span>



# Edge Detection in Computer Vision

Edge detection is a basic but important task in Computer Vision. It is usually done to reduce the amount of data yet preserving the structural properties of the objects in the image. The popular tasks that use edge detection are lane detection and converting an image into a sketch.

Although there are a lot of algorithms for edge detection, today we are going to discuss Canny Edge Detection algorithm (by John F. Canny in 1986).


```
import numpy as np
import cv2
import matplotlib.pyplot as plt
%matplotlib inline
```
> **It is a multi-stage algorithm** :

### Step 1: Noise reduction using Gaussian blur :

A Gaussian filter is convolved over the image to remove noise and to prevent the assumption of noise as edges. The sudden change of intensity of pixels should not be considered as an edge therefore it is vital to first treat the input image with smoothing.


```
A typical Gaussian filter: 
                        [[2,  4,   5,  4,  2],
                        [ 4,  9,  12,  9,  4],
              (1 / 159).[ 5, 12,  15, 12,  5],
                        [ 4,  9,  12,  9,  4],
                        [ 2,  4,   5,  4,  2]]
```

#### example showing how to use Gaussian blur on an image

```
# input image
img = cv2.imread("plant2.jpg", 0)

# gaussian smoothing
blurred = cv2.GaussianBlur(img, (5, 5), 1.4) 

f, (ax1, ax2) = plt.subplots(1,2, figsize=(10,7))
ax1.imshow(img, cmap='gray')
ax1.title.set_text("Original image")
ax2.imshow(blurred, cmap='gray')
ax2.title.set_text("after Gaussian blur")
```
![gaussian blur]({{ '/images/2021-02-03-gaussian_blur.png' | relative_url }})
{: style="width: 600px; max-width: 100%;"}

### Step 2: Finding Gradients:


A gradient is nothing but the change in the intensity of the pixels of the image. After the image is smoothed, gradients are calculated along x and y direction using Sobel-operator. The gradient magnitudes (also known as the edge strengths) can then be determined as an Euclidean distance measure by applying the law of Pythagoras.

```
The Sobel kernels are convolved over the image to determine gradient magnitudes.

Along X direction Kx:  [[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]]
 

Along Y direction Ky:  [[1, 2, 1],
                        [0, 0, 0],
                        [-1,-2,-1]]
                        
gradient magnitude = sqrt(Gx^2 + Gy^2)
angle = arctan(|Gy| / |Gx|)
```
> example showing how to find gradients using Sobel operators

```
# input image
img = cv2.imread("plant2.jpg", 0)

sobelx = cv2.Sobel(np.float32(img), cv2.CV_64F, 1,0, ksize=5)
sobely = cv2.Sobel(np.float32(img), cv2.CV_64F, 0,1, ksize=5)

f, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(15,10))
ax1.imshow(img, cmap='gray')
ax1.title.set_text("Original image")
ax2.imshow(sobelx, cmap='gray')
ax2.title.set_text("Sobel filter along X axis")
ax3.imshow(sobely, cmap='gray')
ax3.title.set_text("Sobel filter along Y axis")

```
