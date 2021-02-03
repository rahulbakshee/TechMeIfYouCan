![nn]({{ '/images/2021-02-03-edge_main.png' | relative_url }})
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
