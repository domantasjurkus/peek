# Peek  
Peek is a proof-of-concept project that investigates how manufacturing quality assurance can be inspected using computer vision.  

![Control and query image feature matching](/img/splash.jpg?raw=true "Feature matching")

### Requirements  
* Python 2.7.13  
* OpenCV 2.4.13.2 (built-in SIFT and SURF feature detection algorithms)  

### Running Examples
`python main.py` - align, subtract and produce score for sample images.  
`python src/align.py` - alignment example.  
`python src/difference.py` - difference example.  

### Approach  
For an arbitrary product image, a quality confidence score is computed by the following steps:  
1. Control and query images are loaded (control image has to be cropped to the very edges of the product)  
2. The query image is scaled, rotated and aligned to the control image  
3. Images are blurred and smoothed to minimise misalignment  
3. Images are subtracted and thresholded to produce a binary "damage map"
4. A final confidence score is output based on damage map

#### Assumptions  
`*` Images should ideally be captured under similar lighting conditions  
`*` The camera should be fixed 90 degrees vertically above the products for inspection  
`*` The product is not rotated from the query image more than 45 degrees

### Nice-to-haves
`*` Automatically crop control image  
`*` Change feature detection algorithm from SIFT (patented) to BRISK or some other free alternative  
`*` Update warping to rotate the query image any number of degrees  
`*` Detect main geometric shape of the product to remove background  
`*` Automatically compute blur and threshold values (color histogram comparison?)  

Project by Domantas Jurkus
