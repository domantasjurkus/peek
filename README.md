# Peek  
Peek is a proof-of-concept project that investigates how manufacturing quality assurance can be inspected using computer vision.

![Control and query image feature matching](/img/splash.jpg?raw=true "Feature matching")

### Requirements  
* Python 2.7.13  
* OpenCV 2.4.13.2 (built-in SIFT and SURF feature detection algorithms)  

### Running Example  
1. Clone this repo  
2. Run `python main.py` from a command line - a sample control and query image will be compared, with a quality assurance score printed in the console.

### Approach  
For an arbitrary product image, a quality confidence score is computed by the following steps:  
1. Control and query images are loaded.  
2. The query image is scaled, rotated and aligned to the control image.  
3. Images are blurred and smoothed to minimise misalignment.  
3. Both images are inspected for differences in shape and contour.  
4. A final confidence score is produced following the observed differences.  

The alignment step uses scale-invariant features (SIFT) for detection and matching of similar regions between the two images, producing a transformation matrix that allows warping the query image in alignment with the control image. After comarison between SIFT and SURF matchers, no significant difference was apparent in choice of matching algorithm.  

Image difference is (currently) computed by the net brightness difference. This approach is sensitive to misalignment, background and lighting variation and thus is likely to be updated in the future. Edge detection and difference was considered, but it also suffers from poor alignment.  

### Todos  
`*` Update warping to rotate the query image any number of degrees  
`*` Update main executable to include command line arguments
`*` Add tests with good and bad query image samples

Project by Domantas Jurkus
