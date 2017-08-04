# CS Sideproject - Manufacturing Quality Assurance

The purpouse of this project is to explore how product manufacturing quality can be ensured. To keep the project withing a manageable scope, the quality of Cooper Software coasters shall be inspected. After an initial investigation into the possible technologies, two methods of carrying out the project emerged: using computer vision and/or machine learning.  

The computer vision approad can be found in `opencv`.  
The machine learning approad is currently gitignored due to the large directory size.

## Computer Vision approach

This approach uses an older Opencv 2.4.13.2 (Python 2.7) version which has SIFT and SURF operators part of the core package.

`*` Detect scale-invariant features (SIFT) of a sample control coaster and a target coaster  
`*` Align both coasters  
`*` Calculate edge difference  
`*` Smooth differnece in case alignment is not optimal  
`*` Represent edge diference in a numerical format  

`+` Single test sample required  
`+` Possibly generalisable to any arbitrary shape/product  
`-` Alignment/rotation can be tricky/computationally expensive  


#### Log `2017`
`27 Jul`  
Found a good example of SIFT, added it in `find_and_align`  

`28 Jul`  
Cleanup, move old scripts into `old`.  
Tried comparing the SIFT and SURF matchers - no significant difference was apparent.  
The query coaster is found in the query image - now need to crop out and align the coaster with the sample image.

`30 Jul`  
Updated `find_and_align` to warp the query coaster to be aligned with the control coaster. Current approach assumes that the angle is no bigger than a few degrees, otherwise the detected coasted may be flipped.

Initial comparison using Canny edge detection picks up a lot of noise that is not actual 'damage' to the product. Will need to adjust the thresholds or consider another method of comparison