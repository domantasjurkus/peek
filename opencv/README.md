## Computer Vision approach

OpenCV is a handy library that has many tools for image manipulation and analysis.

Possible steps:

`*` Detect scale-invariant features (SIFT) of a sample control coaster and a target coaster  
`*` Align both coasters  
`*` Calculate edge difference  
`*` Smooth differnece in case alignment is not optimal  
`*` Represent edge diference in a numerical format  

`+` Single test sample required  
`+` Possibly generalisable to any arbitrary shape/product  
`-` Alignment/rotation can be tricky/computationally expensive  

This approach uses an older Opencv 2.4.13.2 (Python 2.7) version which has SIFT and SURF operators part of the core package.