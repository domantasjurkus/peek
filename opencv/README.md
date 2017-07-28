## Computer Vision approach

This approach uses an older Opencv 2.4.13.2 (Python 2.7) version which has SIFT and SURF operators part of the core package.

Possible steps:

`*` Detect scale-invariant features (SIFT) of a sample control coaster and a target coaster  
`*` Align both coasters  
`*` Calculate edge difference  
`*` Smooth differnece in case alignment is not optimal  
`*` Represent edge diference in a numerical format  

`+` Single test sample required  
`+` Possibly generalisable to any arbitrary shape/product  
`-` Alignment/rotation can be tricky/computationally expensive  


#### Log
`27 Jul`  
Found a good example of SIFT, added it in `find_and_align`  

`28 Jul`  
Cleanup, move old scripts into `old`